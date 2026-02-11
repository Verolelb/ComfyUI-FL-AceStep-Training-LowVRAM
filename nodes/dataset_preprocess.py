"""
ACE-Step Dataset Preprocess Node

Converts labeled samples to tensor files for training.
Uses native ComfyUI MODEL type for the ACE-Step model.

Performance-optimized to match the sdbds reference implementation:
- Models loaded once, kept on GPU for entire loop
- torch.inference_mode() wraps entire loop
- Cached refer_audio tensors and resampler objects
- non_blocking GPU transfers
- Periodic cache clearing
"""

import json
import logging
import random
from pathlib import Path

import torch

try:
    import comfy.model_management as model_management
except ImportError:
    model_management = None

try:
    from comfy.utils import ProgressBar
except ImportError:
    ProgressBar = None

from ..modules.acestep_model import (
    is_acestep_model,
    get_silence_latent,
    get_acestep_encoder,
)
from ..modules.audio_utils import load_audio, vae_encode_direct

logger = logging.getLogger("FL_AceStep_Training")

# SFT generation prompt template (from ACE-Step constants)
SFT_GEN_PROMPT = """# Instruction
{}

# Caption
{}

# Metas
{}<|endoftext|>
"""

DEFAULT_DIT_INSTRUCTION = "Fill the audio semantic mask based on the given conditions:"

# Cache for refer_audio placeholder tensors (avoid GPU alloc per sample)
_REFER_AUDIO_CACHE: dict = {}


def _get_refer_audio_tensors(device, dtype):
    """Get cached refer_audio placeholder tensors for text2music (no reference audio)."""
    cache_key = (device, dtype)
    if cache_key not in _REFER_AUDIO_CACHE:
        _REFER_AUDIO_CACHE[cache_key] = (
            torch.zeros(1, 1, 64, device=device, dtype=dtype),
            torch.zeros(1, device=device, dtype=torch.long),
        )
    refer_audio_hidden, refer_audio_order_mask = _REFER_AUDIO_CACHE[cache_key]
    # Reset in-place (cheap) rather than allocating new tensors
    refer_audio_hidden.zero_()
    refer_audio_order_mask.zero_()
    return refer_audio_hidden, refer_audio_order_mask


def encode_text_and_lyrics(clip, text: str, lyrics: str, device, dtype):
    """
    Encode text and lyrics using ComfyUI's native CLIP pipeline.

    For ACE-Step 1.5, this uses the Qwen3 model:
    - Text: Full forward pass -> last_hidden_state
    - Lyrics: Layer 0 output only (shallow embedding)

    IMPORTANT: Must use return_dict=True to get lyrics embeddings.
    """
    tokens = clip.tokenize(text, lyrics=lyrics)
    result = clip.encode_from_tokens(tokens, return_pooled=True, return_dict=True)

    text_hidden_states = result["cond"].to(device=device, dtype=dtype, non_blocking=True)
    text_attention_mask = torch.ones(
        text_hidden_states.shape[:2], device=device, dtype=dtype
    )

    lyric_hidden_states = result.get("conditioning_lyrics", None)
    if lyric_hidden_states is not None:
        lyric_hidden_states = lyric_hidden_states.to(device=device, dtype=dtype, non_blocking=True)
        if lyric_hidden_states.dim() == 2:
            lyric_hidden_states = lyric_hidden_states.unsqueeze(0)
        lyric_attention_mask = torch.ones(
            lyric_hidden_states.shape[:2], device=device, dtype=dtype
        )
    else:
        lyric_hidden_states = torch.zeros(1, 1, text_hidden_states.shape[-1],
                                          device=device, dtype=dtype)
        lyric_attention_mask = torch.zeros(1, 1, device=device, dtype=dtype)

    return text_hidden_states, text_attention_mask, lyric_hidden_states, lyric_attention_mask


class FL_AceStep_PreprocessDataset:
    """
    Preprocess Dataset

    Converts labeled audio samples to preprocessed tensor files for training.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "dataset": ("ACESTEP_DATASET",),
                "model": ("MODEL",),
                "vae": ("VAE",),
                "clip": ("CLIP",),
                "output_dir": ("STRING", {
                    "default": "./output/acestep/datasets",
                    "multiline": False,
                }),
            },
            "optional": {
                "max_duration": ("FLOAT", {
                    "default": 240.0,
                    "min": 10.0,
                    "max": 600.0,
                    "step": 10.0,
                }),
                "genre_ratio": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 100,
                    "step": 5,
                }),
            }
        }

    RETURN_TYPES = ("STRING", "INT", "STRING")
    RETURN_NAMES = ("output_path", "sample_count", "status")
    FUNCTION = "preprocess"
    CATEGORY = "FL AceStep/Dataset"
    OUTPUT_NODE = True

    def preprocess(
        self,
        dataset,
        model,
        vae,
        clip,
        output_dir,
        max_duration=240.0,
        genre_ratio=0
    ):
        """Preprocess the dataset to tensor files."""
        samples = dataset.samples
        if not samples:
            return (output_dir, 0, "No samples to preprocess")

        if not is_acestep_model(model):
            return (output_dir, 0, "Error: Model is not an ACE-Step model")

        labeled_samples = [s for s in samples if s.labeled or s.caption]
        if not labeled_samples:
            return (output_dir, 0, "No labeled samples to preprocess")

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # --- Setup: Load all models to GPU ONCE ---
        device = model_management.get_torch_device() if model_management else torch.device('cuda')

        # Load VAE to GPU via ComfyUI's model management (one-time cost)
        # Use a single vae.encode() call to trigger the loading, then use the
        # underlying model directly for the rest of the loop
        vae_model = vae.first_stage_model
        vae_dtype = vae.vae_dtype
        if model_management:
            model_management.load_models_gpu(
                [vae.patcher],
                force_full_load=getattr(vae, 'disable_offload', False)
            )
        logger.info(f"VAE loaded: dtype={vae_dtype}, device={device}")

        # Load CLIP/text encoder to GPU via ComfyUI's model management (one-time cost)
        # Can't use clip.load_model() with empty tokens — ACE-Step 1.5's
        # memory_estimation_function expects tokenized input with lm_metadata.
        # Call load_models_gpu directly on the patcher instead.
        if model_management:
            model_management.load_models_gpu([clip.patcher])
        logger.info("CLIP/text encoder loaded")

        # Get condition encoder and move to GPU
        condition_encoder = get_acestep_encoder(model)
        enc_param = next(condition_encoder.parameters())
        enc_dtype = enc_param.dtype
        if enc_param.device != device:
            condition_encoder.to(device)
        logger.info(f"Condition encoder: dtype={enc_dtype}, device={device}")

        # Get silence latent
        silence_latent = get_silence_latent(model)
        if silence_latent is None:
            silence_latent = torch.zeros(1, 750, 64, device=device, dtype=enc_dtype)

        # Progress bar
        pbar = ProgressBar(len(labeled_samples)) if ProgressBar else None

        processed_count = 0
        manifest = []
        errors = []

        logger.info(f"Preprocessing {len(labeled_samples)} samples to {output_dir}")

        # --- Main loop: inference_mode for entire batch ---
        with torch.inference_mode():
            for i, sample in enumerate(labeled_samples):
                try:
                    tensor_data = self._preprocess_sample(
                        sample=sample,
                        vae_model=vae_model,
                        clip=clip,
                        condition_encoder=condition_encoder,
                        silence_latent=silence_latent,
                        max_duration=max_duration,
                        genre_ratio=genre_ratio,
                        custom_tag=dataset.metadata.custom_tag,
                        tag_position=dataset.metadata.tag_position,
                        device=device,
                        vae_dtype=vae_dtype,
                        enc_dtype=enc_dtype,
                    )

                    if tensor_data is None:
                        continue

                    # Save tensor file
                    tensor_filename = f"{sample.id}.pt"
                    tensor_path = output_path / tensor_filename
                    torch.save(tensor_data, tensor_path)

                    manifest.append({
                        "id": sample.id,
                        "filename": tensor_filename,
                        "audio_path": sample.audio_path,
                        "caption": sample.caption,
                        "duration": sample.duration,
                        "bpm": sample.bpm,
                        "keyscale": sample.keyscale,
                        "is_instrumental": sample.is_instrumental,
                    })

                    processed_count += 1
                    logger.info(
                        f"[{processed_count}/{len(labeled_samples)}] "
                        f"{sample.filename} ({sample.duration:.0f}s)"
                    )

                except Exception as e:
                    error_msg = f"Error processing sample {sample.id}: {str(e)}"
                    logger.warning(error_msg)
                    errors.append(error_msg)

                if pbar:
                    pbar.update(1)

                # Periodic GPU cache clearing (every 8 samples, matching sdbds)
                if device.type == "cuda" and (i + 1) % 8 == 0:
                    torch.cuda.empty_cache()

        # Save manifest
        manifest_path = output_path / "manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump({
                "samples": manifest,
                "metadata": {
                    "total_samples": processed_count,
                    "max_duration": max_duration,
                    "genre_ratio": genre_ratio,
                    "custom_tag": dataset.metadata.custom_tag,
                }
            }, f, indent=2, ensure_ascii=False)

        status = f"Preprocessed {processed_count}/{len(labeled_samples)} samples"
        if errors:
            status += f" ({len(errors)} errors)"

        logger.info(status)
        return (str(output_path), processed_count, status)

    def _preprocess_sample(
        self,
        sample,
        vae_model,
        clip,
        condition_encoder,
        silence_latent,
        max_duration,
        genre_ratio,
        custom_tag,
        tag_position,
        device,
        vae_dtype,
        enc_dtype,
    ):
        """Preprocess a single sample to tensor data."""
        # Step 1: Load audio (resampler is cached in audio_utils)
        waveform, sr = load_audio(sample.audio_path, max_duration=max_duration)

        # Step 2: VAE encode — call the underlying model directly (already on GPU)
        audio = waveform.unsqueeze(0).to(device=device, dtype=vae_dtype, non_blocking=True)
        target_latents = vae_encode_direct(vae_model, audio, device, vae_dtype)
        del audio  # Free GPU memory immediately

        latent_length = target_latents.shape[1]
        attention_mask = torch.ones(1, latent_length, device=device)

        # Step 3: Build caption with custom tag
        caption = sample.caption
        if custom_tag:
            if tag_position == "prepend":
                caption = f"{custom_tag}, {caption}"
            elif tag_position == "append":
                caption = f"{caption}, {custom_tag}"
            elif tag_position == "replace":
                caption = custom_tag

        use_genre = random.randint(0, 100) < genre_ratio and sample.genre
        text_content = sample.genre if use_genre else caption

        # Build metadata string (always include all fields, N/A for missing)
        metas_str = (
            f"- bpm: {sample.bpm if sample.bpm else 'N/A'}\n"
            f"- timesignature: {sample.timesignature if sample.timesignature else 'N/A'}\n"
            f"- keyscale: {sample.keyscale if sample.keyscale else 'N/A'}\n"
            f"- duration: {int(sample.duration)} seconds\n"
        )

        text_prompt = SFT_GEN_PROMPT.format(
            DEFAULT_DIT_INSTRUCTION,
            text_content,
            metas_str
        )

        # Step 4: Encode text and lyrics via ComfyUI CLIP
        lyrics = sample.lyrics if sample.lyrics else "[Instrumental]"
        text_hidden_states, text_attention_mask, lyric_hidden_states, lyric_attention_mask = \
            encode_text_and_lyrics(clip, text_prompt, lyrics, device, enc_dtype)

        # Step 5: Run condition encoder to merge text+lyrics+timbre
        refer_audio_hidden, refer_audio_order_mask = _get_refer_audio_tensors(device, enc_dtype)

        encoder_hidden_states, encoder_attention_mask = condition_encoder(
            text_hidden_states=text_hidden_states,
            text_attention_mask=text_attention_mask,
            lyric_hidden_states=lyric_hidden_states,
            lyric_attention_mask=lyric_attention_mask,
            refer_audio_acoustic_hidden_states_packed=refer_audio_hidden,
            refer_audio_order_mask=refer_audio_order_mask,
        )

        # Step 6: Build context latents [1, T, 128] = [silence(64), chunk_mask(64)]
        context_latents = torch.empty((1, latent_length, 128), device=device, dtype=enc_dtype)

        # Fill silence latent into first 64 channels
        src = silence_latent.to(dtype=enc_dtype)
        src_len = src.shape[1]
        take = min(latent_length, src_len)
        context_latents[:, :take, :64] = src[:, :take, :]
        if take < latent_length:
            # Tile silence to fill remaining length
            remaining = latent_length - take
            pos = take
            while remaining > 0:
                chunk = min(remaining, src_len)
                context_latents[:, pos:pos + chunk, :64] = src[:, :chunk, :]
                pos += chunk
                remaining -= chunk

        # Last 64 channels = 1 (chunk mask: generate all)
        context_latents[:, :, 64:] = 1

        # Step 7: Prepare output (squeeze batch dim, move to CPU for storage)
        tensor_data = {
            "target_latents": target_latents.squeeze(0).cpu(),
            "attention_mask": attention_mask.squeeze(0).cpu(),
            "encoder_hidden_states": encoder_hidden_states.squeeze(0).cpu(),
            "encoder_attention_mask": encoder_attention_mask.squeeze(0).cpu(),
            "context_latents": context_latents.squeeze(0).cpu(),
            "metadata": {
                "audio_path": sample.audio_path,
                "filename": sample.filename,
                "caption": caption,
                "lyrics": lyrics,
                "duration": sample.duration,
                "bpm": sample.bpm,
                "keyscale": sample.keyscale,
                "timesignature": sample.timesignature,
                "language": sample.language,
                "is_instrumental": sample.is_instrumental,
            }
        }

        return tensor_data
