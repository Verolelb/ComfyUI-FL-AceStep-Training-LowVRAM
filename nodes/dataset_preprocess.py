"""
ACE-Step Dataset Preprocess Node

Converts labeled samples to tensor files for training.
Uses native ComfyUI MODEL type for the ACE-Step model.
"""

import json
import logging
from pathlib import Path

import torch

try:
    from comfy.utils import ProgressBar
except ImportError:
    ProgressBar = None

from ..modules.acestep_model import (
    is_acestep_model,
    get_silence_latent,
    get_acestep_encoder,
)
from ..modules.audio_utils import load_audio, vae_encode

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


def encode_text_and_lyrics(clip, text: str, lyrics: str, device, dtype):
    """
    Encode text and lyrics using ComfyUI's native CLIP pipeline.

    For ACE-Step 1.5, this uses the Qwen3 model:
    - Text: Full forward pass → last_hidden_state
    - Lyrics: Layer 0 output only (shallow embedding)

    Args:
        clip: ComfyUI CLIP object (ACE-Step text encoder)
        text: Caption/prompt text
        lyrics: Lyrics text (or "[Instrumental]")
        device: Device to use
        dtype: Data type to use

    Returns:
        Tuple of (text_hidden_states, text_attention_mask,
                  lyric_hidden_states, lyric_attention_mask)
    """
    # Tokenize text and lyrics together — ComfyUI handles the separation
    tokens = clip.tokenize(text, lyrics=lyrics)

    # Encode with return_dict to get both text and lyrics embeddings
    result = clip.encode_from_tokens(tokens, return_pooled=True, return_dict=True)

    # Text hidden states (the main conditioning)
    text_hidden_states = result["cond"].to(device).to(dtype)
    text_attention_mask = torch.ones(
        text_hidden_states.shape[:2], device=device, dtype=dtype
    )

    # Lyrics embeddings — ComfyUI returns these separately
    lyric_hidden_states = result.get("conditioning_lyrics", None)
    if lyric_hidden_states is not None:
        lyric_hidden_states = lyric_hidden_states.to(device).to(dtype)
        if lyric_hidden_states.dim() == 2:
            lyric_hidden_states = lyric_hidden_states.unsqueeze(0)
        lyric_attention_mask = torch.ones(
            lyric_hidden_states.shape[:2], device=device, dtype=dtype
        )
    else:
        # Fallback: empty lyrics
        lyric_hidden_states = torch.zeros(1, 1, text_hidden_states.shape[-1],
                                          device=device, dtype=dtype)
        lyric_attention_mask = torch.zeros(1, 1, device=device, dtype=dtype)

    return text_hidden_states, text_attention_mask, lyric_hidden_states, lyric_attention_mask


class FL_AceStep_PreprocessDataset:
    """
    Preprocess Dataset

    Converts labeled audio samples to preprocessed tensor files for training.
    This is a critical step that:
    - Encodes audio to VAE latents
    - Encodes captions/metadata and lyrics to combined embeddings via condition encoder
    - Creates context latents with silence patterns
    - Saves everything as .pt files

    Requires:
    - dataset: Labeled dataset from previous nodes
    - model: ACE-Step MODEL (purple connection) for silence latent
    - vae: ACE-Step VAE for audio encoding
    - clip: ACE-Step CLIP for text encoding

    The output directory will contain:
    - Individual .pt files for each sample
    - manifest.json listing all samples
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "dataset": ("ACESTEP_DATASET",),
                "model": ("MODEL",),  # Native ComfyUI MODEL type (purple connection)
                "vae": ("VAE",),  # Native ComfyUI VAE type (from checkpoint)
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
        model,  # ComfyUI MODEL (ModelPatcher)
        vae,
        clip,
        output_dir,
        max_duration=240.0,
        genre_ratio=0
    ):
        """Preprocess the dataset to tensor files."""
        logger.info(f"Preprocessing dataset to {output_dir}")

        samples = dataset.samples
        if not samples:
            return (output_dir, 0, "No samples to preprocess")

        # Verify this is an ACE-Step model
        if not is_acestep_model(model):
            return (output_dir, 0, "Error: Model is not an ACE-Step model")

        # Filter to labeled samples only
        labeled_samples = [s for s in samples if s.labeled or s.caption]
        if not labeled_samples:
            return (output_dir, 0, "No labeled samples to preprocess")

        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Get device and dtype from VAE's actual parameters
        # ComfyUI VAE wrapper has first_stage_model which is the actual model
        vae_model = vae.first_stage_model if hasattr(vae, 'first_stage_model') else vae
        try:
            first_param = next(vae_model.parameters())
            device = first_param.device
            dtype = first_param.dtype
            logger.info(f"VAE dtype: {dtype}, device: {device}")
        except StopIteration:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            dtype = torch.float16
            logger.warning(f"Could not detect VAE dtype/device, using defaults: {dtype}, {device}")

        # Get silence latent from MODEL via model_options
        silence_latent = get_silence_latent(model)
        if silence_latent is None:
            logger.warning("Silence latent not found in model_options, creating default")
            # Create a default silence latent (zeros)
            silence_latent = torch.zeros(1, 750, 64, device=device, dtype=dtype)

        # Get the condition encoder from the DiT model
        # This merges text + lyrics into combined encoder_hidden_states
        condition_encoder = get_acestep_encoder(model)
        # The encoder may be on CPU (ComfyUI model management) — move to GPU
        # and detect its native dtype (typically bfloat16, different from VAE's float16)
        enc_param = next(condition_encoder.parameters())
        enc_dtype = enc_param.dtype
        logger.info(f"Condition encoder dtype: {enc_dtype}, device: {enc_param.device}")
        if enc_param.device != device:
            logger.info(f"Moving condition encoder to {device}")
            condition_encoder.to(device)
        logger.info("Got condition encoder for text+lyrics merging")

        # Progress bar
        pbar = ProgressBar(len(labeled_samples)) if ProgressBar else None

        processed_count = 0
        manifest = []
        errors = []

        for i, sample in enumerate(labeled_samples):
            try:
                # Process sample
                tensor_data = self._preprocess_sample(
                    sample=sample,
                    vae=vae,
                    clip=clip,
                    condition_encoder=condition_encoder,
                    silence_latent=silence_latent,
                    max_duration=max_duration,
                    genre_ratio=genre_ratio,
                    custom_tag=dataset.metadata.custom_tag,
                    tag_position=dataset.metadata.tag_position,
                    device=device,
                    dtype=dtype,
                )

                if tensor_data is None:
                    continue

                # Save tensor file
                tensor_filename = f"{sample.id}.pt"
                tensor_path = output_path / tensor_filename
                torch.save(tensor_data, tensor_path)

                # Add to manifest
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

            except Exception as e:
                error_msg = f"Error processing sample {sample.id}: {str(e)}"
                logger.warning(error_msg)
                errors.append(error_msg)

            if pbar:
                pbar.update(1)

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

        # Build status
        status = f"Preprocessed {processed_count}/{len(labeled_samples)} samples"
        if errors:
            status += f" ({len(errors)} errors)"

        logger.info(status)
        logger.info(f"Saved to {output_path}")

        return (str(output_path), processed_count, status)

    def _preprocess_sample(
        self,
        sample,
        vae,
        clip,
        condition_encoder,
        silence_latent,
        max_duration,
        genre_ratio,
        custom_tag,
        tag_position,
        device,
        dtype,
    ):
        """Preprocess a single sample to tensor data."""
        # Load audio using shared utility (resamples to 48kHz stereo)
        waveform, sr = load_audio(sample.audio_path, max_duration=max_duration)

        # Add batch dimension
        waveform = waveform.unsqueeze(0)

        # Debug: log waveform shape before VAE encoding
        logger.info(f"Waveform shape before VAE: {waveform.shape}, dtype: {waveform.dtype}")

        # Encode to VAE latents using tiled encoding for long audio
        target_latents = vae_encode(vae, waveform)  # [1, T, 64]

        logger.info(f"VAE latents shape: {target_latents.shape}")
        latent_length = target_latents.shape[1]

        # Create attention mask (all valid)
        attention_mask = torch.ones(1, latent_length, device=device)

        # Build caption with custom tag
        caption = sample.caption
        if custom_tag:
            if tag_position == "prepend":
                caption = f"{custom_tag}, {caption}"
            elif tag_position == "append":
                caption = f"{caption}, {custom_tag}"
            elif tag_position == "replace":
                caption = custom_tag

        # Determine whether to use genre or caption
        import random
        use_genre = random.randint(0, 100) < genre_ratio and sample.genre

        if use_genre:
            text_content = sample.genre
        else:
            text_content = caption

        # Build metadata string (always include all fields, N/A for missing — matches official)
        metas_str = (
            f"- bpm: {sample.bpm if sample.bpm else 'N/A'}\n"
            f"- timesignature: {sample.timesignature if sample.timesignature else 'N/A'}\n"
            f"- keyscale: {sample.keyscale if sample.keyscale else 'N/A'}\n"
            f"- duration: {int(sample.duration)} seconds\n"
        )

        # Build SFT prompt
        text_prompt = SFT_GEN_PROMPT.format(
            DEFAULT_DIT_INSTRUCTION,
            text_content,
            metas_str
        )

        # Encode text and lyrics using ComfyUI's native CLIP pipeline
        lyrics = sample.lyrics if sample.lyrics else "[Instrumental]"
        logger.info(f"Encoding text (len={len(text_prompt)}) + lyrics (len={len(lyrics)})")

        with torch.no_grad():
            # Step 1: Encode text and lyrics via ComfyUI CLIP
            # For ACE-Step 1.5: text = full Qwen3 forward, lyrics = layer 0 only
            enc_dtype = next(condition_encoder.parameters()).dtype
            text_hidden_states, text_attention_mask, lyric_hidden_states, lyric_attention_mask = \
                encode_text_and_lyrics(clip, text_prompt, lyrics, device, enc_dtype)
            logger.info(
                f"Text shape: {text_hidden_states.shape}, lyrics shape: {lyric_hidden_states.shape}, "
                f"dtype: {text_hidden_states.dtype}"
            )

            # Step 2: Run through condition encoder to merge text+lyrics+timbre
            # Pass zero tensors for refer_audio (not None) — matches official implementation
            refer_audio_hidden = torch.zeros(1, 1, 64, device=device, dtype=enc_dtype)
            refer_audio_order_mask = torch.zeros(1, device=device, dtype=torch.long)

            encoder_hidden_states, encoder_attention_mask = condition_encoder(
                text_hidden_states=text_hidden_states,
                text_attention_mask=text_attention_mask,
                lyric_hidden_states=lyric_hidden_states,
                lyric_attention_mask=lyric_attention_mask,
                refer_audio_acoustic_hidden_states_packed=refer_audio_hidden,
                refer_audio_order_mask=refer_audio_order_mask,
            )
        logger.info(f"Condition encoder output shape: {encoder_hidden_states.shape}")

        # Create context latents
        # Context = [silence_latent, chunk_masks]
        # For text-to-music, we use silence as source and mask=1 (generate all)

        # Expand silence latent to match target length
        if silence_latent.shape[1] < latent_length:
            # Repeat silence latent
            repeats = (latent_length + silence_latent.shape[1] - 1) // silence_latent.shape[1]
            src_latents = silence_latent.repeat(1, repeats, 1)[:, :latent_length, :]
        else:
            src_latents = silence_latent[:, :latent_length, :]

        src_latents = src_latents.to(device).to(dtype)

        # Chunk masks = 1 means "generate this region"
        chunk_masks = torch.ones(1, latent_length, 64, device=device, dtype=dtype)

        # Concatenate
        context_latents = torch.cat([src_latents, chunk_masks], dim=-1)  # [1, T, 128]

        # Prepare output tensors (remove batch dimension for storage)
        # encoder_hidden_states already contains merged text+lyrics from condition encoder
        tensor_data = {
            "target_latents": target_latents.squeeze(0).cpu(),  # [T, 64]
            "attention_mask": attention_mask.squeeze(0).cpu(),  # [T]
            "encoder_hidden_states": encoder_hidden_states.squeeze(0).cpu(),  # [L, 2048] (merged text+lyrics)
            "encoder_attention_mask": encoder_attention_mask.squeeze(0).cpu(),  # [L]
            "context_latents": context_latents.squeeze(0).cpu(),  # [T, 128]
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
