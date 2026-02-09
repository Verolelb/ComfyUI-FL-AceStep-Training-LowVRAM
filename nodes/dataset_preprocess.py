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


def encode_text_with_clip(clip, text: str, device, dtype):
    """
    Encode text using ComfyUI's native CLIP.

    ComfyUI CLIP from checkpoint has a different API than our custom wrapper.
    This function handles both cases.

    Args:
        clip: ComfyUI CLIP object
        text: Text to encode
        device: Device to use
        dtype: Data type to use

    Returns:
        Tuple of (hidden_states, attention_mask)
    """
    # Try our custom wrapper's encode method first
    if hasattr(clip, 'encode') and callable(clip.encode):
        try:
            # Check if it accepts max_length (our custom wrapper)
            import inspect
            sig = inspect.signature(clip.encode)
            if 'max_length' in sig.parameters:
                return clip.encode(text, max_length=256)
        except (TypeError, ValueError):
            pass

    # Use ComfyUI's native CLIP encoding
    # ComfyUI CLIP uses tokenize() -> encode_from_tokens()
    if hasattr(clip, 'tokenize'):
        # Native ComfyUI CLIP
        tokens = clip.tokenize(text)

        # encode_from_tokens returns conditioning in ComfyUI format
        if hasattr(clip, 'encode_from_tokens'):
            cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
            # cond is the hidden states tensor
            hidden_states = cond.to(device).to(dtype)
            # Create attention mask (all ones for valid tokens)
            attention_mask = torch.ones(hidden_states.shape[:2], device=device, dtype=dtype)
            return hidden_states, attention_mask

    # Fallback: Try to access the underlying model directly
    if hasattr(clip, 'cond_stage_model'):
        model = clip.cond_stage_model
        if hasattr(model, 'tokenizer') and hasattr(model, 'transformer'):
            tokenizer = model.tokenizer
            transformer = model.transformer

            # Tokenize
            inputs = tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=256
            )

            input_ids = inputs['input_ids'].to(device)
            attention_mask = inputs['attention_mask'].to(device).to(dtype)

            # Encode
            outputs = transformer(input_ids)
            hidden_states = outputs.last_hidden_state.to(dtype)

            return hidden_states, attention_mask

    raise RuntimeError(
        "Could not encode text with CLIP. The CLIP object doesn't have a compatible API. "
        "Please use the ACE-Step CLIP Loader node instead of the native checkpoint loader."
    )


class FL_AceStep_PreprocessDataset:
    """
    Preprocess Dataset

    Converts labeled audio samples to preprocessed tensor files for training.
    This is a critical step that:
    - Encodes audio to VAE latents
    - Encodes captions and metadata to text embeddings
    - Encodes lyrics to separate embeddings
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
                    "default": "./datasets/preprocessed",
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

        # Build metadata string
        metas = []
        if sample.bpm:
            metas.append(f"- bpm: {sample.bpm}")
        if sample.timesignature:
            metas.append(f"- timesignature: {sample.timesignature}")
        if sample.keyscale:
            metas.append(f"- keyscale: {sample.keyscale}")
        metas.append(f"- duration: {int(sample.duration)} seconds")

        metas_str = "\n".join(metas) if metas else ""

        # Build SFT prompt
        text_prompt = SFT_GEN_PROMPT.format(
            DEFAULT_DIT_INSTRUCTION,
            text_content,
            metas_str
        )

        # Encode text using ComfyUI's native CLIP API
        logger.info(f"Encoding text prompt (len={len(text_prompt)})")
        with torch.no_grad():
            encoder_hidden_states, encoder_attention_mask = encode_text_with_clip(
                clip, text_prompt, device, dtype
            )
        logger.info(f"Text encoder output shape: {encoder_hidden_states.shape}")

        # Encode lyrics
        lyrics = sample.lyrics if sample.lyrics else "[Instrumental]"
        logger.info(f"Encoding lyrics (len={len(lyrics)})")

        with torch.no_grad():
            lyric_hidden_states, lyric_attention_mask = encode_text_with_clip(
                clip, lyrics, device, dtype
            )
        logger.info(f"Lyric encoder output shape: {lyric_hidden_states.shape}")

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
        tensor_data = {
            "target_latents": target_latents.squeeze(0).cpu(),  # [T, 64]
            "attention_mask": attention_mask.squeeze(0).cpu(),  # [T]
            "encoder_hidden_states": encoder_hidden_states.squeeze(0).cpu(),  # [L, D]
            "encoder_attention_mask": encoder_attention_mask.squeeze(0).cpu(),  # [L]
            "lyric_hidden_states": lyric_hidden_states.squeeze(0).cpu(),  # [L, D]
            "lyric_attention_mask": lyric_attention_mask.squeeze(0).cpu(),  # [L]
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
