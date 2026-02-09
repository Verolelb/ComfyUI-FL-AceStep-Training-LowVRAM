"""
ACE-Step Dataset Preprocess Node

Converts labeled samples to tensor files for training.
Uses native ComfyUI MODEL type for the ACE-Step model.
"""

import os
import json
import math
import logging
from pathlib import Path
from typing import List

import torch
import torchaudio

# Try to use soundfile for audio loading (more compatible than torchcodec)
try:
    import soundfile as sf
    SOUNDFILE_AVAILABLE = True
except ImportError:
    sf = None
    SOUNDFILE_AVAILABLE = False

try:
    from comfy.utils import ProgressBar
except ImportError:
    ProgressBar = None

from ..modules.acestep_model import (
    is_acestep_model,
    get_silence_latent,
)

logger = logging.getLogger("FL_AceStep_Training")

# Log audio backend availability at import time
logger.info(f"Soundfile available: {SOUNDFILE_AVAILABLE}")

# SFT generation prompt template (from ACE-Step constants)
SFT_GEN_PROMPT = """# Instruction
{}

# Caption
{}

# Metas
{}<|endoftext|>
"""

DEFAULT_DIT_INSTRUCTION = "Fill the audio semantic mask based on the given conditions:"

# Constants for tiled VAE encoding
SAMPLE_RATE = 48000
VAE_CHUNK_SIZE = SAMPLE_RATE * 30  # 30 seconds
VAE_OVERLAP = SAMPLE_RATE * 2  # 2 seconds overlap


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


def tiled_vae_encode(vae, audio_tensor: torch.Tensor, device, dtype) -> torch.Tensor:
    """
    Encode audio to latent space using tiled encoding for long audio.

    The VAE (AutoencoderOobleck) cannot process very long audio in one pass.
    For audio longer than ~30 seconds, we use tiled encoding with overlap
    to avoid boundary artifacts.

    Args:
        vae: ComfyUI VAE object (has first_stage_model attribute)
        audio_tensor: Audio tensor [B, 2, T] at 48kHz (stereo)
        device: Device to use
        dtype: Data type to use (may be overridden by VAE's actual dtype)

    Returns:
        latents: Latent tensor [B, T_latent, 64]
    """
    print(f"[ACEStep] tiled_vae_encode() called with shape {audio_tensor.shape}")
    logger.info(f"[ACEStep] tiled_vae_encode() called with shape {audio_tensor.shape}")

    # Get the actual VAE model from ComfyUI's wrapper
    # ComfyUI VAE has first_stage_model which is the actual model
    if hasattr(vae, 'first_stage_model'):
        vae_model = vae.first_stage_model
    else:
        vae_model = vae

    # Get VAE's actual dtype from its parameters to avoid dtype mismatch
    # The VAE from checkpoint may be float16 while we default to bfloat16
    try:
        vae_dtype = next(vae_model.parameters()).dtype
        vae_device = next(vae_model.parameters()).device
        logger.info(f"[ACEStep] VAE dtype from model: {vae_dtype}, device: {vae_device}")
    except StopIteration:
        # Fallback if VAE has no parameters (shouldn't happen)
        vae_dtype = dtype
        vae_device = device
        logger.warning(f"[ACEStep] Could not detect VAE dtype, using default: {vae_dtype}")

    # Move VAE to GPU if it's on CPU (CPU encoding is extremely slow)
    target_device = torch.device('cuda') if torch.cuda.is_available() else vae_device
    if vae_device != target_device:
        logger.info(f"[ACEStep] Moving VAE from {vae_device} to {target_device} for faster encoding")
        vae_model = vae_model.to(target_device)
        vae_device = target_device

    with torch.no_grad():
        # Use VAE's actual dtype to avoid mismatch errors
        audio = audio_tensor.to(vae_device).to(vae_dtype)
        B, C, S = audio.shape

        # Short audio - encode directly (no tiling needed)
        if S <= VAE_CHUNK_SIZE:
            logger.info(f"[ACEStep] Short audio ({S / SAMPLE_RATE:.1f}s), encoding directly")
            latent_dist = vae_model.encode(audio)
            # Handle different VAE output formats
            if hasattr(latent_dist, 'latent_dist'):
                latents = latent_dist.latent_dist.sample()
            elif hasattr(latent_dist, 'sample'):
                latents = latent_dist.sample()
            else:
                latents = latent_dist
            # Transpose from [B, 64, T] to [B, T, 64] for training
            return latents.transpose(1, 2)

        # Long audio - use tiled encoding with overlap-discard strategy
        logger.info(f"[ACEStep] Using tiled VAE encoding for long audio ({S / SAMPLE_RATE:.1f}s)")
        print(f"[ACEStep] Using tiled VAE encoding for long audio ({S / SAMPLE_RATE:.1f}s)")

        stride = VAE_CHUNK_SIZE - 2 * VAE_OVERLAP  # Core size (non-overlapping part)
        num_steps = math.ceil(S / stride)

        logger.info(f"[ACEStep] Will process {num_steps} chunks with stride {stride}")

        encoded_latent_list: List[torch.Tensor] = []
        downsample_factor = None

        for i in range(num_steps):
            # Calculate core region (non-overlapping part we want to keep)
            core_start = i * stride
            core_end = min(core_start + stride, S)

            # Calculate window region (core + overlap on both sides)
            win_start = max(0, core_start - VAE_OVERLAP)
            win_end = min(S, core_end + VAE_OVERLAP)

            # Extract and encode chunk
            audio_chunk = audio[:, :, win_start:win_end]

            logger.info(f"[ACEStep] Encoding chunk {i+1}/{num_steps}: samples [{win_start}:{win_end}]")

            latent_dist = vae_model.encode(audio_chunk)
            # Handle different VAE output formats
            if hasattr(latent_dist, 'latent_dist'):
                latent_chunk = latent_dist.latent_dist.sample()
            elif hasattr(latent_dist, 'sample'):
                latent_chunk = latent_dist.sample()
            else:
                latent_chunk = latent_dist

            # Get downsample factor from first chunk
            if downsample_factor is None:
                downsample_factor = audio_chunk.shape[-1] / latent_chunk.shape[-1]
                logger.info(f"[ACEStep] VAE downsample factor: {downsample_factor:.2f}")

            # Calculate trim amounts in latent frames
            # We need to trim the overlap regions to get just the core
            trim_start = int(round((core_start - win_start) / downsample_factor))
            added_end = win_end - core_end
            trim_end = int(round(added_end / downsample_factor))

            # Extract core latent (discard overlap regions)
            end_idx = latent_chunk.shape[-1] - trim_end if trim_end > 0 else latent_chunk.shape[-1]
            latent_core = latent_chunk[:, :, trim_start:end_idx]
            encoded_latent_list.append(latent_core)

            logger.debug(f"[ACEStep] Chunk {i+1}/{num_steps}: latent core shape {latent_core.shape}")

        # Concatenate all core latents along time dimension
        final_latents = torch.cat(encoded_latent_list, dim=-1)  # [B, 64, T_total]

        logger.info(f"[ACEStep] Tiled encoding complete: {len(encoded_latent_list)} chunks -> latent shape {final_latents.shape}")
        print(f"[ACEStep] Tiled encoding complete: {len(encoded_latent_list)} chunks -> latent shape {final_latents.shape}")

        # Transpose from [B, 64, T] to [B, T, 64] for training
        return final_latents.transpose(1, 2)


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
        # Load audio - use soundfile directly to avoid torchaudio's torchcodec dependency
        if SOUNDFILE_AVAILABLE:
            # Load with soundfile and convert to torch tensor
            data, sr = sf.read(sample.audio_path, dtype='float32')
            # soundfile returns (samples, channels) for stereo, we need (channels, samples)
            if len(data.shape) == 1:
                # Mono: add channel dimension
                waveform = torch.from_numpy(data).unsqueeze(0)
            else:
                # Stereo/multi-channel: transpose to (channels, samples)
                waveform = torch.from_numpy(data.T)
        else:
            # Fallback to torchaudio (may fail without torchcodec)
            waveform, sr = torchaudio.load(sample.audio_path)

        # Resample to 48kHz
        if sr != 48000:
            resampler = torchaudio.transforms.Resample(sr, 48000)
            waveform = resampler(waveform)

        # Check minimum duration - VAE requires at least ~0.5 seconds at 48kHz
        # The VAE has convolutional layers with kernel size 7 that need sufficient input
        MIN_SAMPLES = 48000  # 1 second minimum at 48kHz
        if waveform.shape[1] < MIN_SAMPLES:
            raise ValueError(f"Audio too short: {waveform.shape[1]} samples ({waveform.shape[1]/48000:.2f}s). Minimum is 1 second.")

        # Convert to stereo
        if waveform.shape[0] == 1:
            waveform = waveform.repeat(2, 1)
        elif waveform.shape[0] > 2:
            waveform = waveform[:2, :]

        # Truncate to max duration
        max_samples = int(max_duration * 48000)
        if waveform.shape[1] > max_samples:
            waveform = waveform[:, :max_samples]

        # Add batch dimension
        waveform = waveform.unsqueeze(0)

        # Debug: log waveform shape before VAE encoding
        logger.info(f"Waveform shape before VAE: {waveform.shape}, dtype: {waveform.dtype}")

        # Encode to VAE latents using tiled encoding for long audio
        target_latents = tiled_vae_encode(vae, waveform, device, dtype)  # [1, T, 64]

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
