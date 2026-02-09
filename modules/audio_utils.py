"""
Shared audio utilities for ACE-Step Training.

Provides audio loading, VAE encoding, and audio-to-codes conversion
used by both the labeling and preprocessing nodes.
"""

import logging
from typing import Tuple

import torch
import torchaudio

try:
    import soundfile as sf
    SOUNDFILE_AVAILABLE = True
except ImportError:
    sf = None
    SOUNDFILE_AVAILABLE = False

logger = logging.getLogger("FL_AceStep_Training")

# Constants
SAMPLE_RATE = 48000
MIN_SAMPLES = SAMPLE_RATE  # 1 second minimum


def load_audio(audio_path: str, max_duration: float = None) -> Tuple[torch.Tensor, int]:
    """
    Load an audio file, resample to 48kHz stereo.

    Args:
        audio_path: Path to audio file
        max_duration: Optional max duration in seconds to truncate to

    Returns:
        Tuple of (waveform [C, T], sample_rate)
    """
    # Load audio - prefer soundfile for broader codec support
    if SOUNDFILE_AVAILABLE:
        data, sr = sf.read(audio_path, dtype='float32')
        if len(data.shape) == 1:
            waveform = torch.from_numpy(data).unsqueeze(0)
        else:
            waveform = torch.from_numpy(data.T)
    else:
        waveform, sr = torchaudio.load(audio_path)

    # Resample to 48kHz
    if sr != SAMPLE_RATE:
        resampler = torchaudio.transforms.Resample(sr, SAMPLE_RATE)
        waveform = resampler(waveform)

    # Validate minimum duration
    if waveform.shape[-1] < MIN_SAMPLES:
        raise ValueError(
            f"Audio too short: {waveform.shape[-1]} samples "
            f"({waveform.shape[-1] / SAMPLE_RATE:.2f}s). Minimum is 1 second."
        )

    # Convert to stereo
    if waveform.shape[0] == 1:
        waveform = waveform.repeat(2, 1)
    elif waveform.shape[0] > 2:
        waveform = waveform[:2, :]

    # Truncate to max duration
    if max_duration is not None:
        max_samples = int(max_duration * SAMPLE_RATE)
        if waveform.shape[-1] > max_samples:
            waveform = waveform[:, :max_samples]

    return waveform, SAMPLE_RATE


def vae_encode(vae, audio_tensor: torch.Tensor) -> torch.Tensor:
    """
    Encode audio to latent space using ComfyUI's managed VAE API.

    Uses ComfyUI's vae.encode() which handles:
    - GPU memory management (load_models_gpu)
    - Automatic tiled encoding fallback on OOM
    - Proper dtype handling
    - Keeping the VAE loaded between calls

    Args:
        vae: ComfyUI VAE object (the wrapper, not first_stage_model)
        audio_tensor: Audio tensor [B, C, T] at 48kHz (stereo, C=2)

    Returns:
        latents: Latent tensor [B, T_latent, 64]
    """
    # ComfyUI's vae.encode() expects [B, T, C] format for 1D audio
    # (it internally does movedim(-1, 1) to get [B, C, T])
    audio_for_vae = audio_tensor.movedim(1, -1)  # [B, C, T] -> [B, T, C]

    logger.info(
        f"[ACEStep] VAE encoding audio: {audio_tensor.shape[2] / SAMPLE_RATE:.1f}s "
        f"({audio_tensor.shape})"
    )

    # ComfyUI's encode() returns [B, 64, T_latent] and handles all
    # device management, tiling, and OOM recovery internally
    latents = vae.encode(audio_for_vae)  # [B, 64, T_latent]

    # Transpose to [B, T_latent, 64] for training/tokenization
    latents = latents.transpose(1, 2)

    logger.info(f"[ACEStep] VAE encode complete: latent shape {latents.shape}")

    return latents


def audio_to_codes(
    vae,
    tokenizer,
    audio_path: str,
    device,
    dtype,
    max_duration: float = 30.0,
) -> str:
    """
    Convert an audio file to discrete audio code tokens for LLM understanding.

    Pipeline: Audio -> VAE encode -> DiT tokenizer (FSQ quantize) -> code string

    Args:
        vae: ComfyUI VAE object
        tokenizer: AceStepAudioTokenizer from the DiT model
        audio_path: Path to audio file
        device: Device for computation (used for tokenizer)
        dtype: Data type for computation (used for tokenizer)
        max_duration: Max audio duration in seconds (default 30s for labeling)

    Returns:
        String of audio codes like "<|audio_code_123|><|audio_code_456|>..."
    """
    # 1. Load and preprocess audio
    waveform, sr = load_audio(audio_path, max_duration=max_duration)
    waveform = waveform.unsqueeze(0)  # Add batch dim: [1, 2, T]

    # 2. VAE encode to latents [1, T_latent, 64]
    # ComfyUI handles device/dtype/tiling internally
    latents = vae_encode(vae, waveform)

    # 3. Tokenize latents to discrete codes
    # Move latents to tokenizer's device/dtype for FSQ quantization
    tokenizer_device = next(tokenizer.parameters()).device
    tokenizer_dtype = next(tokenizer.parameters()).dtype
    latents_for_tok = latents.to(tokenizer_device).to(tokenizer_dtype)

    with torch.no_grad():
        # tokenizer.tokenize(x) expects [B, T, D] and returns (quantized, indices)
        # It handles padding T to a multiple of pool_window_size (5) internally
        quantized, indices = tokenizer.tokenize(latents_for_tok)

    # indices shape: [B, T_5Hz, num_quantizers] or [B, T_5Hz, 1]
    indices_flat = indices.flatten().cpu().tolist()

    logger.info(f"[ACEStep] audio_to_codes: {len(indices_flat)} codes from {audio_path}")

    # 4. Format as code string
    codes_string = "".join(f"<|audio_code_{int(idx)}|>" for idx in indices_flat)

    return codes_string
