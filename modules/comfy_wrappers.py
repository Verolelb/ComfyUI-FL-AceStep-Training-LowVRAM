"""
ComfyUI Compatibility Wrappers

Wraps ACE-Step models to be compatible with ComfyUI's type system.
- ACEStepVAEWrapper: Wraps AutoencoderOobleck for ComfyUI VAE type (red connection)
- ACEStepCLIPWrapper: Wraps Qwen3-Embedding for ComfyUI CLIP type (yellow connection)

Note: The DiT model now uses native ComfyUI MODEL type via ModelPatcher.
See modules/acestep_model.py for MODEL-related utilities.
ACEStepDiTHandler is deprecated and kept only for backwards compatibility.
"""

import os
import math
import logging
from typing import Optional, Dict, Any, Tuple, Union, List
from copy import deepcopy

import torch
import torchaudio

logger = logging.getLogger("FL_AceStep_Training")


class ACEStepVAEWrapper:
    """
    Wraps AutoencoderOobleck for ComfyUI VAE compatibility.

    This wrapper provides a ComfyUI-compatible interface for the ACE-Step VAE,
    which operates on audio rather than images.
    """

    def __init__(self, vae_model, device: str = "cuda", dtype: torch.dtype = torch.bfloat16):
        """
        Initialize the VAE wrapper.

        Args:
            vae_model: The AutoencoderOobleck model
            device: Device to use
            dtype: Data type for computations
        """
        self.vae = vae_model
        self.device = device
        self.dtype = dtype
        self.sample_rate = 48000

        # ComfyUI VAE compatibility - these attributes may be checked
        self.first_stage_model = self

    def encode(self, audio_tensor: torch.Tensor) -> torch.Tensor:
        """
        Encode audio to latent space using tiled encoding for long audio.

        The VAE (AutoencoderOobleck) cannot process very long audio in one pass.
        For audio longer than ~30 seconds, we use tiled encoding with overlap
        to avoid boundary artifacts.

        Args:
            audio_tensor: Audio tensor [B, 2, T] at 48kHz (stereo)

        Returns:
            latents: Latent tensor [B, T_latent, 64]
        """
        # Debug: confirm this method is being called
        print(f"[ACEStep VAE Wrapper] encode() called with shape {audio_tensor.shape}")
        logger.info(f"[VAE Wrapper] encode() called with shape {audio_tensor.shape}")

        # Tiled encoding parameters
        chunk_size = self.sample_rate * 30  # 30 seconds at 48kHz = 1,440,000 samples
        overlap = self.sample_rate * 2  # 2 seconds overlap = 96,000 samples

        with torch.no_grad():
            audio = audio_tensor.to(self.device).to(self.dtype)
            B, C, S = audio.shape

            # Short audio - encode directly (no tiling needed)
            if S <= chunk_size:
                latent_dist = self.vae.encode(audio)
                latents = latent_dist.latent_dist.sample()
                # Transpose from [B, 64, T] to [B, T, 64] for training
                return latents.transpose(1, 2)

            # Long audio - use tiled encoding with overlap-discard strategy
            logger.info(f"Using tiled VAE encoding for long audio ({S / self.sample_rate:.1f}s)")

            stride = chunk_size - 2 * overlap  # Core size (non-overlapping part)
            num_steps = math.ceil(S / stride)

            encoded_latent_list: List[torch.Tensor] = []
            downsample_factor = None

            for i in range(num_steps):
                # Calculate core region (non-overlapping part we want to keep)
                core_start = i * stride
                core_end = min(core_start + stride, S)

                # Calculate window region (core + overlap on both sides)
                win_start = max(0, core_start - overlap)
                win_end = min(S, core_end + overlap)

                # Extract and encode chunk
                audio_chunk = audio[:, :, win_start:win_end]

                latent_dist = self.vae.encode(audio_chunk)
                latent_chunk = latent_dist.latent_dist.sample()  # [B, 64, T_chunk]

                # Get downsample factor from first chunk
                if downsample_factor is None:
                    downsample_factor = audio_chunk.shape[-1] / latent_chunk.shape[-1]
                    logger.debug(f"VAE downsample factor: {downsample_factor:.2f}")

                # Calculate trim amounts in latent frames
                # We need to trim the overlap regions to get just the core
                trim_start = int(round((core_start - win_start) / downsample_factor))
                added_end = win_end - core_end
                trim_end = int(round(added_end / downsample_factor))

                # Extract core latent (discard overlap regions)
                end_idx = latent_chunk.shape[-1] - trim_end if trim_end > 0 else latent_chunk.shape[-1]
                latent_core = latent_chunk[:, :, trim_start:end_idx]
                encoded_latent_list.append(latent_core)

                logger.debug(f"Chunk {i+1}/{num_steps}: audio [{win_start}:{win_end}] -> latent core [{trim_start}:{end_idx}]")

            # Concatenate all core latents along time dimension
            final_latents = torch.cat(encoded_latent_list, dim=-1)  # [B, 64, T_total]

            logger.info(f"Tiled encoding complete: {len(encoded_latent_list)} chunks -> latent shape {final_latents.shape}")

            # Transpose from [B, 64, T] to [B, T, 64] for training
            return final_latents.transpose(1, 2)

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """
        Decode latents to audio.

        Args:
            latents: Latent tensor [B, T_latent, 64]

        Returns:
            audio: Audio tensor [B, 2, T] at 48kHz
        """
        with torch.no_grad():
            # Transpose from [B, T, 64] to [B, 64, T]
            latents = latents.transpose(1, 2).to(self.device).to(self.dtype)
            return self.vae.decode(latents).sample

    def encode_audio_file(self, audio_path: str, max_duration: float = 240.0) -> torch.Tensor:
        """
        Load and encode an audio file.

        Args:
            audio_path: Path to audio file
            max_duration: Maximum duration in seconds

        Returns:
            latents: Latent tensor [1, T_latent, 64]
        """
        # Load audio
        waveform, sr = torchaudio.load(audio_path)

        # Resample to 48kHz if needed
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)

        # Convert to stereo if mono
        if waveform.shape[0] == 1:
            waveform = waveform.repeat(2, 1)
        elif waveform.shape[0] > 2:
            waveform = waveform[:2, :]

        # Truncate to max duration
        max_samples = int(max_duration * self.sample_rate)
        if waveform.shape[1] > max_samples:
            waveform = waveform[:, :max_samples]

        # Add batch dimension
        waveform = waveform.unsqueeze(0)

        return self.encode(waveform)

    def to(self, device):
        """Move to device."""
        self.device = device
        self.vae = self.vae.to(device)
        return self


class ACEStepCLIPWrapper:
    """
    Wraps Qwen3-Embedding for ComfyUI CLIP compatibility.

    This wrapper provides a ComfyUI-compatible interface for the ACE-Step
    text encoder (Qwen3-Embedding-0.6B).
    """

    def __init__(
        self,
        text_encoder,
        tokenizer,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16
    ):
        """
        Initialize the CLIP wrapper.

        Args:
            text_encoder: The Qwen3-Embedding model
            tokenizer: The tokenizer
            device: Device to use
            dtype: Data type for computations
        """
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.device = device
        self.dtype = dtype

        # ComfyUI CLIP compatibility
        self.cond_stage_model = self

    def tokenize(self, text: str, max_length: int = 256) -> Dict[str, torch.Tensor]:
        """
        Tokenize text input.

        Args:
            text: Input text
            max_length: Maximum sequence length

        Returns:
            Dictionary with input_ids and attention_mask
        """
        return self.tokenizer(
            text,
            padding="max_length",
            max_length=max_length,
            truncation=True,
            return_tensors="pt",
        )

    def encode(
        self,
        text: str,
        max_length: int = 256
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode text to hidden states.

        Args:
            text: Input text
            max_length: Maximum sequence length

        Returns:
            Tuple of (hidden_states [B, L, D], attention_mask [B, L])
        """
        tokens = self.tokenize(text, max_length)
        input_ids = tokens.input_ids.to(self.device)
        attention_mask = tokens.attention_mask.to(self.device)

        with torch.no_grad():
            outputs = self.text_encoder(input_ids)
            hidden_states = outputs.last_hidden_state.to(self.dtype)

        return hidden_states, attention_mask

    def embed_tokens(
        self,
        text: str,
        max_length: int = 512
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get token embeddings (for lyrics encoding).

        Args:
            text: Input text (lyrics)
            max_length: Maximum sequence length

        Returns:
            Tuple of (embeddings [B, L, D], attention_mask [B, L])
        """
        tokens = self.tokenize(text, max_length)
        input_ids = tokens.input_ids.to(self.device)
        attention_mask = tokens.attention_mask.to(self.device)

        with torch.no_grad():
            embeddings = self.text_encoder.embed_tokens(input_ids)

        return embeddings.to(self.dtype), attention_mask

    def to(self, device):
        """Move to device."""
        self.device = device
        self.text_encoder = self.text_encoder.to(device)
        return self


class ACEStepDiTHandler:
    """
    DEPRECATED: Use ComfyUI's native MODEL type instead.

    This class is kept for backwards compatibility only.
    New code should use:
    - model_loader.ACEStep_ModelLoader for loading (outputs MODEL type)
    - modules/acestep_model.py utilities for accessing model components

    Handler for ACE-Step DiT model with all utilities.
    """

    def __init__(
        self,
        model,
        vae: Optional[ACEStepVAEWrapper] = None,
        text_encoder: Optional[ACEStepCLIPWrapper] = None,
        silence_latent: Optional[torch.Tensor] = None,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        config: Optional[Any] = None,
    ):
        """
        Initialize the DiT handler.

        Args:
            model: The AceStepConditionGenerationModel
            vae: VAE wrapper (optional, can be connected separately)
            text_encoder: Text encoder wrapper (optional)
            silence_latent: Silence latent tensor for preprocessing
            device: Device to use
            dtype: Data type
            config: Model configuration
        """
        self.model = model
        self.vae = vae
        self.text_encoder = text_encoder
        self.silence_latent = silence_latent
        self.device = device
        self.dtype = dtype
        self.config = config or getattr(model, 'config', None)

        # LoRA state
        self.lora_loaded = False
        self.use_lora = False
        self.lora_scale = 1.0
        self._base_decoder = None

        # Sample rate
        self.sample_rate = 48000

    def is_turbo_model(self) -> bool:
        """Check if this is a turbo model."""
        if self.config is None:
            return False
        return getattr(self.config, 'is_turbo', False)

    def get_decoder(self):
        """Get the decoder module."""
        return self.model.decoder

    def get_encoder(self):
        """Get the condition encoder module."""
        return self.model.encoder

    # =============== LoRA Management ===============

    def load_lora(self, lora_path: str) -> str:
        """
        Load a LoRA adapter into the decoder.

        Args:
            lora_path: Path to LoRA adapter directory

        Returns:
            Status message
        """
        if not lora_path or not lora_path.strip():
            return "Error: Please provide a LoRA path"

        lora_path = lora_path.strip()

        if not os.path.exists(lora_path):
            return f"Error: LoRA path not found: {lora_path}"

        # Check for adapter config
        config_file = os.path.join(lora_path, "adapter_config.json")
        if not os.path.exists(config_file):
            return f"Error: Invalid LoRA adapter - adapter_config.json not found"

        try:
            from peft import PeftModel
        except ImportError:
            return "Error: PEFT not installed. Run: pip install peft"

        try:
            # Backup base decoder if not already backed up
            if self._base_decoder is None:
                self._base_decoder = deepcopy(self.model.decoder)
                logger.info("Base decoder backed up")
            else:
                # Restore base decoder before loading new LoRA
                self.model.decoder = deepcopy(self._base_decoder)
                logger.info("Restored base decoder before loading new LoRA")

            # Load PEFT adapter
            logger.info(f"Loading LoRA from {lora_path}")
            self.model.decoder = PeftModel.from_pretrained(
                self.model.decoder,
                lora_path,
                is_trainable=False,
            )
            self.model.decoder = self.model.decoder.to(self.device).to(self.dtype)
            self.model.decoder.eval()

            self.lora_loaded = True
            self.use_lora = True

            return f"LoRA loaded from {lora_path}"

        except Exception as e:
            logger.exception("Failed to load LoRA")
            return f"Error loading LoRA: {str(e)}"

    def unload_lora(self) -> str:
        """
        Unload LoRA and restore base decoder.

        Returns:
            Status message
        """
        if not self.lora_loaded:
            return "No LoRA loaded"

        if self._base_decoder is None:
            return "Error: No base decoder backup available"

        try:
            self.model.decoder = deepcopy(self._base_decoder)
            self.model.decoder = self.model.decoder.to(self.device).to(self.dtype)
            self.model.decoder.eval()

            self.lora_loaded = False
            self.use_lora = False

            return "LoRA unloaded, base decoder restored"

        except Exception as e:
            logger.exception("Failed to unload LoRA")
            return f"Error unloading LoRA: {str(e)}"

    def set_use_lora(self, use: bool) -> str:
        """
        Enable or disable LoRA usage without unloading.

        Args:
            use: Whether to use LoRA

        Returns:
            Status message
        """
        if not self.lora_loaded:
            return "No LoRA loaded"

        self.use_lora = use

        try:
            # Try to use PEFT's enable/disable methods
            if hasattr(self.model.decoder, 'enable_adapter_layers'):
                if use:
                    self.model.decoder.enable_adapter_layers()
                else:
                    self.model.decoder.disable_adapter_layers()
        except Exception as e:
            logger.warning(f"Could not toggle adapter layers: {e}")

        return f"LoRA {'enabled' if use else 'disabled'}"

    def set_lora_scale(self, scale: float) -> str:
        """
        Adjust LoRA influence scale.

        Args:
            scale: Scale factor (0.0 to 1.0)

        Returns:
            Status message
        """
        if not self.lora_loaded:
            return "No LoRA loaded"

        scale = max(0.0, min(1.0, scale))
        self.lora_scale = scale

        try:
            # Try to set scaling on PEFT layers
            for name, module in self.model.decoder.named_modules():
                if hasattr(module, 'scaling'):
                    for key in module.scaling:
                        module.scaling[key] = scale
        except Exception as e:
            logger.warning(f"Could not set LoRA scale directly: {e}")

        return f"LoRA scale set to {scale:.2f}"

    def get_lora_status(self) -> Dict[str, Any]:
        """Get current LoRA status."""
        return {
            "lora_loaded": self.lora_loaded,
            "use_lora": self.use_lora,
            "lora_scale": self.lora_scale,
        }

    # =============== Audio Processing ===============

    def convert_audio_to_latents(
        self,
        audio_path: str,
        max_duration: float = 240.0,
    ) -> torch.Tensor:
        """
        Convert audio file to VAE latents.

        Args:
            audio_path: Path to audio file
            max_duration: Maximum duration in seconds

        Returns:
            latents: [1, T, 64]
        """
        if self.vae is None:
            raise ValueError("VAE not available")

        return self.vae.encode_audio_file(audio_path, max_duration)

    def convert_src_audio_to_codes(
        self,
        audio_path: str,
        duration_in_seconds: Optional[float] = None,
    ) -> Tuple[str, int]:
        """
        Convert source audio to semantic codes for LLM understanding.

        This is used during auto-labeling to convert audio to tokens
        that the LLM can understand.

        Args:
            audio_path: Path to audio file
            duration_in_seconds: Optional duration override

        Returns:
            Tuple of (codes_string, duration_seconds)
        """
        if self.vae is None:
            raise ValueError("VAE not available")

        # Load audio
        waveform, sr = torchaudio.load(audio_path)

        # Resample to 48kHz if needed
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)

        # Get duration
        duration = waveform.shape[1] / self.sample_rate
        if duration_in_seconds is not None:
            duration = min(duration, duration_in_seconds)

        # Convert to stereo
        if waveform.shape[0] == 1:
            waveform = waveform.repeat(2, 1)
        elif waveform.shape[0] > 2:
            waveform = waveform[:2, :]

        # Truncate if needed
        max_samples = int(duration * self.sample_rate)
        waveform = waveform[:, :max_samples]

        # Add batch dimension and encode
        waveform = waveform.unsqueeze(0).to(self.device).to(self.dtype)

        with torch.no_grad():
            latents = self.vae.encode(waveform)  # [1, T, 64]

        # Quantize to codes (simplified - actual implementation may differ)
        # This is a placeholder - the actual conversion depends on the LM tokenizer
        codes = latents.mean(dim=-1)  # [1, T]
        codes = (codes * 1000).long().clamp(0, 32000)

        # Convert to code string
        code_tokens = [f"<|audio_code_{c.item()}|>" for c in codes[0]]
        codes_string = "".join(code_tokens)

        return codes_string, int(duration)

    def to(self, device):
        """Move all components to device."""
        self.device = device
        self.model = self.model.to(device)
        if self.vae is not None:
            self.vae = self.vae.to(device)
        if self.text_encoder is not None:
            self.text_encoder = self.text_encoder.to(device)
        if self.silence_latent is not None:
            self.silence_latent = self.silence_latent.to(device)
        return self
