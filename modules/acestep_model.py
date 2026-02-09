"""
ACE-Step Model Utilities for ComfyUI Native Integration

This module provides utilities for working with ACE-Step 1.5 models
using ComfyUI's native MODEL type system. ComfyUI already has built-in
support for ACE-Step 1.5 via:
- comfy.model_base.ACEStep15 (BaseModel)
- comfy.supported_models.ACEStep15 (model config)
- comfy.ldm.ace.ace_step15.AceStepConditionGenerationModel (DiT)

We leverage this infrastructure rather than creating custom wrappers.
"""

import os
import logging
from typing import Optional, Dict, Any, Tuple

import torch

logger = logging.getLogger("FL_AceStep_Training")


def get_acestep_dit(model_patcher) -> torch.nn.Module:
    """
    Extract the DiT (diffusion model) from a ComfyUI ModelPatcher.

    For ACE-Step 1.5, this is an AceStepConditionGenerationModel.

    Args:
        model_patcher: ComfyUI ModelPatcher wrapping an ACEStep15 BaseModel

    Returns:
        The diffusion_model (AceStepConditionGenerationModel)
    """
    return model_patcher.model.diffusion_model


def get_acestep_decoder(model_patcher) -> torch.nn.Module:
    """
    Extract the decoder (DiT core) from a ComfyUI ModelPatcher.

    The decoder is the actual DiT transformer that we apply LoRA to.

    Args:
        model_patcher: ComfyUI ModelPatcher wrapping an ACEStep15 BaseModel

    Returns:
        The decoder (AceStepDiTModel)
    """
    dit = get_acestep_dit(model_patcher)
    return dit.decoder


def get_acestep_encoder(model_patcher) -> torch.nn.Module:
    """
    Extract the condition encoder from a ComfyUI ModelPatcher.

    Args:
        model_patcher: ComfyUI ModelPatcher wrapping an ACEStep15 BaseModel

    Returns:
        The encoder (AceStepConditionEncoder)
    """
    dit = get_acestep_dit(model_patcher)
    return dit.encoder


def get_acestep_tokenizer(model_patcher) -> torch.nn.Module:
    """
    Extract the audio tokenizer from a ComfyUI ModelPatcher.

    Used for converting audio to codes for LLM understanding.

    Args:
        model_patcher: ComfyUI ModelPatcher wrapping an ACEStep15 BaseModel

    Returns:
        The tokenizer (AceStepAudioTokenizer)
    """
    dit = get_acestep_dit(model_patcher)
    return dit.tokenizer


def is_acestep_model(model_patcher) -> bool:
    """
    Check if a ModelPatcher contains an ACE-Step 1.5 model.

    Args:
        model_patcher: ComfyUI ModelPatcher to check

    Returns:
        True if this is an ACE-Step 1.5 model
    """
    try:
        # Check if the model has the ACE-Step specific attributes
        dit = model_patcher.model.diffusion_model
        return (
            hasattr(dit, 'decoder') and
            hasattr(dit, 'encoder') and
            hasattr(dit, 'tokenizer') and
            hasattr(dit, 'detokenizer')
        )
    except (AttributeError, TypeError):
        return False


def get_silence_latent(model_patcher) -> Optional[torch.Tensor]:
    """
    Get the silence latent from model_options if stored there.

    The silence latent is used for creating context during preprocessing.

    Args:
        model_patcher: ComfyUI ModelPatcher

    Returns:
        Silence latent tensor or None
    """
    return model_patcher.model_options.get("silence_latent", None)


def set_silence_latent(model_patcher, silence_latent: torch.Tensor):
    """
    Store the silence latent in model_options.

    Args:
        model_patcher: ComfyUI ModelPatcher
        silence_latent: The silence latent tensor
    """
    model_patcher.model_options["silence_latent"] = silence_latent


def clone_model_for_training(model_patcher):
    """
    Clone a ModelPatcher for training purposes.

    This creates an independent copy that can be modified
    without affecting the original model.

    Args:
        model_patcher: ComfyUI ModelPatcher to clone

    Returns:
        Cloned ModelPatcher
    """
    return model_patcher.clone()


def apply_lora_patches(model_patcher, patches: Dict[str, Any], strength: float = 1.0):
    """
    Apply LoRA patches to a ModelPatcher using ComfyUI's native patching.

    Args:
        model_patcher: ComfyUI ModelPatcher (should be cloned first)
        patches: Dictionary of parameter name -> patch data
        strength: LoRA strength multiplier

    Returns:
        List of keys that were successfully patched
    """
    return model_patcher.add_patches(patches, strength_patch=strength)


def convert_peft_to_comfy_patches(peft_state_dict: Dict[str, torch.Tensor],
                                   prefix: str = "diffusion_model.decoder.") -> Dict[str, Tuple]:
    """
    Convert PEFT LoRA state dict to ComfyUI patch format.

    ComfyUI patches are in the format:
        key -> (strength_patch, weight_diff, strength_model, offset, function)

    For LoRA, we need to compute the weight diff from lora_A and lora_B matrices.

    Args:
        peft_state_dict: State dict from PEFT adapter
        prefix: Prefix to add to keys for matching model parameters

    Returns:
        Dictionary in ComfyUI patch format
    """
    patches = {}

    # Group lora_A and lora_B pairs
    lora_pairs = {}
    for key, value in peft_state_dict.items():
        if "lora_A" in key:
            base_key = key.replace(".lora_A.weight", "").replace(".lora_A.default.weight", "")
            if base_key not in lora_pairs:
                lora_pairs[base_key] = {}
            lora_pairs[base_key]["A"] = value
        elif "lora_B" in key:
            base_key = key.replace(".lora_B.weight", "").replace(".lora_B.default.weight", "")
            if base_key not in lora_pairs:
                lora_pairs[base_key] = {}
            lora_pairs[base_key]["B"] = value

    # Compute weight diffs for each LoRA pair
    for base_key, pair in lora_pairs.items():
        if "A" in pair and "B" in pair:
            lora_A = pair["A"]
            lora_B = pair["B"]

            # Compute weight diff: W_diff = B @ A (or appropriate transpose)
            # This is BA in the original LoRA paper
            if lora_A.dim() == 2 and lora_B.dim() == 2:
                weight_diff = lora_B @ lora_A
            else:
                # Handle conv layers if needed
                weight_diff = torch.einsum("oi,ir->or", lora_B.flatten(1), lora_A.flatten(1))
                # Reshape to match original weight shape
                # This may need adjustment based on actual layer shapes

            # Create ComfyUI patch format
            # The full key should match the parameter name in the model
            full_key = f"{prefix}{base_key}.weight"

            # ComfyUI patch tuple: (strength, (weight, ...), strength_model)
            # Using the "diff" format where we add weight_diff * strength
            patches[full_key] = ("diff", weight_diff)

    return patches


def load_lora_from_peft_adapter(adapter_path: str,
                                 device: str = "cpu") -> Dict[str, torch.Tensor]:
    """
    Load a PEFT adapter and return the state dict.

    Args:
        adapter_path: Path to PEFT adapter directory
        device: Device to load to

    Returns:
        State dict from the adapter
    """
    import safetensors.torch

    # Check for safetensors or bin file
    safetensors_path = os.path.join(adapter_path, "adapter_model.safetensors")
    bin_path = os.path.join(adapter_path, "adapter_model.bin")

    if os.path.exists(safetensors_path):
        state_dict = safetensors.torch.load_file(safetensors_path, device=device)
    elif os.path.exists(bin_path):
        state_dict = torch.load(bin_path, map_location=device, weights_only=True)
    else:
        raise FileNotFoundError(f"No adapter_model found in {adapter_path}")

    return state_dict


def load_lora_from_safetensors(lora_path: str,
                                device: str = "cpu") -> Dict[str, torch.Tensor]:
    """
    Load a LoRA from a safetensors file.

    Args:
        lora_path: Path to safetensors file
        device: Device to load to

    Returns:
        State dict from the file
    """
    import safetensors.torch
    return safetensors.torch.load_file(lora_path, device=device)


class ACEStepModelHelper:
    """
    Helper class for working with ACE-Step models in ComfyUI.

    Provides convenient methods for accessing model components
    and managing LoRA adapters.
    """

    def __init__(self, model_patcher):
        """
        Initialize with a ComfyUI ModelPatcher.

        Args:
            model_patcher: ComfyUI ModelPatcher wrapping ACEStep15
        """
        if not is_acestep_model(model_patcher):
            raise ValueError("Model is not an ACE-Step 1.5 model")

        self.model_patcher = model_patcher

    @property
    def dit(self) -> torch.nn.Module:
        """Get the full DIT model (AceStepConditionGenerationModel)."""
        return get_acestep_dit(self.model_patcher)

    @property
    def decoder(self) -> torch.nn.Module:
        """Get the decoder (AceStepDiTModel) - the main transformer."""
        return get_acestep_decoder(self.model_patcher)

    @property
    def encoder(self) -> torch.nn.Module:
        """Get the condition encoder."""
        return get_acestep_encoder(self.model_patcher)

    @property
    def tokenizer(self) -> torch.nn.Module:
        """Get the audio tokenizer."""
        return get_acestep_tokenizer(self.model_patcher)

    @property
    def silence_latent(self) -> Optional[torch.Tensor]:
        """Get the silence latent if available."""
        return get_silence_latent(self.model_patcher)

    @silence_latent.setter
    def silence_latent(self, value: torch.Tensor):
        """Set the silence latent."""
        set_silence_latent(self.model_patcher, value)

    def clone(self):
        """Create a clone for modifications."""
        return ACEStepModelHelper(clone_model_for_training(self.model_patcher))

    def get_decoder_state_dict(self) -> Dict[str, torch.Tensor]:
        """Get the decoder's state dict for LoRA target identification."""
        return self.decoder.state_dict()

    def get_trainable_parameters(self):
        """Get parameters suitable for LoRA training."""
        return self.decoder.parameters()
