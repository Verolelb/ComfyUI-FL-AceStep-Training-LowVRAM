"""
ACE-Step Training Configuration Node

Configures LoRA and training hyperparameters.
"""

import logging
from dataclasses import dataclass, field
from typing import List

logger = logging.getLogger("FL_AceStep_Training")


@dataclass
class LoRAConfig:
    """LoRA adapter configuration."""
    r: int = 64  # LoRA rank
    alpha: int = 128  # LoRA alpha (scaling factor)
    dropout: float = 0.1  # LoRA dropout
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj"
    ])
    bias: str = "none"  # Don't train bias


@dataclass
class TrainingConfig:
    """Training hyperparameters."""
    # Fixed for turbo model
    shift: float = 3.0
    num_inference_steps: int = 8

    # Trainable parameters
    learning_rate: float = 3e-4
    batch_size: int = 1
    gradient_accumulation_steps: int = 1
    max_epochs: int = 1000
    save_every_n_epochs: int = 200
    warmup_steps: int = 100
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0

    # Output
    output_dir: str = "./lora_output"
    seed: int = 42

    # Always bf16
    mixed_precision: str = "bf16"


class FL_AceStep_TrainingConfig:
    """
    Training Configuration

    Configure LoRA adapter and training hyperparameters.

    LoRA Settings:
    - rank: Higher rank = more capacity, more VRAM (typical: 8-128)
    - alpha: Scaling factor, usually 2x rank
    - dropout: Regularization (typical: 0.05-0.2)

    Training Settings:
    - learning_rate: Start with 1e-4 to 3e-4
    - max_epochs: 500-2000 depending on dataset size
    - batch_size: Usually 1 due to VRAM constraints
    - gradient_accumulation: Effective batch = batch_size * accumulation

    Note: The turbo model uses fixed 8-step discrete timesteps with shift=3.0
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # LoRA Config
                "lora_rank": ("INT", {
                    "default": 64,
                    "min": 4,
                    "max": 256,
                    "step": 4,
                }),
                "lora_alpha": ("INT", {
                    "default": 128,
                    "min": 4,
                    "max": 512,
                    "step": 4,
                }),
                "lora_dropout": ("FLOAT", {
                    "default": 0.1,
                    "min": 0.0,
                    "max": 0.5,
                    "step": 0.05,
                }),

                # Training Config
                "learning_rate": ("FLOAT", {
                    "default": 3e-4,
                    "min": 1e-6,
                    "max": 1e-2,
                    "step": 1e-5,
                }),
                "max_epochs": ("INT", {
                    "default": 1000,
                    "min": 100,
                    "max": 10000,
                    "step": 100,
                }),
                "batch_size": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 8,
                    "step": 1,
                }),
                "gradient_accumulation": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 16,
                    "step": 1,
                }),
                "save_every_n_epochs": ("INT", {
                    "default": 200,
                    "min": 50,
                    "max": 1000,
                    "step": 50,
                }),

                "output_dir": ("STRING", {
                    "default": "./lora_output",
                    "multiline": False,
                }),
                "seed": ("INT", {
                    "default": 42,
                    "min": 0,
                    "max": 2147483647,
                }),
            },
            "optional": {
                "warmup_steps": ("INT", {
                    "default": 100,
                    "min": 0,
                    "max": 1000,
                    "step": 10,
                }),
                "weight_decay": ("FLOAT", {
                    "default": 0.01,
                    "min": 0.0,
                    "max": 0.1,
                    "step": 0.001,
                }),
                "max_grad_norm": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.1,
                    "max": 10.0,
                    "step": 0.1,
                }),
                "target_modules": ("STRING", {
                    "default": "q_proj,k_proj,v_proj,o_proj",
                    "multiline": False,
                }),
            }
        }

    RETURN_TYPES = ("ACESTEP_TRAINING_CONFIG",)
    RETURN_NAMES = ("config",)
    FUNCTION = "configure"
    CATEGORY = "FL AceStep/Training"

    def configure(
        self,
        lora_rank,
        lora_alpha,
        lora_dropout,
        learning_rate,
        max_epochs,
        batch_size,
        gradient_accumulation,
        save_every_n_epochs,
        output_dir,
        seed,
        warmup_steps=100,
        weight_decay=0.01,
        max_grad_norm=1.0,
        target_modules="q_proj,k_proj,v_proj,o_proj"
    ):
        """Create training configuration."""
        # Parse target modules
        target_modules_list = [m.strip() for m in target_modules.split(",") if m.strip()]

        # Create LoRA config
        lora_config = LoRAConfig(
            r=lora_rank,
            alpha=lora_alpha,
            dropout=lora_dropout,
            target_modules=target_modules_list,
        )

        # Create training config
        training_config = TrainingConfig(
            learning_rate=learning_rate,
            max_epochs=max_epochs,
            batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation,
            save_every_n_epochs=save_every_n_epochs,
            warmup_steps=warmup_steps,
            weight_decay=weight_decay,
            max_grad_norm=max_grad_norm,
            output_dir=output_dir,
            seed=seed,
        )

        # Log configuration
        logger.info(f"LoRA Config: rank={lora_rank}, alpha={lora_alpha}, dropout={lora_dropout}")
        logger.info(f"Training Config: lr={learning_rate}, epochs={max_epochs}, batch={batch_size}")
        logger.info(f"Output directory: {output_dir}")

        # Return as dictionary
        config = {
            "lora": lora_config,
            "training": training_config,
        }

        return (config,)
