"""
FL-AceStep-Training - LoRA Training nodes for ACE-Step 1.5 Music Generation

Provides comprehensive training pipeline using NATIVE ComfyUI types:
- MODEL (purple) - ACE-Step DiT wrapped in ModelPatcher
- VAE (red) - ACE-Step VAE (AutoencoderOobleck)
- CLIP (yellow) - ACE-Step text encoder (Qwen3-Embedding)

Features:
- Model loaders with auto-download from HuggingFace
- Dataset management (scan, label, preprocess)
- LoRA training with rich frontend widget
- LoRA management using native ComfyUI patching
"""

import os
import sys
import logging
from pathlib import Path

# Setup logging
logger = logging.getLogger("FL_AceStep_Training")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[FL AceStep] %(message)s"))
    logger.addHandler(handler)

# Get the directory containing this file
EXTENSION_DIR = os.path.dirname(os.path.abspath(__file__))

# Add the ACE-Step project to the Python path for imports
ACESTEP_PROJECT_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(EXTENSION_DIR))), "Git Projects", "ACE-Step-1.5")
if os.path.exists(ACESTEP_PROJECT_PATH) and ACESTEP_PROJECT_PATH not in sys.path:
    sys.path.insert(0, ACESTEP_PROJECT_PATH)
    logger.info(f"Added ACE-Step project to path: {ACESTEP_PROJECT_PATH}")

# Register JavaScript directory for frontend extensions
try:
    import nodes as comfy_nodes
    js_dir = os.path.join(EXTENSION_DIR, "js")
    if os.path.exists(js_dir):
        comfy_nodes.EXTENSION_WEB_DIRS["FL_AceStep_Training"] = js_dir
        logger.info(f"Registered JS extension directory: {js_dir}")
except Exception as e:
    logger.warning(f"Could not register JS extension: {e}")

# Import folder_paths for ComfyUI integration
try:
    import folder_paths

    # Register ACEStep models folder
    acestep_models_path = os.path.join(folder_paths.models_dir, "acestep")
    os.makedirs(acestep_models_path, exist_ok=True)

    # Register with supported extensions
    supported_extensions = {".safetensors", ".bin", ".pt", ".pth", ".ckpt", ".json"}

    if "acestep" not in folder_paths.folder_names_and_paths:
        folder_paths.folder_names_and_paths["acestep"] = ([acestep_models_path], supported_extensions)
    else:
        existing_paths, existing_exts = folder_paths.folder_names_and_paths["acestep"]
        if acestep_models_path not in existing_paths:
            existing_paths.append(acestep_models_path)

    logger.info(f"ACEStep models directory: {acestep_models_path}")

except ImportError:
    logger.warning("folder_paths not available - running outside ComfyUI?")
    acestep_models_path = None

# Import all nodes
try:
    # Loaders
    from .nodes.llm_loader import FL_AceStep_LLMLoader

    # Dataset nodes
    from .nodes.dataset_scan import FL_AceStep_ScanDirectory
    from .nodes.dataset_label import FL_AceStep_LabelSamples
    from .nodes.dataset_preprocess import FL_AceStep_PreprocessDataset

    # Training nodes
    from .nodes.training_config import FL_AceStep_TrainingConfig
    from .nodes.training_ui import FL_AceStep_Train

    NODE_CLASS_MAPPINGS = {
        # Loaders
        "FL_AceStep_LLMLoader": FL_AceStep_LLMLoader,

        # Dataset
        "FL_AceStep_ScanDirectory": FL_AceStep_ScanDirectory,
        "FL_AceStep_LabelSamples": FL_AceStep_LabelSamples,
        "FL_AceStep_PreprocessDataset": FL_AceStep_PreprocessDataset,

        # Training
        "FL_AceStep_TrainingConfig": FL_AceStep_TrainingConfig,
        "FL_AceStep_Train": FL_AceStep_Train,
    }

    NODE_DISPLAY_NAME_MAPPINGS = {
        # Loaders
        "FL_AceStep_LLMLoader": "FL AceStep LLM Loader",

        # Dataset
        "FL_AceStep_ScanDirectory": "FL AceStep Scan Audio Directory",
        "FL_AceStep_LabelSamples": "FL AceStep Auto-Label Samples",
        "FL_AceStep_PreprocessDataset": "FL AceStep Preprocess Dataset",

        # Training
        "FL_AceStep_TrainingConfig": "FL AceStep Training Configuration",
        "FL_AceStep_Train": "FL AceStep Train LoRA",
    }

    # ASCII art banner
    ascii_art = """
     _    ____ _____   ____  _
    / \  / ___| ____|_/ ___|| |_ ___ _ __
   / _ \| |   |  _| |_\___ \| __/ _ \ '_ \\
  / ___ \ |___| |___ _ ___) | ||  __/ |_) |
 /_/   \_\____|_____|_|____/ \__\___| .__/
                                    |_|    Training
"""
    print(f"\033[35m{ascii_art}\033[0m")
    print("FL AceStep Training Custom Nodes Loaded - Version 1.1.0")
    logger.info(f"Loaded {len(NODE_CLASS_MAPPINGS)} nodes successfully")

except Exception as e:
    logger.error(f"Failed to load FL AceStep Training nodes: {e}")
    import traceback
    traceback.print_exc()

    # Provide empty mappings on failure
    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}

# Web directory for frontend
WEB_DIRECTORY = "./js"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
