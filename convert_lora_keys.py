#!/usr/bin/env python3
"""
Convert old ACE-Step LoRA files with double base_model.model. prefix to the correct format.

Usage:
    python convert_lora_keys.py input_lora.safetensors output_lora.safetensors

Or convert in place:
    python convert_lora_keys.py input_lora.safetensors
"""

import sys
from pathlib import Path

def convert_lora_keys(input_path: str, output_path: str = None):
    """Convert LoRA keys from old format to new format."""
    from safetensors.torch import load_file, save_file

    if output_path is None:
        # Convert in place - create backup
        input_p = Path(input_path)
        backup_path = input_p.with_suffix('.backup.safetensors')
        output_path = input_path

        # Rename original to backup
        import shutil
        shutil.copy(input_path, backup_path)
        print(f"Created backup: {backup_path}")

    # Load the old LoRA
    state_dict = load_file(input_path if output_path != input_path else str(Path(input_path).with_suffix('.backup.safetensors')))

    print(f"Loaded {len(state_dict)} keys from {input_path}")
    print(f"Sample old keys: {list(state_dict.keys())[:3]}")

    # Convert keys
    new_state_dict = {}
    converted = 0
    for key, value in state_dict.items():
        new_key = key

        # Strip double prefix until only one remains
        while "base_model.model.base_model.model." in new_key:
            new_key = new_key.replace("base_model.model.base_model.model.", "base_model.model.", 1)
            converted += 1

        new_state_dict[new_key] = value

    if converted == 0:
        print("No keys needed conversion - file already has correct format!")
    else:
        print(f"Converted {converted} key prefixes")

    print(f"Sample new keys: {list(new_state_dict.keys())[:3]}")

    # Save the new LoRA
    save_file(new_state_dict, output_path)
    print(f"Saved to {output_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        print("Usage:")
        print("  python convert_lora_keys.py input.safetensors output.safetensors")
        print("  python convert_lora_keys.py input.safetensors  # converts in place (creates .backup)")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) == 3 else None

    convert_lora_keys(input_file, output_file)
