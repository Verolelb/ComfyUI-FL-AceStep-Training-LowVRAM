# FL AceStep Training

LoRA training nodes for ComfyUI powered by ACE-Step 1.5, the open-source music generation foundation model. Train custom LoRAs to personalize music generation with your own style, voice, or genre — entirely within ComfyUI's node graph.

[![ACE-Step](https://img.shields.io/badge/ACE--Step-Original%20Repo-blue?style=for-the-badge&logo=github&logoColor=white)](https://github.com/ace-step/ACE-Step-1.5)
[![ComfyUI](https://img.shields.io/badge/ComfyUI-Custom%20Nodes-green?style=for-the-badge&logo=github&logoColor=white)](https://github.com/comfyanonymous/ComfyUI)

![Workflow Preview](assets/workflow_preview.png)

## Features

- **End-to-End Training** - Full LoRA training pipeline inside ComfyUI's node graph
- **Dataset Management** - Scan audio directories, auto-label with LLM, edit metadata manually
- **Tiled VAE Encoding** - Handles long audio (30s chunks with overlap) for preprocessing
- **Real-Time Training UI** - Live loss chart, progress bar, and stats widget on the training node
- **Auto Model Download** - Models download automatically from HuggingFace on first use

## Nodes

| Node | Category | Description |
|------|----------|-------------|
| **FL AceStep LLM Loader** | Loaders | Load 5Hz causal LM (0.6B/1.7B/4B) for auto-labeling |
| **FL AceStep Scan Audio Directory** | Dataset | Recursively scan folders for audio files with sidecar metadata |
| **FL AceStep Auto-Label Samples** | Dataset | Generate metadata (caption, BPM, key, genre, lyrics) via LLM |
| **FL AceStep Preprocess Dataset** | Dataset | VAE-encode audio, CLIP-encode text, save as .pt tensors |
| **FL AceStep Training Configuration** | Training | Configure LoRA rank/alpha/dropout and training hyperparameters |
| **FL AceStep Train LoRA** | Training | Run training loop with real-time progress widget |

## Installation

### Manual
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/filliptm/ComfyUI-FL-AceStep-Training.git
cd ComfyUI-FL-AceStep-Training
pip install -r requirements.txt
```

### Frontend (optional rebuild)
```bash
npm install
npm run build
```

The pre-built JS is included in `js/`, so rebuilding is only needed if modifying the training widget UI.

## Quick Start

### Training Pipeline

1. **Load LLM** - Add `FL AceStep LLM Loader` for auto-labeling support
2. **Prepare Dataset** - Use `FL AceStep Scan Audio Directory` to find your audio files
3. **Label** - Connect to `FL AceStep Auto-Label Samples` for LLM-generated metadata
4. **Preprocess** - Run `FL AceStep Preprocess Dataset` to encode audio/text to tensors
5. **Configure** - Set LoRA rank, learning rate, epochs in `FL AceStep Training Configuration`
6. **Train** - Connect everything to `FL AceStep Train LoRA` and execute

### Using Trained LoRAs

Use ComfyUI's native LoRA loading nodes to apply your trained LoRA for inference with the built-in ACE-Step nodes.

## Training Configuration Defaults

| Parameter | Default | Range |
|-----------|---------|-------|
| LoRA Rank | 64 | 4-256 |
| LoRA Alpha | 128 | 4-512 |
| Dropout | 0.1 | 0-0.5 |
| Learning Rate | 3e-4 | — |
| Batch Size | 1 | 1-8 |
| Max Epochs | 1000 | 100-10000 |
| Save Every N Epochs | 200 | — |
| Target Modules | q_proj, k_proj, v_proj, o_proj | — |
| Mixed Precision | bf16 | — |

## Models

| Model | Type | Notes |
|-------|------|-------|
| acestep-v15-turbo | DiT | Recommended for training |
| acestep-v15-turbo-shift1 | DiT | Alternative turbo variant |
| acestep-v15-turbo-shift3 | DiT | Alternative turbo variant |
| acestep-v15-sft | DiT | Supervised fine-tuned |
| acestep-v15-base | DiT | Base model |
| acestep-v15-turbo-continuous | DiT | Continuous timesteps |

Models download automatically on first use from HuggingFace (`ACE-Step/Ace-Step1.5`).

## Supported Audio Formats

`.wav`, `.mp3`, `.flac`, `.ogg`, `.opus`, `.m4a`

Sidecar files are supported for metadata:
- `.txt` files alongside audio for lyrics
- `key_bpm.csv` or `metadata.csv` for BPM, key, and caption data

## Requirements

- Python 3.10+
- NVIDIA GPU with 8GB+ VRAM (bf16 training)
- PyTorch 2.0+
- PEFT (Parameter-Efficient Fine-Tuning)

## License

MIT
