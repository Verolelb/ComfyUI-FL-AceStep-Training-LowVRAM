"""
ACE-Step Dataset Label Node

Auto-labels audio samples using the LLM for metadata generation.
Uses native ComfyUI MODEL type for the ACE-Step model.
"""

import logging

try:
    from comfy.utils import ProgressBar
except ImportError:
    ProgressBar = None

from ..modules.acestep_model import (
    is_acestep_model,
    get_acestep_tokenizer,
)
from ..modules.audio_utils import audio_to_codes

logger = logging.getLogger("FL_AceStep_Training")


class FL_AceStep_LabelSamples:
    """
    Auto-Label Samples

    Uses the 5Hz-lm model to automatically generate metadata for audio samples.
    This includes:
    - Caption/description
    - Genre tags
    - BPM (tempo)
    - Key/scale
    - Time signature
    - Language
    - Lyrics (transcription or formatting)

    Requires:
    - dataset: Dataset from Scan Directory node
    - model: ACE-Step MODEL (purple connection) for audio tokenization
    - vae: ACE-Step VAE (red connection) for audio encoding
    - llm: LLM model for metadata generation
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "dataset": ("ACESTEP_DATASET",),
                "model": ("MODEL",),  # Native ComfyUI MODEL type (purple connection)
                "vae": ("VAE",),  # Native ComfyUI VAE type (red connection)
                "llm": ("ACESTEP_LLM",),
            },
            "optional": {
                "skip_metas": ("BOOLEAN", {
                    "default": False,
                    "label": "Skip BPM/Key/TimeSig (generate caption only)"
                }),
                "only_unlabeled": ("BOOLEAN", {
                    "default": False,
                    "label": "Only process samples without captions"
                }),
                "format_lyrics": ("BOOLEAN", {
                    "default": False,
                    "label": "Format user-provided lyrics with LLM"
                }),
                "transcribe_lyrics": ("BOOLEAN", {
                    "default": False,
                    "label": "Transcribe lyrics from audio"
                }),
            }
        }

    RETURN_TYPES = ("ACESTEP_DATASET", "INT", "STRING")
    RETURN_NAMES = ("dataset", "labeled_count", "status")
    FUNCTION = "label"
    CATEGORY = "FL AceStep/Dataset"

    def label(
        self,
        dataset,
        model,  # ComfyUI MODEL (ModelPatcher)
        vae,  # ComfyUI VAE
        llm,
        skip_metas=False,
        only_unlabeled=False,
        format_lyrics=False,
        transcribe_lyrics=False
    ):
        """Label all samples in the dataset."""
        logger.info("Starting auto-labeling...")

        # Verify this is an ACE-Step model
        if not is_acestep_model(model):
            return (dataset, 0, "Error: Model is not an ACE-Step model")

        # Get the tokenizer from the MODEL for audio-to-codes conversion
        tokenizer = get_acestep_tokenizer(model)

        # Move tokenizer to GPU once before the loop to avoid per-sample overhead
        import torch
        target_device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        tokenizer_device = next(tokenizer.parameters()).device
        if tokenizer_device.type != target_device.type:
            logger.info(f"Moving audio tokenizer from {tokenizer_device} to {target_device}")
            tokenizer.to(target_device)

        # device/dtype for audio_to_codes (tokenizer location)
        device = target_device
        dtype = next(tokenizer.parameters()).dtype

        samples = dataset.samples
        if not samples:
            return (dataset, 0, "No samples to label")

        # Filter samples if only_unlabeled
        samples_to_label = []
        for i, sample in enumerate(samples):
            if only_unlabeled and (sample.labeled or sample.caption):
                continue
            samples_to_label.append((i, sample))

        if not samples_to_label:
            return (dataset, 0, "All samples already labeled")

        # Progress bar
        pbar = ProgressBar(len(samples_to_label)) if ProgressBar else None

        labeled_count = 0
        errors = []

        for idx, sample in samples_to_label:
            try:
                metadata = None

                # Path 1: Format user-provided lyrics with LLM
                if format_lyrics and sample.raw_lyrics:
                    logger.info(f"Formatting lyrics for sample {idx}: {sample.filename}")
                    metadata = llm.format_sample(
                        caption=sample.caption,
                        lyrics=sample.raw_lyrics,
                    )

                else:
                    # Path 2: Understand audio from codes
                    # Convert audio to discrete codes via VAE + tokenizer
                    logger.info(f"Encoding audio to codes for sample {idx}: {sample.filename}")
                    try:
                        codes = audio_to_codes(
                            vae=vae,
                            tokenizer=tokenizer,
                            audio_path=sample.audio_path,
                            device=device,
                            dtype=dtype,
                            max_duration=30.0,
                        )
                    except Exception as e:
                        logger.warning(f"Audio encoding failed for sample {idx}: {e}")
                        codes = ""

                    if codes:
                        logger.info(
                            f"Generated {len(codes)} chars of audio codes for sample {idx}, "
                            f"calling LLM..."
                        )
                        metadata = llm.understand_audio_from_codes(codes)
                    else:
                        logger.warning(
                            f"No audio codes for sample {idx}, skipping LLM labeling"
                        )

                # Apply metadata to sample
                if metadata is None:
                    errors.append(f"Sample {idx}: no metadata generated")
                    if pbar:
                        pbar.update(1)
                    continue

                if metadata.get("caption"):
                    sample.caption = metadata["caption"]
                if metadata.get("genre"):
                    sample.genre = metadata["genre"]

                if not skip_metas:
                    if metadata.get("bpm") and sample.bpm is None:
                        sample.bpm = metadata["bpm"]
                    if metadata.get("keyscale") and not sample.keyscale:
                        sample.keyscale = metadata["keyscale"]
                    if metadata.get("timesignature"):
                        sample.timesignature = metadata["timesignature"]

                if metadata.get("language"):
                    sample.language = metadata["language"]
                    sample.is_instrumental = metadata["language"].lower() == "instrumental"

                # Update lyrics from LLM output
                if metadata.get("lyrics") and metadata["lyrics"] != "[Instrumental]":
                    if transcribe_lyrics or format_lyrics:
                        sample.lyrics = metadata["lyrics"]
                        sample.formatted_lyrics = metadata["lyrics"]
                        sample.is_instrumental = False

                sample.labeled = True
                labeled_count += 1

                logger.info(
                    f"Sample {idx} labeled: caption='{sample.caption[:60]}...', "
                    f"bpm={sample.bpm}, key={sample.keyscale}"
                )

            except Exception as e:
                error_msg = f"Error labeling sample {idx}: {str(e)}"
                logger.warning(error_msg)
                errors.append(error_msg)

            if pbar:
                pbar.update(1)

        # Build status message
        status = f"Labeled {labeled_count}/{len(samples_to_label)} samples"
        if errors:
            status += f" ({len(errors)} errors)"

        logger.info(status)

        return (dataset, labeled_count, status)
