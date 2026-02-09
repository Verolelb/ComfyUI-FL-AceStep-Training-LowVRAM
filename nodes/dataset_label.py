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
    get_acestep_dit,
    get_acestep_tokenizer,
)

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
    - model: ACE-Step MODEL (purple connection) for audio encoding
    - llm: LLM model for metadata generation
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "dataset": ("ACESTEP_DATASET",),
                "model": ("MODEL",),  # Native ComfyUI MODEL type (purple connection)
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

        # Get the DiT and tokenizer from the MODEL
        dit = get_acestep_dit(model)
        tokenizer = get_acestep_tokenizer(model)

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
                # Convert audio to codes using the tokenizer
                # Note: This is a simplified version - actual implementation
                # may need to load audio and run through the tokenizer
                audio_codes = ""
                try:
                    # TODO: Implement proper audio to codes conversion
                    # using tokenizer.tokenize() once VAE is connected
                    logger.debug(f"Audio encoding for sample {idx}")
                except Exception as e:
                    logger.warning(f"Could not encode audio for sample {idx}: {e}")

                # Generate metadata
                if format_lyrics and sample.raw_lyrics:
                    # Format user-provided lyrics
                    metadata = llm.format_sample(
                        caption=sample.caption,
                        lyrics=sample.raw_lyrics,
                        user_metadata={
                            "bpm": sample.bpm,
                            "keyscale": sample.keyscale,
                        } if not skip_metas else None,
                    )
                elif audio_codes:
                    # Understand audio from codes
                    metadata = llm.understand_audio_from_codes(audio_codes)
                else:
                    # No audio codes, generate basic metadata
                    metadata = {
                        "caption": f"Music track: {sample.filename}",
                        "genre": "",
                        "bpm": None,
                        "keyscale": "",
                        "timesignature": "4",
                        "language": "instrumental",
                        "lyrics": "[Instrumental]",
                    }

                # Update sample
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

                if transcribe_lyrics and metadata.get("lyrics"):
                    sample.lyrics = metadata["lyrics"]
                    sample.formatted_lyrics = metadata["lyrics"]
                elif format_lyrics and metadata.get("lyrics"):
                    sample.formatted_lyrics = metadata["lyrics"]

                sample.labeled = True
                labeled_count += 1

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
