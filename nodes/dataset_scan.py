"""
ACE-Step Dataset Scan Node

Scans a directory for audio files and creates a dataset for training.
"""

import os
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

try:
    import torchaudio
    TORCHAUDIO_AVAILABLE = True
except ImportError:
    TORCHAUDIO_AVAILABLE = False

try:
    import soundfile
    SOUNDFILE_AVAILABLE = True
except ImportError:
    SOUNDFILE_AVAILABLE = False

try:
    from comfy.utils import ProgressBar
except ImportError:
    ProgressBar = None

logger = logging.getLogger("FL_AceStep_Training")

# Supported audio formats
SUPPORTED_AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".opus", ".m4a"}


@dataclass
class AudioSample:
    """Represents a single audio sample in the dataset."""
    id: str
    audio_path: str
    filename: str
    caption: str = ""
    genre: str = ""
    lyrics: str = ""
    raw_lyrics: str = ""  # Original user-provided lyrics from .txt
    formatted_lyrics: str = ""  # LM-formatted lyrics
    bpm: Optional[int] = None
    keyscale: str = ""
    timesignature: str = "4"
    duration: float = 0.0
    language: str = "instrumental"
    is_instrumental: bool = True
    custom_tag: str = ""
    labeled: bool = False
    prompt_override: Optional[str] = None  # "caption" or "genre"


@dataclass
class DatasetMetadata:
    """Metadata for the entire dataset."""
    directory: str = ""
    all_instrumental: bool = True
    custom_tag: str = ""
    tag_position: str = "prepend"  # prepend, append, replace
    genre_ratio: int = 0  # % of samples to use genre vs caption


class ACEStepDatasetBuilder:
    """
    Builder for ACE-Step training datasets.

    Handles scanning directories, loading metadata, and managing samples.
    """

    def __init__(self):
        self.samples: List[AudioSample] = []
        self.metadata = DatasetMetadata()

    def scan_directory(self, directory: str, pbar=None) -> tuple:
        """
        Scan a directory for audio files.

        Args:
            directory: Path to scan
            pbar: Optional progress bar instance

        Returns:
            Tuple of (samples list, status message)
        """
        if not directory or not os.path.isdir(directory):
            return [], f"Invalid directory: {directory}"

        self.metadata.directory = directory
        self.samples = []

        directory_path = Path(directory)
        audio_files = []

        # Find all audio files
        for ext in SUPPORTED_AUDIO_EXTENSIONS:
            audio_files.extend(directory_path.rglob(f"*{ext}"))

        # Sort by name
        audio_files = sorted(audio_files)

        if not audio_files:
            return [], f"No audio files found in {directory}"

        # Update progress bar total if available
        if pbar is not None:
            pbar.update_absolute(0, len(audio_files))

        # Load CSV metadata if available
        csv_metadata = self._load_csv_metadata(directory_path)

        # Process each audio file
        for i, audio_path in enumerate(audio_files):
            sample = self._create_sample(i, audio_path, csv_metadata)
            self.samples.append(sample)
            if pbar is not None:
                pbar.update(1)

        status = f"Found {len(self.samples)} audio files in {directory}"
        logger.info(status)

        return self.samples, status

    def _load_csv_metadata(self, directory: Path) -> Dict[str, Dict]:
        """Load metadata from CSV file if present."""
        metadata = {}

        # Look for key_bpm.csv or metadata.csv
        csv_files = ["key_bpm.csv", "metadata.csv"]

        for csv_name in csv_files:
            csv_path = directory / csv_name
            if csv_path.exists():
                try:
                    import csv
                    with open(csv_path, "r", encoding="utf-8") as f:
                        reader = csv.DictReader(f)
                        for row in reader:
                            # Get filename (could be various column names)
                            filename = row.get("filename") or row.get("file") or row.get("name")
                            if filename:
                                metadata[filename] = row
                    logger.info(f"Loaded metadata from {csv_path}")
                except Exception as e:
                    logger.warning(f"Failed to load CSV metadata: {e}")

        return metadata

    def _create_sample(
        self,
        index: int,
        audio_path: Path,
        csv_metadata: Dict[str, Dict]
    ) -> AudioSample:
        """Create an AudioSample from an audio file."""
        filename = audio_path.name
        stem = audio_path.stem

        # Get duration - prefer soundfile as it's more reliable
        duration = 0.0
        if SOUNDFILE_AVAILABLE:
            try:
                info = soundfile.info(str(audio_path))
                duration = info.duration
            except Exception as e:
                logger.warning(f"Could not get duration for {filename} with soundfile: {e}")
        elif TORCHAUDIO_AVAILABLE:
            try:
                info = torchaudio.info(str(audio_path))
                duration = info.num_frames / info.sample_rate
            except Exception as e:
                logger.warning(f"Could not get duration for {filename}: {e}")

        # Load lyrics from .txt file if present
        lyrics = ""
        txt_path = audio_path.with_suffix(".txt")
        if txt_path.exists():
            try:
                with open(txt_path, "r", encoding="utf-8") as f:
                    lyrics = f.read().strip()
            except Exception as e:
                logger.warning(f"Could not load lyrics for {filename}: {e}")

        # Get CSV metadata
        csv_data = csv_metadata.get(filename, {})

        # Extract metadata from CSV
        bpm = None
        if "bpm" in csv_data or "BPM" in csv_data:
            try:
                bpm = int(csv_data.get("bpm") or csv_data.get("BPM"))
            except:
                pass

        keyscale = csv_data.get("key") or csv_data.get("Key") or csv_data.get("keyscale") or ""
        caption = csv_data.get("caption") or csv_data.get("Caption") or csv_data.get("description") or ""

        # Check if instrumental
        is_instrumental = not lyrics or lyrics.lower() == "[instrumental]"

        return AudioSample(
            id=f"sample_{index:04d}",
            audio_path=str(audio_path),
            filename=filename,
            caption=caption,
            genre="",
            lyrics=lyrics if not is_instrumental else "[Instrumental]",
            raw_lyrics=lyrics,
            bpm=bpm,
            keyscale=keyscale,
            duration=duration,
            is_instrumental=is_instrumental,
            language="instrumental" if is_instrumental else "unknown",
        )

    def get_sample(self, index: int) -> Optional[AudioSample]:
        """Get a sample by index."""
        if 0 <= index < len(self.samples):
            return self.samples[index]
        return None

    def update_sample(self, index: int, **kwargs) -> tuple:
        """Update a sample's metadata."""
        if index < 0 or index >= len(self.samples):
            return None, f"Invalid index: {index}"

        sample = self.samples[index]

        for key, value in kwargs.items():
            if hasattr(sample, key) and value is not None and value != "":
                setattr(sample, key, value)

        return sample, f"Updated sample {index}"

    def set_custom_tag(self, tag: str, position: str = "prepend"):
        """Set custom tag for all samples."""
        self.metadata.custom_tag = tag
        self.metadata.tag_position = position

        for sample in self.samples:
            sample.custom_tag = tag

    def set_all_instrumental(self, instrumental: bool):
        """Set all samples as instrumental."""
        self.metadata.all_instrumental = instrumental

        for sample in self.samples:
            sample.is_instrumental = instrumental
            if instrumental:
                sample.lyrics = "[Instrumental]"
                sample.language = "instrumental"

    def get_labeled_count(self) -> int:
        """Get number of labeled samples."""
        return sum(1 for s in self.samples if s.labeled or s.caption)

    def get_unlabeled_count(self) -> int:
        """Get number of unlabeled samples."""
        return len(self.samples) - self.get_labeled_count()

    def to_dict(self) -> Dict[str, Any]:
        """Convert dataset to dictionary for serialization."""
        return {
            "metadata": {
                "directory": self.metadata.directory,
                "all_instrumental": self.metadata.all_instrumental,
                "custom_tag": self.metadata.custom_tag,
                "tag_position": self.metadata.tag_position,
                "genre_ratio": self.metadata.genre_ratio,
            },
            "samples": [
                {
                    "id": s.id,
                    "audio_path": s.audio_path,
                    "filename": s.filename,
                    "caption": s.caption,
                    "genre": s.genre,
                    "lyrics": s.lyrics,
                    "bpm": s.bpm,
                    "keyscale": s.keyscale,
                    "timesignature": s.timesignature,
                    "duration": s.duration,
                    "language": s.language,
                    "is_instrumental": s.is_instrumental,
                    "custom_tag": s.custom_tag,
                    "labeled": s.labeled,
                }
                for s in self.samples
            ]
        }


class FL_AceStep_ScanDirectory:
    """
    Scan Audio Directory

    Scans a directory for audio files and creates a dataset for training.
    Supports WAV, MP3, FLAC, OGG, OPUS, and M4A formats.

    The node will also:
    - Load accompanying .txt files as lyrics
    - Load metadata from key_bpm.csv if present
    - Calculate audio durations

    Outputs:
    - dataset: The dataset builder object for use with other nodes
    - sample_count: Number of audio files found
    - status: Status message
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "directory": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "Path to audio directory"
                }),
                "all_instrumental": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "custom_tag": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "LoRA activation tag (e.g., 'my_style')"
                }),
                "tag_position": (["prepend", "append", "replace"], {"default": "prepend"}),
            }
        }

    RETURN_TYPES = ("ACESTEP_DATASET", "INT", "STRING")
    RETURN_NAMES = ("dataset", "sample_count", "status")
    FUNCTION = "scan"
    CATEGORY = "FL AceStep/Dataset"

    def scan(self, directory, all_instrumental, custom_tag="", tag_position="prepend"):
        """Scan the directory for audio files."""
        logger.info(f"Scanning directory: {directory}")

        # Create progress bar (will be updated with actual count during scan)
        pbar = ProgressBar(1) if ProgressBar else None

        # Create dataset builder
        builder = ACEStepDatasetBuilder()

        # Scan directory with progress bar
        samples, status = builder.scan_directory(directory, pbar=pbar)

        if not samples:
            return (builder, 0, status)

        # Set custom tag if provided
        if custom_tag and custom_tag.strip():
            builder.set_custom_tag(custom_tag.strip(), tag_position)

        # Set instrumental flag
        builder.set_all_instrumental(all_instrumental)

        sample_count = len(samples)
        status = f"Found {sample_count} audio files"

        if custom_tag:
            status += f" (tag: '{custom_tag}')"

        logger.info(status)

        return (builder, sample_count, status)
