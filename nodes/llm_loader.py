"""
ACE-Step LLM Loader Node

Loads the 5Hz-lm model for audio understanding and auto-labeling.
"""

import os
import logging
import torch

try:
    from comfy.utils import ProgressBar
except ImportError:
    ProgressBar = None

from ..modules.model_downloader import (
    get_acestep_models_dir,
    ensure_lm_model,
)

logger = logging.getLogger("FL_AceStep_Training")

# Available LLM model variants
LLM_MODELS = [
    "acestep-5Hz-lm-1.7B",
    "acestep-5Hz-lm-0.6B",
    "acestep-5Hz-lm-4B",
]

DEVICE_OPTIONS = ["auto", "cuda", "cpu"]
BACKEND_OPTIONS = ["pt", "vllm"]


class ACEStepLLMHandler:
    """
    Handler for the 5Hz-lm model.

    Wraps the LLM for audio understanding and metadata generation.
    """

    def __init__(self, model, tokenizer, device, dtype, model_name):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.dtype = dtype
        self.model_name = model_name

    def understand_audio_from_codes(
        self,
        audio_codes: str,
        temperature: float = 0.3,
        top_k: int = 50,
        top_p: float = 0.95,
        max_new_tokens: int = 2048,
    ):
        """
        Generate metadata and lyrics from audio codes.

        Args:
            audio_codes: String of audio code tokens
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Nucleus sampling
            max_new_tokens: Maximum tokens to generate

        Returns:
            Dictionary with generated metadata
        """
        # Build prompt for understanding
        prompt = self._build_understanding_prompt(audio_codes)

        # Generate
        response = self._generate(
            prompt,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
        )

        # Parse response
        return self._parse_understanding_response(response)

    def format_sample(
        self,
        caption: str,
        lyrics: str,
        user_metadata: dict = None,
        temperature: float = 0.85,
        max_new_tokens: int = 2048,
    ):
        """
        Format user-provided caption and lyrics.

        Args:
            caption: User-provided caption
            lyrics: User-provided lyrics
            user_metadata: Optional metadata dict
            temperature: Sampling temperature
            max_new_tokens: Maximum tokens to generate

        Returns:
            Dictionary with formatted metadata
        """
        # Build prompt for formatting
        prompt = self._build_formatting_prompt(caption, lyrics, user_metadata)

        # Generate
        response = self._generate(
            prompt,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
        )

        # Parse response
        return self._parse_formatting_response(response)

    def _build_understanding_prompt(self, audio_codes: str) -> str:
        """Build prompt for audio understanding."""
        return f"""You are an AI assistant that analyzes music. Given the following audio codes, describe the music.

Audio codes: {audio_codes}

Please provide:
1. A caption describing the music style, mood, and instrumentation
2. The estimated BPM (tempo)
3. The key/scale (e.g., C Major, Am)
4. The time signature (2, 3, 4, or 6)
5. The language of vocals (or "instrumental" if no vocals)
6. Genre tags
7. Lyrics (if applicable, or "[Instrumental]")

Format your response as:
Caption: [description]
BPM: [number]
Key: [key/scale]
TimeSignature: [number]
Language: [language]
Genre: [genre tags]
Lyrics:
[lyrics or [Instrumental]]
"""

    def _build_formatting_prompt(self, caption: str, lyrics: str, metadata: dict) -> str:
        """Build prompt for sample formatting."""
        meta_str = ""
        if metadata:
            meta_str = f"\nExisting metadata: {metadata}"

        return f"""You are an AI assistant that formats music metadata. Please format the following information.

Caption: {caption}
Lyrics: {lyrics}{meta_str}

Please provide formatted output as:
Caption: [enhanced caption]
BPM: [estimated number]
Key: [key/scale]
TimeSignature: [number]
Language: [language]
Genre: [genre tags]
Lyrics:
[formatted lyrics]
"""

    def _generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.95,
        max_new_tokens: int = 2048,
    ) -> str:
        """Generate response from LLM."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Remove the prompt from response
        if response.startswith(prompt):
            response = response[len(prompt):]

        return response.strip()

    def _parse_understanding_response(self, response: str) -> dict:
        """Parse understanding response into metadata dict."""
        result = {
            "caption": "",
            "bpm": None,
            "keyscale": "",
            "timesignature": "4",
            "language": "instrumental",
            "genre": "",
            "lyrics": "[Instrumental]",
        }

        lines = response.split("\n")
        current_field = None
        lyrics_lines = []

        for line in lines:
            line = line.strip()
            if not line:
                continue

            if line.lower().startswith("caption:"):
                result["caption"] = line.split(":", 1)[1].strip()
            elif line.lower().startswith("bpm:"):
                try:
                    result["bpm"] = int(line.split(":", 1)[1].strip())
                except:
                    pass
            elif line.lower().startswith("key:"):
                result["keyscale"] = line.split(":", 1)[1].strip()
            elif line.lower().startswith("timesignature:"):
                result["timesignature"] = line.split(":", 1)[1].strip()
            elif line.lower().startswith("language:"):
                result["language"] = line.split(":", 1)[1].strip().lower()
            elif line.lower().startswith("genre:"):
                result["genre"] = line.split(":", 1)[1].strip()
            elif line.lower().startswith("lyrics:"):
                current_field = "lyrics"
            elif current_field == "lyrics":
                lyrics_lines.append(line)

        if lyrics_lines:
            result["lyrics"] = "\n".join(lyrics_lines)

        return result

    def _parse_formatting_response(self, response: str) -> dict:
        """Parse formatting response into metadata dict."""
        # Same parsing logic as understanding
        return self._parse_understanding_response(response)


class FL_AceStep_LLMLoader:
    """
    Load ACE-Step LLM (5Hz Language Model)

    Loads the 5Hz-lm model for audio understanding and auto-labeling.
    This model is used to automatically generate captions, metadata, and lyrics
    from audio samples during dataset preparation.

    Available models:
    - acestep-5Hz-lm-1.7B (default, balanced)
    - acestep-5Hz-lm-0.6B (lightweight)
    - acestep-5Hz-lm-4B (high quality, requires more VRAM)
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (LLM_MODELS, {"default": "acestep-5Hz-lm-1.7B"}),
                "device": (DEVICE_OPTIONS, {"default": "auto"}),
                "backend": (BACKEND_OPTIONS, {"default": "pt"}),
            },
            "optional": {
                "checkpoint_path": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "Leave empty for auto-download"
                }),
            }
        }

    RETURN_TYPES = ("ACESTEP_LLM",)
    RETURN_NAMES = ("llm",)
    FUNCTION = "load"
    CATEGORY = "FL AceStep/Loaders"

    def load(self, model_name, device, backend, checkpoint_path=""):
        """Load the LLM model."""
        logger.info(f"Loading ACE-Step LLM: {model_name}")

        # Progress bar
        pbar = ProgressBar(2) if ProgressBar else None

        # Determine models directory
        if checkpoint_path and checkpoint_path.strip():
            models_dir = checkpoint_path.strip()
        else:
            models_dir = get_acestep_models_dir()

        # Step 1: Ensure LLM is downloaded
        if pbar:
            pbar.update(1)

        success, status = ensure_lm_model(model_name, models_dir)
        if not success:
            raise RuntimeError(f"Failed to ensure LLM: {status}")

        logger.info(status)

        # Determine device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        # Step 2: Load the LLM
        if pbar:
            pbar.update(1)

        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            lm_path = os.path.join(models_dir, model_name)

            logger.info(f"Loading LLM from {lm_path}")
            logger.info(f"Device: {device}, Backend: {backend}")

            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(lm_path)

            # Determine dtype based on device
            if device == "cuda":
                torch_dtype = torch.bfloat16
            else:
                torch_dtype = torch.float32

            # Load model
            if backend == "vllm":
                # vLLM backend (if available)
                try:
                    from vllm import LLM
                    model = LLM(model=lm_path)
                    logger.info("Using vLLM backend")
                except ImportError:
                    logger.warning("vLLM not available, falling back to PyTorch")
                    backend = "pt"

            if backend == "pt":
                # PyTorch backend
                model = AutoModelForCausalLM.from_pretrained(
                    lm_path,
                    torch_dtype=torch_dtype,
                    device_map=device if device != "cpu" else None,
                )
                if device == "cpu":
                    model = model.to(device)
                model.eval()

        except Exception as e:
            logger.exception("Failed to load LLM")
            raise RuntimeError(f"Failed to load LLM: {str(e)}")

        # Create handler
        handler = ACEStepLLMHandler(
            model=model,
            tokenizer=tokenizer,
            device=device,
            dtype=torch_dtype if backend == "pt" else None,
            model_name=model_name,
        )

        logger.info(f"LLM '{model_name}' loaded successfully")

        return (handler,)
