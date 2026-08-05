"""Kaldi ark/scp features from any HuggingFace speech encoder."""
from .extractor import (
    EXPECTED_SAMPLE_RATE,
    WHISPER_SAMPLES_PER_FRAME,
    WHISPER_WINDOW_SECONDS,
    SSLExtractor,
    preprocess_waveform,
)

__version__ = "0.1.0"
__all__ = [
    "SSLExtractor",
    "preprocess_waveform",
    "EXPECTED_SAMPLE_RATE",
    "WHISPER_WINDOW_SECONDS",
    "WHISPER_SAMPLES_PER_FRAME",
    "__version__",
]
