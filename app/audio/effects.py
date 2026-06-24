from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

from config import BEEP_AMPLITUDE, BEEP_FREQ_HZ

from .audio_io import decode_audio_file


def generate_beep(duration_s: float, sample_rate: int,
                  frequency: float = BEEP_FREQ_HZ,
                  amplitude: float = BEEP_AMPLITUDE,
                  fade_ms: float = 5.0) -> np.ndarray:
    """Sine-wave beep with tiny fade-in/out to avoid clicks."""
    n = max(1, int(round(duration_s * sample_rate)))
    t = np.arange(n, dtype=np.float32) / sample_rate
    signal = amplitude * np.sin(2.0 * np.pi * frequency * t).astype(np.float32)
    return _apply_fade(signal, int(fade_ms * 1e-3 * sample_rate))


def _apply_fade(signal: np.ndarray, fade_samples: int) -> np.ndarray:
    """Ramp the first and last ``fade_samples`` up from and down to zero, so a
    generated tone starts and ends without an audible click."""
    fade = min(fade_samples, signal.size // 2)
    if fade > 0:
        ramp = np.linspace(0.0, 1.0, fade, dtype=np.float32)
        signal[:fade] *= ramp
        signal[-fade:] *= ramp[::-1]
    return signal


def generate_silence(duration_s: float, sample_rate: int) -> np.ndarray:
    n = max(1, int(round(duration_s * sample_rate)))
    return np.zeros(n, dtype=np.float32)


@lru_cache(maxsize=32)
def _load_sfx_cached(path: str, sample_rate: int) -> np.ndarray:
    data, sr = decode_audio_file(path)
    if sr != sample_rate:
        data = resample_poly(data, sample_rate, sr).astype(np.float32)
    return data.astype(np.float32)


def load_sfx_clip(path: str | Path, sample_rate: int) -> np.ndarray:
    """The complete sound-effect clip at its natural speed, resampled to
    ``sample_rate``. Returns an empty array only if the file itself is empty."""
    return _load_sfx_cached(str(path), sample_rate).copy()
