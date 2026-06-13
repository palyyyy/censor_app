from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from config import BEEP_AMPLITUDE, BEEP_FREQ_HZ


def generate_beep(duration_s: float, sample_rate: int,
                  frequency: float = BEEP_FREQ_HZ,
                  amplitude: float = BEEP_AMPLITUDE,
                  fade_ms: float = 5.0) -> np.ndarray:
    """Sine-wave beep with tiny fade-in/out to avoid clicks."""
    n = max(1, int(round(duration_s * sample_rate)))
    t = np.arange(n, dtype=np.float32) / sample_rate
    signal = amplitude * np.sin(2.0 * np.pi * frequency * t).astype(np.float32)

    fade_samples = min(int(fade_ms * 1e-3 * sample_rate), n // 2)
    if fade_samples > 0:
        ramp = np.linspace(0.0, 1.0, fade_samples, dtype=np.float32)
        signal[:fade_samples] *= ramp
        signal[-fade_samples:] *= ramp[::-1]
    return signal


def generate_silence(duration_s: float, sample_rate: int) -> np.ndarray:
    n = max(1, int(round(duration_s * sample_rate)))
    return np.zeros(n, dtype=np.float32)


@lru_cache(maxsize=32)
def _load_sfx_cached(path: str, sample_rate: int) -> np.ndarray:
    try:
        data, sr = sf.read(path, dtype="float32", always_2d=False)
    except Exception:
        from pydub import AudioSegment
        seg = AudioSegment.from_file(path)
        sr = seg.frame_rate
        samples = np.array(seg.get_array_of_samples(), dtype=np.float32)
        if seg.channels > 1:
            samples = samples.reshape(-1, seg.channels).mean(axis=1)
        max_val = float(1 << (8 * seg.sample_width - 1))
        data = samples / max_val
    if data.ndim > 1:
        data = data.mean(axis=1)
    if sr != sample_rate:
        data = resample_poly(data, sample_rate, sr).astype(np.float32)
    return data.astype(np.float32)


def load_sfx_clip(path: str | Path, sample_rate: int) -> np.ndarray:
    """The complete sound-effect clip at its natural speed, resampled to
    ``sample_rate``. Returns an empty array only if the file itself is empty."""
    return _load_sfx_cached(str(path), sample_rate).copy()
