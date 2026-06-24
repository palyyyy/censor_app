from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf


def decode_audio_file(path: str | Path) -> tuple[np.ndarray, int]:
    """Decode an audio file to a mono float32 array in [-1, 1] and its native
    sample rate.

    libsndfile (via soundfile) handles wav/flac/ogg directly; for formats it
    cannot read, such as mp3, this falls back to pydub/ffmpeg.
    """
    try:
        data, sr = sf.read(str(path), dtype="float32", always_2d=False)
    except RuntimeError:
        from pydub import AudioSegment
        seg = AudioSegment.from_file(str(path))
        sr = seg.frame_rate
        samples = np.array(seg.get_array_of_samples(), dtype=np.float32)
        if seg.channels > 1:
            samples = samples.reshape(-1, seg.channels).mean(axis=1)
        max_val = float(1 << (8 * seg.sample_width - 1))
        data = samples / max_val
    if data.ndim > 1:
        data = data.mean(axis=1)
    return data.astype(np.float32), int(sr)
