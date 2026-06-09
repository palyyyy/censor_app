from __future__ import annotations

from typing import Iterable

import numpy as np

from app.audio.effects import generate_beep, generate_silence, load_sfx
from app.stt.base import Word

from .censor_rules import CensorMode
from .word_matcher import WordMatcher


def apply_censors(
    audio: np.ndarray,
    sample_rate: int,
    words: Iterable[Word],
    matcher: WordMatcher,
    padding_ms: float = 40.0,
) -> tuple[np.ndarray, list[Word]]:
    if audio.ndim != 1:
        raise ValueError("apply_censors expects mono float32 audio")

    out = audio.copy()
    pad = padding_ms * 1e-3
    censored: list[Word] = []

    for w in words:
        rule = matcher.match(w.text)
        if rule is None:
            continue

        start_s = max(0.0, w.start - pad)
        end_s = min(len(audio) / sample_rate, w.end + pad)
        if end_s <= start_s:
            continue

        start_i = int(round(start_s * sample_rate))
        end_i = int(round(end_s * sample_rate))
        end_i = min(end_i, out.size)
        region_n = end_i - start_i
        if region_n <= 0:
            continue

        dur = region_n / sample_rate
        if rule.mode == CensorMode.BEEP:
            replacement = generate_beep(dur, sample_rate)
        elif rule.mode == CensorMode.SILENCE:
            replacement = generate_silence(dur, sample_rate)
        elif rule.mode == CensorMode.SFX and rule.sfx_path:
            replacement = load_sfx(rule.sfx_path, dur, sample_rate, stretch=True)
        else:
            replacement = generate_beep(dur, sample_rate)

        # Size-match defensively
        if replacement.size < region_n:
            replacement = np.pad(replacement, (0, region_n - replacement.size))
        else:
            replacement = replacement[:region_n]

        out[start_i:end_i] = replacement
        censored.append(w)

    return out, censored
