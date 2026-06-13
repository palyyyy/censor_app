from __future__ import annotations

from typing import Iterable, Mapping

import numpy as np

from app.stt.base import Word

from .censor_rules import CensorMode
from .effects import CensorEffect
from .word_matcher import WordMatcher


def apply_censors(
    audio: np.ndarray,
    sample_rate: int,
    words: Iterable[Word],
    matcher: WordMatcher,
    effects: Mapping[CensorMode, CensorEffect],
    padding_ms: float = 40.0,
) -> tuple[np.ndarray, list[Word]]:
    """Return a censored copy of ``audio`` and the list of replaced words.

    Each matched word's region is overwritten by the effect registered for
    its rule's mode. An effect may additionally return a tail, which is
    mixed over the audio that follows the region (a sound effect ringing
    out past the word, for example).
    """
    if audio.ndim != 1:
        raise ValueError("apply_censors expects mono float32 audio")

    out = audio.copy()
    pad = padding_ms * 1e-3
    censored: list[Word] = []

    for w in words:
        rule = matcher.match(w.text)
        if rule is None:
            continue
        effect = effects.get(rule.mode)
        if effect is None:
            continue

        start_s = max(0.0, w.start - pad)
        end_s = min(len(audio) / sample_rate, w.end + pad)
        if end_s <= start_s:
            continue
        start_i = int(round(start_s * sample_rate))
        end_i = min(int(round(end_s * sample_rate)), out.size)
        region_n = end_i - start_i
        if region_n <= 0:
            continue

        rendered = effect.render(rule, region_n, sample_rate)
        out[start_i:end_i] = rendered.replacement[:region_n]
        _mix_tail(out, end_i, rendered.tail)
        censored.append(w)

    return out, censored


def _mix_tail(out: np.ndarray, start_i: int, tail: np.ndarray) -> None:
    """Mix ``tail`` over ``out`` starting at ``start_i``, clipped to [-1, 1]."""
    if tail.size == 0 or start_i >= out.size:
        return
    n = min(tail.size, out.size - start_i)
    mixed = out[start_i:start_i + n] + tail[:n]
    out[start_i:start_i + n] = np.clip(mixed, -1.0, 1.0)
