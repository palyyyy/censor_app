from __future__ import annotations

import numpy as np

from app.censor.censor_rules import CensorMode, CensorRule
from app.censor.effects import CensorEffect, EffectAudio, register_effect

from .effects import generate_beep, generate_silence, load_sfx_clip


def _exactly(signal: np.ndarray, n: int) -> np.ndarray:
    """``signal`` padded with silence or truncated to exactly ``n`` samples."""
    if signal.size < n:
        return np.pad(signal, (0, n - signal.size))
    return signal[:n]


@register_effect
class BeepEffect(CensorEffect):
    mode = CensorMode.BEEP

    def render(self, rule: CensorRule, region_samples: int,
               sample_rate: int) -> EffectAudio:
        beep = generate_beep(region_samples / sample_rate, sample_rate)
        return EffectAudio(replacement=_exactly(beep, region_samples))


@register_effect
class SilenceEffect(CensorEffect):
    mode = CensorMode.SILENCE

    def render(self, rule: CensorRule, region_samples: int,
               sample_rate: int) -> EffectAudio:
        return EffectAudio(replacement=np.zeros(region_samples, dtype=np.float32))


@register_effect
class SfxEffect(CensorEffect):
    mode = CensorMode.SFX

    def render(self, rule: CensorRule, region_samples: int,
               sample_rate: int) -> EffectAudio:
        clip = self._load_clip(rule, region_samples, sample_rate)
        replacement = _exactly(clip, region_samples)
        tail = clip[region_samples:] if self.options.sfx_tail else np.zeros(0, dtype=np.float32)
        return EffectAudio(replacement=replacement, tail=tail)

    @staticmethod
    def _load_clip(rule: CensorRule, region_samples: int,
                   sample_rate: int) -> np.ndarray:
        """The configured clip, or a safe fallback that never leaves the word
        audible: a beep when no clip is set, silence when the file is empty."""
        if not rule.sfx_path:
            return generate_beep(region_samples / sample_rate, sample_rate)
        clip = load_sfx_clip(rule.sfx_path, sample_rate)
        if clip.size == 0:
            return generate_silence(region_samples / sample_rate, sample_rate)
        return clip
