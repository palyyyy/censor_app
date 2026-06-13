from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import ClassVar, Type

import numpy as np

from .censor_rules import CensorMode, CensorRule


def _empty() -> np.ndarray:
    return np.zeros(0, dtype=np.float32)


@dataclass
class EffectOptions:
    """Options handed to every effect at construction time."""

    sfx_tail: bool = False  # let a sound effect ring out past the censored region


@dataclass
class EffectAudio:
    """The audio an effect produces for one censored word.

    ``replacement`` overwrites the censored region exactly; ``tail`` (which
    may be empty) is mixed over the audio that follows the region.
    """

    replacement: np.ndarray
    tail: np.ndarray = field(default_factory=_empty)


class CensorEffect(ABC):
    """One way of replacing a censored word in the signal."""

    mode: ClassVar[CensorMode]

    def __init__(self, options: EffectOptions) -> None:
        self.options = options

    @abstractmethod
    def render(self, rule: CensorRule, region_samples: int,
               sample_rate: int) -> EffectAudio:
        """Render the audio for a censored region of ``region_samples`` samples.

        ``replacement`` must contain exactly ``region_samples`` samples.
        """


_EFFECTS: dict[CensorMode, Type[CensorEffect]] = {}


def register_effect(cls: Type[CensorEffect]) -> Type[CensorEffect]:
    """Class decorator: register a concrete effect under its ``mode``."""
    mode = getattr(cls, "mode", None)
    if mode is None:
        raise ValueError(f"Effect {cls!r} must declare a .mode")
    _EFFECTS[mode] = cls
    return cls


def create_effects(options: EffectOptions | None = None) -> dict[CensorMode, CensorEffect]:
    """Instantiate every registered effect with the given options."""
    opts = options or EffectOptions()
    return {mode: cls(opts) for mode, cls in _EFFECTS.items()}
