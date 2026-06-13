from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar, Sequence

import numpy as np


@dataclass(frozen=True)
class Word:
    text: str
    start: float
    end: float
    confidence: float | None = None


@dataclass
class Transcript:
    words: list[Word] = field(default_factory=list)
    language: str | None = None

    @property
    def text(self) -> str:
        return " ".join(w.text for w in self.words).strip()


class STTEngine(ABC):
    name: ClassVar[str]
    display_name: ClassVar[str]

    def __init__(self, model: str, language: str | None = "en") -> None:
        self.model = model
        self.language = language

    @classmethod
    def is_available(cls) -> bool:
        return True

    @classmethod
    @abstractmethod
    def available_models(cls) -> Sequence[str]:
        """The models this engine can load."""

    @abstractmethod
    def transcribe_file(self, audio_path: str | Path) -> Transcript:
        """Transcribe an entire audio file and return a Transcript."""

    @abstractmethod
    def transcribe_chunk(
        self,
        pcm_float32: np.ndarray,
        sample_rate: int,
        time_offset: float = 0.0,
    ) -> list[Word]:
        """Transcribe a single audio chunk (mono float32 in [-1, 1]).

        ``time_offset`` is added to every Word's start/end so the caller
        gets absolute timestamps against the live stream. Implementations
        should be fast enough to keep up with the chunk duration.
        """

    def close(self) -> None:
        """Release any model/resources. Default is a no-op."""
