from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from app.utils.logger import get_logger
from config import TARGET_SAMPLE_RATE

from .base import STTEngine, Transcript, Word
from .registry import register_engine

log = get_logger(__name__)


@register_engine
class FasterWhisperEngine(STTEngine):
    name = "faster-whisper"
    display_name = "faster-whisper (CPU/Metal)"

    @classmethod
    def available_models(cls) -> Sequence[str]:
        return (
            "tiny.en", "tiny",
            "base.en", "base",
            "small.en", "small",
            "medium.en", "medium",
            "large-v3",
            "distil-small.en", "distil-medium.en", "distil-large-v3",
        )

    @classmethod
    def is_available(cls) -> bool:
        try:
            import faster_whisper  # noqa: F401
            return True
        except ImportError:
            return False

    def __init__(self, model: str, language: str | None = "en") -> None:
        super().__init__(model=model, language=language)
        from faster_whisper import WhisperModel

        log.info("Loading faster-whisper model %s ...", model)
        self._model = WhisperModel(
            model,
            device="auto",
            compute_type="int8",
        )
        log.info("faster-whisper model ready")

    def _run(self, audio, time_offset: float = 0.0) -> list[Word]:
        segments, _info = self._model.transcribe(
            audio,
            language=self.language if self.language != "auto" else None,
            word_timestamps=True,
            vad_filter=True,
            beam_size=1,
            condition_on_previous_text=False,
        )
        words: list[Word] = []
        for seg in segments:
            if not seg.words:
                continue
            for w in seg.words:
                if w.start is None or w.end is None:
                    continue
                words.append(Word(
                    text=w.word.strip(),
                    start=float(w.start) + time_offset,
                    end=float(w.end) + time_offset,
                    confidence=float(w.probability) if w.probability is not None else None,
                ))
        return words

    def transcribe_file(self, audio_path: str | Path) -> Transcript:
        words = self._run(str(audio_path))
        return Transcript(words=words, language=self.language)

    def transcribe_chunk(
        self,
        pcm_float32: np.ndarray,
        sample_rate: int,
        time_offset: float = 0.0,
    ) -> list[Word]:
        if sample_rate != TARGET_SAMPLE_RATE:
            from scipy.signal import resample_poly
            pcm_float32 = resample_poly(pcm_float32, TARGET_SAMPLE_RATE, sample_rate).astype(np.float32)
        return self._run(pcm_float32, time_offset=time_offset)
