from __future__ import annotations

import platform
from pathlib import Path
from typing import Sequence

import numpy as np

from app.utils.logger import get_logger

from .base import STTEngine, Transcript, Word
from .registry import register_engine

log = get_logger(__name__)


def _is_apple_silicon() -> bool:
    return platform.system() == "Darwin" and platform.machine() == "arm64"


@register_engine
class MLXWhisperEngine(STTEngine):
    name = "mlx-whisper"
    display_name = "MLX Whisper (Apple Silicon)"
    @classmethod
    def available_models(cls) -> Sequence[str]:
        return (
            "mlx-community/whisper-tiny.en-mlx",
            "mlx-community/whisper-base.en-mlx",
            "mlx-community/whisper-small.en-mlx",
            "mlx-community/whisper-medium.en-mlx",
            "mlx-community/whisper-large-v3-mlx",
            "mlx-community/distil-whisper-large-v3",
        )

    @classmethod
    def is_available(cls) -> bool:
        if not _is_apple_silicon():
            return False
        try:
            import mlx_whisper  # noqa: F401
            return True
        except ImportError:
            return False

    def __init__(self, model: str, language: str | None = "en") -> None:
        super().__init__(model=model, language=language)
        import mlx_whisper  # noqa: F401
        self._path_or_repo = model
        log.info("MLX Whisper engine ready (model will load on first use): %s", model)

    def _to_words(self, result: dict, time_offset: float = 0.0) -> list[Word]:
        """mlx_whisper returns segments with optional word timing."""
        words: list[Word] = []
        for seg in result.get("segments", []):
            seg_words = seg.get("words") or []
            if not seg_words:
                text = seg.get("text", "").strip()
                if not text:
                    continue
                tokens = text.split()
                if not tokens:
                    continue
                seg_start = float(seg.get("start", 0.0))
                seg_end = float(seg.get("end", seg_start))
                dur = max(seg_end - seg_start, 1e-3)
                step = dur / len(tokens)
                for i, tok in enumerate(tokens):
                    words.append(Word(
                        text=tok,
                        start=seg_start + i * step + time_offset,
                        end=seg_start + (i + 1) * step + time_offset,
                        confidence=None,
                    ))
                continue
            for w in seg_words:
                words.append(Word(
                    text=str(w.get("word", "")).strip(),
                    start=float(w.get("start", 0.0)) + time_offset,
                    end=float(w.get("end", 0.0)) + time_offset,
                    confidence=float(w["probability"]) if "probability" in w else None,
                ))
        return words

    def _transcribe(self, audio, time_offset: float = 0.0) -> list[Word]:
        import mlx_whisper

        kwargs = dict(
            path_or_hf_repo=self._path_or_repo,
            word_timestamps=True,
        )
        if self.language and self.language != "auto":
            kwargs["language"] = self.language

        result = mlx_whisper.transcribe(audio, **kwargs)
        return self._to_words(result, time_offset=time_offset)

    def transcribe_file(self, audio_path: str | Path) -> Transcript:
        words = self._transcribe(str(audio_path))
        return Transcript(words=words, language=self.language)

    def transcribe_chunk(
        self,
        pcm_float32: np.ndarray,
        sample_rate: int,
        time_offset: float = 0.0,
    ) -> list[Word]:
        if sample_rate != 16000:
            from scipy.signal import resample_poly
            pcm_float32 = resample_poly(pcm_float32, 16000, sample_rate).astype(np.float32)
        return self._transcribe(pcm_float32.astype(np.float32), time_offset=time_offset)
