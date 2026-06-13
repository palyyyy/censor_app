from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import sounddevice as sd

import app.audio.censor_effects  # noqa: F401  (registers the concrete effects)
from app.censor import CensorList, WordMatcher
from app.censor.effects import EffectOptions, create_effects
from app.stt.base import STTEngine, Word
from app.utils.logger import get_logger

log = get_logger(__name__)

_OVERLAP_SECONDS = 0.25  # window overlap that protects words on chunk boundaries


@dataclass
class LiveConfig:
    sample_rate: int = 16000
    block_size: int = 1024
    chunk_seconds: float = 1.0
    lookahead_seconds: float = 2.0
    input_device: int | str | None = None
    output_device: int | str | None = None
    padding_ms: float = 50.0
    sfx_tail: bool = False


@dataclass
class _ActiveEffect:
    """A fully rendered effect anchored at an absolute stream position.

    The first ``replace_samples`` samples of ``audio`` overwrite the signal
    (the censored region itself); anything beyond that is a tail that is
    mixed over the audio that follows.
    """

    start_sample: int
    replace_samples: int
    audio: np.ndarray


class LiveProcessor:
    def __init__(self, engine: STTEngine, censor_list: CensorList, config: LiveConfig) -> None:
        self.engine = engine
        self.config = config
        self.matcher = WordMatcher(censor_list)
        self._effects = create_effects(EffectOptions(sfx_tail=config.sfx_tail))

        self._ring: np.ndarray | None = None
        self._ring_lock = threading.Lock()
        self._write_idx = 0
        self._play_idx = 0

        self._in_stream: sd.InputStream | None = None
        self._out_stream: sd.OutputStream | None = None
        self._stt_thread: threading.Thread | None = None
        self._stop_evt = threading.Event()

        self._active: list[_ActiveEffect] = []
        self._active_lock = threading.Lock()

        self._last_stt_end_sample = 0
        self.on_word: Callable[[Word, bool], None] | None = None

    @property
    def _ring_size(self) -> int:
        sr = self.config.sample_rate
        return int((self.config.lookahead_seconds + self.config.chunk_seconds + 2.0) * sr)

    def _sample_to_time(self, sample_idx: int) -> float:
        return sample_idx / self.config.sample_rate

    def start(self) -> None:
        if self._in_stream is not None:
            return
        self._stop_evt.clear()

        self._ring = np.zeros(self._ring_size, dtype=np.float32)
        self._write_idx = 0
        self._play_idx = 0
        self._last_stt_end_sample = 0
        with self._active_lock:
            self._active.clear()

        sr = self.config.sample_rate
        bs = self.config.block_size

        self._in_stream = sd.InputStream(
            samplerate=sr,
            blocksize=bs,
            channels=1,
            dtype="float32",
            device=self.config.input_device,
            callback=self._on_input,
        )
        self._out_stream = sd.OutputStream(
            samplerate=sr,
            blocksize=bs,
            channels=1,
            dtype="float32",
            device=self.config.output_device,
            callback=self._on_output,
        )

        self._in_stream.start()
        self._out_stream.start()

        self._stt_thread = threading.Thread(target=self._stt_loop, daemon=True)
        self._stt_thread.start()

        log.info("Live processor started (lookahead=%.2fs, chunk=%.2fs)",
                 self.config.lookahead_seconds, self.config.chunk_seconds)

    def stop(self) -> None:
        self._stop_evt.set()
        for s in (self._in_stream, self._out_stream):
            if s is not None:
                try:
                    s.stop()
                    s.close()
                except Exception:
                    pass
        self._in_stream = None
        self._out_stream = None
        if self._stt_thread is not None:
            self._stt_thread.join(timeout=2.0)
            self._stt_thread = None
        log.info("Live processor stopped")

    def _ring_write(self, samples: np.ndarray) -> None:
        assert self._ring is not None
        n = samples.size
        size = self._ring.size
        with self._ring_lock:
            start = self._write_idx % size
            end = start + n
            if end <= size:
                self._ring[start:end] = samples
            else:
                first = size - start
                self._ring[start:] = samples[:first]
                self._ring[:n - first] = samples[first:]
            self._write_idx += n

    def _ring_read(self, sample_idx: int, n: int) -> np.ndarray:
        """Read ``n`` samples starting at absolute ``sample_idx``."""
        assert self._ring is not None
        size = self._ring.size
        out = np.zeros(n, dtype=np.float32)
        with self._ring_lock:
            # If requested region is outside available data, leave zeros
            if sample_idx + n > self._write_idx:
                n = max(0, self._write_idx - sample_idx)
                if n == 0:
                    return out
            start = sample_idx % size
            end = start + n
            if end <= size:
                out[:n] = self._ring[start:end]
            else:
                first = size - start
                out[:first] = self._ring[start:]
                out[first:n] = self._ring[:n - first]
        return out


    def _on_input(self, indata, frames, time_info, status):  # noqa: D401
        if status:
            log.debug("input status: %s", status)
        self._ring_write(indata[:, 0].copy() if indata.ndim > 1 else indata.copy())

    def _on_output(self, outdata, frames, time_info, status):
        if status:
            log.debug("output status: %s", status)

        sr = self.config.sample_rate
        lookahead_samples = int(self.config.lookahead_seconds * sr)

        desired = self._write_idx - lookahead_samples

        if desired < 0 or desired < self._play_idx:
            outdata.fill(0.0)
            return

        start = self._play_idx
        end = start + frames

        block = self._ring_read(start, frames)
        block = self._splice_censors(block, start_sample=start)

        if block.size < frames:
            block = np.pad(block, (0, frames - block.size))

        if outdata.ndim == 2:
            outdata[:, 0] = block
        else:
            outdata[:] = block
        self._play_idx = end


    def _splice_censors(self, block: np.ndarray, start_sample: int) -> np.ndarray:
        """Overlay every active effect on ``block`` and retire finished ones."""
        block_end = start_sample + block.size
        with self._active_lock:
            if not self._active:
                return block
            still_active: list[_ActiveEffect] = []
            for eff in self._active:
                eff_end = eff.start_sample + eff.audio.size
                if eff_end <= start_sample:
                    continue  # entirely in the past
                if eff.start_sample < block_end:
                    self._apply_effect(block, start_sample, eff)
                if eff_end > block_end:
                    still_active.append(eff)
            self._active = still_active
        return block

    @staticmethod
    def _apply_effect(block: np.ndarray, block_start: int, eff: _ActiveEffect) -> None:
        """Write one effect's overlap with ``block``: replace, then mix the tail."""
        lo = max(eff.start_sample, block_start)
        hi = min(eff.start_sample + eff.audio.size, block_start + block.size)
        if hi <= lo:
            return
        n = hi - lo
        s0 = lo - eff.start_sample
        d0 = lo - block_start
        src = eff.audio[s0:s0 + n]

        split = int(np.clip(eff.replace_samples - s0, 0, n))
        if split > 0:
            block[d0:d0 + split] = src[:split]
        if split < n:
            mixed = block[d0 + split:d0 + n] + src[split:]
            block[d0 + split:d0 + n] = np.clip(mixed, -1.0, 1.0)


    def _stt_loop(self) -> None:
        """Repeatedly transcribe the newest window and queue matched effects."""
        while not self._stop_evt.is_set():
            bounds = self._next_window()
            if bounds is None:
                time.sleep(0.02)
                continue
            start, end = bounds
            self._handle_words(self._transcribe(start, end))
            self._last_stt_end_sample = end

    def _next_window(self) -> tuple[int, int] | None:
        """Sample bounds of the next transcription window, or None if there is
        not yet a full chunk of new audio. The window starts slightly before
        the previously processed audio so words on a chunk boundary are not
        split and missed."""
        sr = self.config.sample_rate
        if self._write_idx - self._last_stt_end_sample < int(self.config.chunk_seconds * sr):
            return None
        end = self._write_idx
        start = max(self._last_stt_end_sample - int(_OVERLAP_SECONDS * sr), 0)
        if end - start < sr // 4:
            return None
        return start, end

    def _transcribe(self, start: int, end: int) -> list[Word]:
        """Transcribe ring samples ``[start, end)`` with absolute timestamps."""
        audio = self._ring_read(start, end - start)
        try:
            return self.engine.transcribe_chunk(
                audio, self.config.sample_rate,
                time_offset=self._sample_to_time(start))
        except Exception as e:
            log.warning("STT chunk failed: %s", e)
            return []

    def _handle_words(self, words: list[Word]) -> None:
        """Match recognised words and queue a rendered effect for each hit.

        A word that ends inside the already-processed audio was handled by
        the previous, overlapping window and is skipped, so the overlap does
        not produce duplicate detections.
        """
        sr = self.config.sample_rate
        pad = self.config.padding_ms * 1e-3
        for w in words:
            if w.end * sr < self._last_stt_end_sample:
                continue  # already handled by the previous window
            rule = self.matcher.match(w.text)
            self._notify_word(w, censored=rule is not None)
            if rule is None:
                continue
            effect = self._effects.get(rule.mode)
            if effect is None:
                continue

            start = max(int(round((w.start - pad) * sr)), 0)
            region = max(int(round((w.end + pad) * sr)) - start, 1)
            rendered = effect.render(rule, region, sr)
            audio = (np.concatenate([rendered.replacement, rendered.tail])
                     if rendered.tail.size else rendered.replacement)
            with self._active_lock:
                self._active.append(_ActiveEffect(
                    start_sample=start,
                    replace_samples=region,
                    audio=audio.astype(np.float32, copy=False),
                ))

    def _notify_word(self, word: Word, censored: bool) -> None:
        if self.on_word:
            try:
                self.on_word(word, censored)
            except Exception:
                pass
