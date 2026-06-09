from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import sounddevice as sd

from app.audio.effects import generate_beep, generate_silence, load_sfx
from app.censor import CensorList, WordMatcher
from app.censor.censor_rules import CensorMode
from app.stt.base import STTEngine, Word
from app.utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class LiveConfig:
    sample_rate: int = 16000
    block_size: int = 1024
    chunk_seconds: float = 1.0
    lookahead_seconds: float = 2.0
    input_device: int | str | None = None
    output_device: int | str | None = None
    padding_ms: float = 50.0


@dataclass
class _PendingWord:
    word: Word
    rule_mode: CensorMode
    sfx_path: Optional[str]


class LiveProcessor:
    def __init__(self, engine: STTEngine, censor_list: CensorList, config: LiveConfig) -> None:
        self.engine = engine
        self.config = config
        self.matcher = WordMatcher(censor_list)

        self._ring: np.ndarray | None = None
        self._ring_lock = threading.Lock()
        self._write_idx = 0          
        self._play_idx = 0           

        self._in_stream: sd.InputStream | None = None
        self._out_stream: sd.OutputStream | None = None
        self._stt_thread: threading.Thread | None = None
        self._stop_evt = threading.Event()

        self._pending: list[_PendingWord] = []
        self._pending_lock = threading.Lock()

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

        size = self._ring_size
        self._ring = np.zeros(size, dtype=np.float32)
        self._write_idx = 0
        self._play_idx = 0
        self._last_stt_end_sample = 0
        with self._pending_lock:
            self._pending.clear()

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
        """Replace regions of ``block`` that correspond to pending censor words."""
        if not self._pending:
            return block

        sr = self.config.sample_rate
        block_start_s = start_sample / sr
        block_end_s = (start_sample + block.size) / sr
        pad = self.config.padding_ms * 1e-3

        still_pending: list[_PendingWord] = []
        with self._pending_lock:
            for pw in self._pending:
                w = pw.word
                w_start = max(0.0, w.start - pad)
                w_end = w.end + pad

                if w_end < block_start_s:
                    continue
                if w_start > block_end_s:
                    still_pending.append(pw)
                    continue

                # Overlap: splice.
                lo_s = max(w_start, block_start_s)
                hi_s = min(w_end, block_end_s)
                lo_i = int(round((lo_s - block_start_s) * sr))
                hi_i = int(round((hi_s - block_start_s) * sr))
                lo_i = max(0, lo_i)
                hi_i = min(block.size, hi_i)
                region_n = hi_i - lo_i
                if region_n <= 0:
                    still_pending.append(pw)
                    continue
                dur = region_n / sr

                if pw.rule_mode == CensorMode.BEEP:
                    rep = generate_beep(dur, sr)
                elif pw.rule_mode == CensorMode.SILENCE:
                    rep = generate_silence(dur, sr)
                elif pw.rule_mode == CensorMode.SFX and pw.sfx_path:
                    rep = load_sfx(pw.sfx_path, dur, sr, stretch=True)
                else:
                    rep = generate_beep(dur, sr)

                if rep.size < region_n:
                    rep = np.pad(rep, (0, region_n - rep.size))
                block[lo_i:hi_i] = rep[:region_n]

                if w_end > block_end_s:
                    still_pending.append(pw)

            self._pending = still_pending
        return block


    def _stt_loop(self) -> None:
        sr = self.config.sample_rate
        chunk_samples = int(self.config.chunk_seconds * sr)

        while not self._stop_evt.is_set():
            if self._write_idx - self._last_stt_end_sample < chunk_samples:
                time.sleep(0.02)
                continue

            window_end = self._write_idx
            # Give a bit of overlap for word boundaries
            overlap_samples = int(0.25 * sr)
            window_start = max(self._last_stt_end_sample - overlap_samples, 0)
            n = window_end - window_start
            if n < sr // 4:
                time.sleep(0.02)
                continue

            audio = self._ring_read(window_start, n)
            time_offset = window_start / sr

            try:
                words = self.engine.transcribe_chunk(audio, sr, time_offset=time_offset)
            except Exception as e:
                log.warning("STT chunk failed: %s", e)
                self._last_stt_end_sample = window_end
                continue

            for w in words:
                # Ignore words that are entirely inside the overlap region we've
                # already processed (best-effort dedup).
                if w.end * sr < self._last_stt_end_sample - overlap_samples:
                    continue
                rule = self.matcher.match(w.text)
                censored = rule is not None
                if self.on_word:
                    try:
                        self.on_word(w, censored)
                    except Exception:
                        pass
                if not censored:
                    continue
                with self._pending_lock:
                    self._pending.append(_PendingWord(
                        word=w,
                        rule_mode=rule.mode,
                        sfx_path=rule.sfx_path,
                    ))

            self._last_stt_end_sample = window_end
