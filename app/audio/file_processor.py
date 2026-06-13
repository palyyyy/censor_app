from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import soundfile as sf

import app.audio.censor_effects  # noqa: F401  (registers the concrete effects)
from app.censor import CensorList, WordMatcher, apply_censors
from app.censor.effects import EffectOptions, create_effects
from app.stt import STTEngine, Transcript
from app.utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class FileProcessResult:
    audio_out_path: Path
    transcript_path: Path
    transcript: Transcript
    censored_words: list


def _load_audio_mono(path: Path) -> tuple[np.ndarray, int]:
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
    else:
        if data.ndim > 1:
            data = data.mean(axis=1).astype(np.float32)
    return data.astype(np.float32), int(sr)


def _save_audio(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix in (".wav", ".flac", ".ogg"):
        sf.write(str(path), audio, sample_rate)
    elif suffix == ".mp3":
        from pydub import AudioSegment
        pcm16 = np.clip(audio, -1.0, 1.0)
        pcm16 = (pcm16 * 32767.0).astype(np.int16)
        seg = AudioSegment(
            pcm16.tobytes(),
            frame_rate=sample_rate,
            sample_width=2,
            channels=1,
        )
        seg.export(str(path), format="mp3", bitrate="192k")
    else:
        raise ValueError(f"Unsupported output format: {suffix}")


def process_file(
    input_path: str | Path,
    output_audio_path: str | Path,
    output_transcript_path: str | Path,
    engine: STTEngine,
    censor_list: CensorList,
    progress_cb: Callable[[str], None] | None = None,
    effect_options: EffectOptions | None = None,
    transcript: Transcript | None = None,
) -> FileProcessResult:
    """Censor ``input_path`` and write the audio and a transcript.

    ``effect_options`` configures the replacement effects (e.g. whether a
    sound effect may ring out past the censored word). A previously obtained
    ``transcript`` of the same file can be passed in to skip transcription.
    """
    input_path = Path(input_path)
    output_audio_path = Path(output_audio_path)
    output_transcript_path = Path(output_transcript_path)

    def progress(msg: str) -> None:
        log.info(msg)
        if progress_cb:
            progress_cb(msg)

    progress("Loading audio...")
    audio, sr = _load_audio_mono(input_path)
    dur = len(audio) / sr
    progress(f"Loaded {dur:.1f}s of audio at {sr} Hz")

    if transcript is None:
        progress("Transcribing... (this can take a while for large files)")
        transcript = engine.transcribe_file(input_path)
        progress(f"Transcribed {len(transcript.words)} words")
    else:
        progress(f"Using cached transcript ({len(transcript.words)} words)")

    matcher = WordMatcher(censor_list)
    effects = create_effects(effect_options)
    progress("Applying censors...")
    censored_audio, censored_words = apply_censors(
        audio, sr, transcript.words, matcher, effects)
    progress(f"Censored {len(censored_words)} word(s)")

    progress(f"Writing audio to {output_audio_path.name}")
    _save_audio(output_audio_path, censored_audio, sr)

    progress(f"Writing transcript to {output_transcript_path.name}")
    output_transcript_path.parent.mkdir(parents=True, exist_ok=True)
    output_transcript_path.write_text(_format_transcript(transcript, censored_words))

    progress("Done.")
    return FileProcessResult(
        audio_out_path=output_audio_path,
        transcript_path=output_transcript_path,
        transcript=transcript,
        censored_words=censored_words,
    )


def _format_transcript(transcript: Transcript, censored_words: list) -> str:
    censored_ids = {id(w) for w in censored_words}
    lines = ["# Transcript", ""]
    lines.append(transcript.text)
    lines.append("")
    lines.append("# Word timings")
    lines.append("#  start    end  [censored]  word")
    for w in transcript.words:
        marker = "  *" if id(w) in censored_ids else "   "
        lines.append(f"{w.start:7.2f}  {w.end:7.2f} {marker}  {w.text}")
    return "\n".join(lines) + "\n"
