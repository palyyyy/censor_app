from __future__ import annotations

from pathlib import Path

import sounddevice as sd

import config
from app.utils.logger import get_logger

from .effects import load_sfx_clip

log = get_logger(__name__)

SFX_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg"}

_PREVIEW_SAMPLE_RATE = 44_100


def list_stock_sfx(folder: Path | None = None) -> list[Path]:
    """Audio files in the stock SFX folder, sorted by name.

    Returns an empty list if the folder does not exist, so a missing or
    empty library is never an error.
    """
    folder = folder or config.STOCK_SFX_DIR
    if not folder.exists() or not folder.is_dir():
        return []
    return sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in SFX_EXTENSIONS
    )


def preview_sfx(path: str | Path) -> None:
    """Play ``path`` through the default output device.

    Any clip already previewing is stopped first, so pressing one preview
    button after another never overlaps two sounds.
    """
    clip = load_sfx_clip(path, _PREVIEW_SAMPLE_RATE)
    if clip.size == 0:
        return
    sd.stop()
    sd.play(clip, _PREVIEW_SAMPLE_RATE)


def stop_preview() -> None:
    """Stop any preview currently playing."""
    sd.stop()
