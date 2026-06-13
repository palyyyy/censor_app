from __future__ import annotations

from app.stt.base import Transcript, Word

from .censor_rules import normalize_token


def find_word_occurrences(transcript: Transcript, query: str) -> list[Word]:
    key = normalize_token(query)
    if not key:
        return []
    return [w for w in transcript.words if normalize_token(w.text) == key]


def format_timestamp(seconds: float) -> str:
    total = max(0, int(seconds))
    minutes, secs = divmod(total, 60)
    return f"{minutes}:{secs:02d}"
