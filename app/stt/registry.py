from __future__ import annotations

from typing import Type

from .base import STTEngine

_REGISTRY: dict[str, Type[STTEngine]] = {}


def register_engine(cls: Type[STTEngine]) -> Type[STTEngine]:
    """Class decorator. Adds ``cls`` to the global registry under ``cls.name``."""
    name = getattr(cls, "name", None)
    if not name:
        raise ValueError(f"Engine {cls!r} must declare a non-empty .name")
    _REGISTRY[name] = cls
    return cls


def get_engine(name: str, model: str, language: str | None = "en") -> STTEngine:
    """Instantiate a registered engine by name."""
    if name not in _REGISTRY:
        raise KeyError(f"Unknown STT engine: {name!r}. Known: {list(_REGISTRY)}")
    cls = _REGISTRY[name]
    return cls(model=model, language=language)


def list_engines(only_available: bool = True) -> list[dict]:
    """Return engine metadata for the GUI.

    Each entry: {"name": str, "display_name": str, "available": bool,
                 "models": list[str]}.
    """
    out = []
    for name, cls in _REGISTRY.items():
        available = cls.is_available()
        if only_available and not available:
            continue
        out.append({
            "name": name,
            "display_name": cls.display_name,
            "available": available,
            "models": list(cls.available_models()),
        })
    return out
