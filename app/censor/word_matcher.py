from __future__ import annotations

from typing import Iterable

from .censor_rules import CensorList, CensorRule


_PUNCT = ".,!?;:'\"()[]{}"


def _normalize(token: str) -> str:
    return token.strip().lower().strip(_PUNCT)


class WordMatcher:

    def __init__(self, censor_list: CensorList) -> None:
        self._censor_list = censor_list
        self._index: dict[str, CensorRule] = {}
        self._rebuild_index()

    def refresh(self) -> None:
        """Call after editing the underlying CensorList."""
        self._rebuild_index()

    def _rebuild_index(self) -> None:
        self._index = {r.normalized_word: r for r in self._censor_list if r.normalized_word}

    def match(self, token: str) -> CensorRule | None:
        """Return the rule that matches ``token``, or None."""
        key = _normalize(token)
        if not key:
            return None
        return self._index.get(key)

    def any_match(self, tokens: Iterable[str]) -> bool:
        return any(self.match(t) for t in tokens)
