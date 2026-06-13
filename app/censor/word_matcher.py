from __future__ import annotations

from .censor_rules import CensorList, CensorRule, normalize_token


class WordMatcher:
    def __init__(self, censor_list: CensorList) -> None:
        self._censor_list = censor_list
        self._index: dict[str, CensorRule] = {}
        self._rebuild_index()

    def refresh(self) -> None:
        self._rebuild_index()

    def _rebuild_index(self) -> None:
        self._index = {r.normalized_word: r for r in self._censor_list if r.normalized_word}

    def match(self, token: str) -> CensorRule | None:
        key = normalize_token(token)
        if not key:
            return None
        return self._index.get(key)
