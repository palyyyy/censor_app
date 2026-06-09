from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Iterable, Iterator


class CensorMode(str, Enum):
    BEEP = "beep"
    SILENCE = "silence"
    SFX = "sfx"


@dataclass
class CensorRule:
    word: str
    mode: CensorMode = CensorMode.BEEP
    sfx_path: str | None = None    

    def __post_init__(self) -> None:
        self.word = self.word.strip()
            

    @property
    def normalized_word(self) -> str:
        return self.word.strip().lower().strip(".,!?;:'\"()[]{}")


@dataclass
class CensorList:
    rules: list[CensorRule] = field(default_factory=list)

    def __iter__(self) -> Iterator[CensorRule]:
        return iter(self.rules)

    def __len__(self) -> int:
        return len(self.rules)

    def add(self, rule: CensorRule) -> None:
        key = rule.normalized_word
        if not key:
            return
        for i, existing in enumerate(self.rules):
            if existing.normalized_word == key:
                self.rules[i] = rule
                return
        self.rules.append(rule)

    def remove(self, word: str) -> None:
        key = word.strip().lower()
        self.rules = [r for r in self.rules if r.normalized_word != key]

    def clear(self) -> None:
        self.rules.clear()

    def get(self, word: str) -> CensorRule | None:
        key = word.strip().lower().strip(".,!?;:'\"()[]{}")
        for r in self.rules:
            if r.normalized_word == key:
                return r
        return None

    def from_iterable(self, words: Iterable[str], mode: CensorMode = CensorMode.BEEP) -> None:
        for w in words:
            w = w.strip()
            if w:
                self.add(CensorRule(word=w, mode=mode))

    def words(self) -> list[str]:
        return [r.word for r in self.rules]
