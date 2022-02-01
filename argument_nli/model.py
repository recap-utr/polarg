from __future__ import annotations

import pickle
import typing as t
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from ordered_set import OrderedSet


class EntailmentLabel(Enum):
    ENTAILMENT = "entailment"
    CONTRADICTION = "contradiction"
    NEUTRAL = "neutral"


# frozen cannot be used when pickling this class
@dataclass(eq=True, unsafe_hash=True)
class Annotation:
    premise: str
    claim: str
    label: EntailmentLabel


@dataclass(eq=True, unsafe_hash=True)
class AnnotationDataset:
    train: OrderedSet[Annotation] = field(default_factory=OrderedSet)
    test: OrderedSet[Annotation] = field(default_factory=OrderedSet)

    def update(self, other: AnnotationDataset) -> None:
        self.train.update(other.train)
        self.test.update(other.test)

    def save(self, file: Path) -> None:
        with file.with_suffix(".pickle").open("wb") as f:
            pickle.dump(self, f)

    @classmethod
    def open(cls, file: Path) -> AnnotationDataset:
        with Path(file).open("rb") as f:
            return pickle.load(f)
