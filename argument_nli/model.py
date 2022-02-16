from __future__ import annotations

import pickle
import typing as t
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from ordered_set import OrderedSet

from argument_nli.config import config


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
        if not config.convert.include_neutral:
            self.train = _remove_neutral(self.train)
            self.test = _remove_neutral(self.test)

        with file.with_suffix(".pickle").open("wb") as f:
            pickle.dump(self, f)

    @classmethod
    def open(cls, file: Path) -> AnnotationDataset:
        with Path(file).open("rb") as f:
            return pickle.load(f)


def _remove_neutral(annotations: t.Iterable[Annotation]) -> OrderedSet[Annotation]:
    return OrderedSet(
        annotation
        for annotation in annotations
        if annotation.label != EntailmentLabel.NEUTRAL
    )
