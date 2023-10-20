from __future__ import annotations

import pickle
import typing as t
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from arg_services.mining.v1beta.entailment_pb2 import EntailmentType
from ordered_set import OrderedSet


# The labels have to be in [0, ..., num_labels-1]
class EntailmentLabel(Enum):
    ENTAILMENT = 0
    CONTRADICTION = 1
    NEUTRAL = 2


label2proto: t.Dict[EntailmentLabel, EntailmentType.ValueType] = {
    EntailmentLabel.NEUTRAL: EntailmentType.ENTAILMENT_TYPE_NEUTRAL,
    EntailmentLabel.ENTAILMENT: EntailmentType.ENTAILMENT_TYPE_ENTAILMENT,
    EntailmentLabel.CONTRADICTION: EntailmentType.ENTAILMENT_TYPE_CONTRADICTION,
}


# frozen cannot be used when pickling this class
@dataclass(eq=True, unsafe_hash=True)
class Annotation:
    premise: str
    claim: str
    label: t.Optional[EntailmentLabel]


@dataclass(eq=True, unsafe_hash=True)
class AnnotationDataset:
    train: OrderedSet[Annotation] = field(default_factory=OrderedSet)
    test: OrderedSet[Annotation] = field(default_factory=OrderedSet)
    validation: OrderedSet[Annotation] = field(default_factory=OrderedSet)

    def update(self, other: AnnotationDataset) -> None:
        self.train.update(other.train)
        self.test.update(other.test)
        self.validation.update(other.validation)

    def remove_neutral(self) -> None:
        self.train = _remove_neutral(self.train)
        self.test = _remove_neutral(self.test)
        self.validation = _remove_neutral(self.validation)

    def save(self, file: Path) -> None:
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
