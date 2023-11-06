from __future__ import annotations

import typing as t
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import immutables
from arg_services.mining.v1beta import entailment_pb2
from mashumaro.mixins.orjson import DataClassORJSONMixin
from ordered_set import OrderedSet


# The labels have to be in [0, ..., num_labels-1]
class EntailmentLabel(Enum):
    ENTAILMENT = 0
    CONTRADICTION = 1
    NEUTRAL = 2


class AnnotationContextType(Enum):
    PARENT = 0
    CHILD = 1
    SIBLING = 2


label_to_proto: dict[EntailmentLabel, entailment_pb2.EntailmentType.ValueType] = {
    EntailmentLabel.NEUTRAL: entailment_pb2.EntailmentType.ENTAILMENT_TYPE_NEUTRAL,
    EntailmentLabel.ENTAILMENT: entailment_pb2.EntailmentType.ENTAILMENT_TYPE_ENTAILMENT,
    EntailmentLabel.CONTRADICTION: entailment_pb2.EntailmentType.ENTAILMENT_TYPE_CONTRADICTION,
}

contexttype_to_proto: dict[
    AnnotationContextType, entailment_pb2.EntailmentContextType.ValueType
] = {
    AnnotationContextType.PARENT: entailment_pb2.EntailmentContextType.ENTAILMENT_CONTEXT_TYPE_PARENT,
    AnnotationContextType.CHILD: entailment_pb2.EntailmentContextType.ENTAILMENT_CONTEXT_TYPE_CHILD,
    AnnotationContextType.SIBLING: entailment_pb2.EntailmentContextType.ENTAILMENT_CONTEXT_TYPE_SIBLING,
}

contexttype_from_proto = {value: key for key, value in contexttype_to_proto.items()}


@dataclass(frozen=True)
class AnnotationContext(DataClassORJSONMixin):
    adu_id: str
    weight: float
    type: AnnotationContextType


# frozen cannot be used when pickling this class
@dataclass(frozen=True)
class Annotation(DataClassORJSONMixin):
    premise_id: str
    claim_id: str
    context: tuple[AnnotationContext, ...]
    adus: immutables.Map[str, str]
    label: EntailmentLabel | None


@dataclass(eq=True)
class AnnotationDataset(DataClassORJSONMixin):
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
        with file.with_suffix(".json").open("wb") as f:
            f.write(self.to_jsonb())

    @classmethod
    def open(cls, file: Path) -> AnnotationDataset:
        with Path(file).open("rb") as f:
            return cls.from_json(f.read())


def _remove_neutral(annotations: t.Iterable[Annotation]) -> OrderedSet[Annotation]:
    return OrderedSet(
        annotation
        for annotation in annotations
        if annotation.label != EntailmentLabel.NEUTRAL
    )
