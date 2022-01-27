from __future__ import annotations

import gzip
import itertools
import json
import shutil
import typing as t
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from xml.etree import ElementTree as ET

import arguebuf
import pandas as pd
import typer
from dataclasses_json import DataClassJsonMixin
from sklearn.model_selection import train_test_split

from argument_nli.config import config

app = typer.Typer()


class EntailmentLabel(str, Enum):
    ENTAILMENT = "entailment"
    CONTRADICTION = "contradiction"
    NEUTRAL = "neutral"


pheme_label = {
    "ENTAILMENT": EntailmentLabel.ENTAILMENT,
    "CONTRADICTION": EntailmentLabel.CONTRADICTION,
    "UNKNOWN": EntailmentLabel.NEUTRAL,
}

snli_label = {
    "entailment": EntailmentLabel.ENTAILMENT,
    "contradiction": EntailmentLabel.CONTRADICTION,
    "neutral": EntailmentLabel.NEUTRAL,
}

arguebuf_label = {
    arguebuf.SchemeType.SUPPORT: EntailmentLabel.ENTAILMENT,
    arguebuf.SchemeType.ATTACK: EntailmentLabel.CONTRADICTION,
}


@dataclass
class Annotation(DataClassJsonMixin):
    premise: str
    claim: str
    label: EntailmentLabel


@dataclass
class AnnotationDataset(DataClassJsonMixin):
    train: t.List[Annotation] = field(default_factory=list)
    test: t.List[Annotation] = field(default_factory=list)

    def extend(self, other: AnnotationDataset) -> None:
        self.train.extend(other.train)
        self.test.extend(other.test)


def _pheme(files: t.Iterable[Path]) -> t.List[Annotation]:
    annotations = []

    for file in files:
        root = ET.parse(file)

        for pair in root.findall(".//pair"):
            premise = pair.findtext("t")
            claim = pair.findtext("h")
            label = pheme_label.get(pair.attrib["entailment"])

            if premise and claim and label:
                annotations.append(Annotation(premise, claim, label))

    return annotations


def _argument_graph(files: t.Iterable[Path]) -> t.List[Annotation]:
    annotations = []

    for file in files:
        graph = arguebuf.Graph.from_file(file)

        for scheme in graph.scheme_nodes.values():
            for premise, claim in itertools.product(
                graph.incoming_atom_nodes(scheme), graph.outgoing_atom_nodes(scheme)
            ):
                if label := arguebuf_label.get(scheme.type):
                    annotations.append(
                        Annotation(premise.plain_text, claim.plain_text, label)
                    )

        for premise, claim in itertools.product(
            graph.atom_nodes.values(), graph.atom_nodes.values()
        ):
            if graph.node_distance(premise, claim, 9, directed=False) is None:
                annotations.append(
                    Annotation(
                        premise.plain_text, claim.plain_text, EntailmentLabel.NEUTRAL
                    )
                )

    return annotations


def _convert(
    input: Path,
    train_pattern: str,
    test_pattern: str,
    output: Path,
    convert_func: t.Callable[[t.Iterable[Path]], t.List[Annotation]],
):
    dataset = AnnotationDataset()

    dataset.train.extend(convert_func(input.glob(train_pattern)))
    dataset.test.extend(convert_func(input.glob(test_pattern)))

    with gzip.open(output.with_suffix(".json.gz"), "wt", encoding="utf-8") as f:
        json.dump(dataset.to_dict(), f)


@app.command()
def pheme(
    input: Path,
    train_pattern: str,
    test_pattern: str,
    output: Path,
):
    _convert(input, train_pattern, test_pattern, output, _pheme)


@app.command()
def argument_graph(input: Path, train_pattern: str, test_pattern: str, output: Path):
    _convert(input, train_pattern, test_pattern, output, _argument_graph)


@app.command()
def split(
    input: Path,
    pattern: str,
    train_output: Path,
    test_output: Path,
    test_size: t.Optional[float] = None,
    train_size: t.Optional[float] = None,
    random_state: int = 0,
):
    train_output.mkdir(exist_ok=True)
    test_output.mkdir(exist_ok=True)

    files = list(input.glob(pattern))
    train, test = train_test_split(
        files, test_size=test_size, train_size=train_size, random_state=random_state
    )
    file: Path

    for file in train:
        shutil.copyfile(file, train_output / file.relative_to(input))

    for file in test:
        shutil.copyfile(file, test_output / file.relative_to(input))


if __name__ == "__main__":
    app()
