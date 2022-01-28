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
import networkx as nx
import pandas as pd
import typer
from dataclasses_json import DataClassJsonMixin
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

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


@dataclass(frozen=True, eq=True)
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


def _pheme(files: t.Collection[Path]) -> t.List[Annotation]:
    annotations = []
    file: Path

    with typer.progressbar(files, show_pos=True) as batches:
        for file in batches:
            root = ET.parse(file)

            for pair in root.findall(".//pair"):
                premise = pair.findtext("t")
                claim = pair.findtext("h")
                label = pheme_label.get(pair.attrib["entailment"])

                if premise and claim and label:
                    annotations.append(Annotation(premise, claim, label))

    return annotations


def _argument_graph(files: t.Collection[Path]) -> t.List[Annotation]:
    annotations = []
    file: Path

    with typer.progressbar(files, show_pos=True) as batches:
        for file in batches:
            graph = arguebuf.Graph.from_file(file)
            non_neutral_annotations = []

            for scheme in graph.scheme_nodes.values():
                for premise, claim in itertools.product(
                    graph.incoming_nodes(scheme), graph.outgoing_nodes(scheme)
                ):
                    if (
                        (label := arguebuf_label.get(scheme.type))
                        and isinstance(premise, arguebuf.AtomNode)
                        and isinstance(claim, arguebuf.AtomNode)
                    ):
                        non_neutral_annotations.append(
                            Annotation(premise.plain_text, claim.plain_text, label)
                        )

            # To speed up the computation for neutral samples, we use networkx
            nx_graph = graph.to_nx().to_undirected()
            dist = dict(nx.all_pairs_shortest_path_length(nx_graph, cutoff=15))
            atom_nodes = set(graph.atom_nodes.keys())

            neutral_annotations = []

            for node1, node2 in itertools.product(nx_graph.nodes, nx_graph.nodes):
                if (
                    node1 in atom_nodes
                    and node2 in atom_nodes
                    and (
                        # distance in graph > 9 due to specified cutoff
                        node2 not in dist[node1]
                        or (
                            # leaf nodes only need distance > 3
                            # otherwise, small corpora like araucaria
                            # would have no neutral samples
                            len(graph.incoming_nodes(node1)) == 0
                            and len(graph.incoming_nodes(node2)) == 0
                            and dist[node1][node2] > 3
                        )
                    )
                ):
                    neutral_annotations.append(
                        Annotation(
                            graph.atom_nodes[node1].plain_text,
                            graph.atom_nodes[node2].plain_text,
                            EntailmentLabel.NEUTRAL,
                        )
                    )

            if len(neutral_annotations) > len(non_neutral_annotations):
                neutral_annotations = t.cast(
                    t.List[Annotation],
                    resample(
                        neutral_annotations,
                        replace=False,
                        random_state=config.convert.random_state,
                        n_samples=len(non_neutral_annotations),
                    ),
                )

            annotations.extend(non_neutral_annotations)
            annotations.extend(neutral_annotations)

    return annotations


def _convert(
    input: Path,
    train_pattern: str,
    test_pattern: str,
    output: Path,
    convert_func: t.Callable[[t.Collection[Path]], t.List[Annotation]],
):
    dataset = AnnotationDataset()

    typer.echo("Converting training data")
    dataset.train.extend(convert_func(list(input.glob(train_pattern))))

    typer.echo("Converting testing data")
    dataset.test.extend(convert_func(list(input.glob(test_pattern))))

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
):
    train_output.mkdir(exist_ok=True)
    test_output.mkdir(exist_ok=True)

    files = list(input.glob(pattern))
    train, test = train_test_split(
        files,
        test_size=test_size,
        train_size=train_size,
        random_state=config.convert.random_state,
    )
    file: Path

    for file in train:
        shutil.copyfile(file, train_output / file.relative_to(input))

    for file in test:
        shutil.copyfile(file, test_output / file.relative_to(input))


if __name__ == "__main__":
    app()
