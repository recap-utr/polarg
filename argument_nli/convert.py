from __future__ import annotations

import itertools
import shutil
import traceback
import typing as t
from pathlib import Path
from xml.etree import ElementTree as ET

import arguebuf
import networkx as nx
import typer
from ordered_set import OrderedSet
from sklearn.model_selection import train_test_split

from argument_nli.config import config
from argument_nli.model import Annotation, AnnotationDataset, EntailmentLabel

app = typer.Typer()


pheme_label: dict[str, EntailmentLabel] = {
    "ENTAILMENT": EntailmentLabel.ENTAILMENT,
    "CONTRADICTION": EntailmentLabel.CONTRADICTION,
    "UNKNOWN": EntailmentLabel.NEUTRAL,
}

snli_label: dict[str, EntailmentLabel] = {
    "entailment": EntailmentLabel.ENTAILMENT,
    "contradiction": EntailmentLabel.CONTRADICTION,
    "neutral": EntailmentLabel.NEUTRAL,
}

arguebuf_label: dict[t.Type[arguebuf.Scheme | None], EntailmentLabel] = {
    arguebuf.Support: EntailmentLabel.ENTAILMENT,
    arguebuf.Attack: EntailmentLabel.CONTRADICTION,
    type(None): EntailmentLabel.NEUTRAL,
}


def _pheme(files: t.Collection[Path]) -> OrderedSet[Annotation]:
    annotations = OrderedSet()
    file: Path

    with typer.progressbar(files, show_pos=True) as batches:
        for file in batches:
            root = ET.parse(file)

            for pair in root.findall(".//pair"):
                premise = pair.findtext("t")
                claim = pair.findtext("h")
                label = pheme_label.get(pair.attrib["entailment"])

                if premise and claim and label:
                    annotations.add(Annotation(premise, claim, label))

    return annotations


def _argument_graph(files: t.Collection[Path]) -> OrderedSet[Annotation]:
    annotations = OrderedSet()
    file: Path

    with typer.progressbar(files, show_pos=True) as batches:
        for file in batches:
            try:
                graph = arguebuf.load.file(file)
            except Exception:
                typer.echo(f"Skipping graph '{file}' because an error occured:")
                typer.echo(traceback.print_exc())
                continue

            non_neutral_annotations = OrderedSet()

            for scheme_node in graph.scheme_nodes.values():
                for premise, claim in itertools.product(
                    graph.incoming_nodes(scheme_node), graph.outgoing_nodes(scheme_node)
                ):
                    if (
                        (label := arguebuf_label[type(scheme_node.scheme)])
                        and isinstance(premise, arguebuf.AtomNode)
                        and isinstance(claim, arguebuf.AtomNode)
                    ):
                        non_neutral_annotations.add(
                            Annotation(premise.plain_text, claim.plain_text, label)
                        )

            neutral_annotations = OrderedSet()

            if config.convert.include_neutral:
                # To speed up the computation for neutral samples, we use networkx
                nx_graph = arguebuf.dump.networkx(graph).to_undirected()
                dist = dict(
                    nx.all_pairs_shortest_path_length(
                        nx_graph, cutoff=config.convert.neutral_distance
                    )
                )
                atom_nodes = set(graph.atom_nodes.keys())

                # distance in graph > cutoff (see nx.all_pairs_shortest_path_length)
                for node1, node2 in itertools.product(nx_graph.nodes, nx_graph.nodes):
                    if (
                        node1 in atom_nodes
                        and node2 in atom_nodes
                        and (node2 not in dist[node1])
                    ):
                        neutral_annotations.add(
                            Annotation(
                                graph.atom_nodes[node1].plain_text,
                                graph.atom_nodes[node2].plain_text,
                                EntailmentLabel.NEUTRAL,
                            )
                        )

                # leaf nodes only need distance > 3
                # otherwise, small corpora like araucaria would have no neutral samples
                # if not neutral_annotations:
                #     for node1, node2 in itertools.product(nx_graph.nodes, nx_graph.nodes):
                #         if (
                #             node1 in atom_nodes
                #             and node2 in atom_nodes
                #             and (
                #                 len(graph.incoming_nodes(node1)) == 0
                #                 and len(graph.incoming_nodes(node2)) == 0
                #                 and dist[node1][node2] > 3
                #             )
                #         ):
                #             neutral_annotations.add(
                #                 Annotation(
                #                     graph.atom_nodes[node1].plain_text,
                #                     graph.atom_nodes[node2].plain_text,
                #                     EntailmentLabel.NEUTRAL,
                #                 )
                #             )

                # if len(neutral_annotations) > len(non_neutral_annotations):
                #     neutral_annotations = t.cast(
                #         t.List[Annotation],
                #         resample(
                #             neutral_annotations,
                #             replace=False,
                #             random_state=config.convert.random_state,
                #             n_samples=len(non_neutral_annotations),
                #         ),
                #     )

            annotations.update(non_neutral_annotations)
            annotations.update(neutral_annotations)

    return annotations


def _convert(
    input: Path,
    train_pattern: str,
    test_pattern: str,
    output: Path,
    convert_func: t.Callable[[t.Collection[Path]], OrderedSet[Annotation]],
):
    dataset = AnnotationDataset()

    typer.echo("Converting training data")
    dataset.train = convert_func(list(input.glob(train_pattern)))

    typer.echo("Converting testing data")
    dataset.test = convert_func(list(input.glob(test_pattern)))

    # with gzip.open(output.with_suffix(".json.gz"), "wt", encoding="utf-8") as f:
    #     json.dump(dataset.to_dict(), f)
    dataset.save(output)


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
