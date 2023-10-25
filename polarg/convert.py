from __future__ import annotations

import itertools
import traceback
import typing as t
from pathlib import Path
from xml.etree import ElementTree as ET

import arguebuf
import typer
from ordered_set import OrderedSet

from polarg.model import Annotation, AnnotationDataset, EntailmentLabel

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
    annotations = OrderedSet([])
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
    annotations = OrderedSet([])
    file: Path

    with typer.progressbar(files, show_pos=True) as batches:
        for file in batches:
            try:
                graph = arguebuf.load.file(file)
            except Exception:
                typer.echo(f"Skipping graph '{file}' because an error occured:")
                typer.echo(traceback.print_exc())
                continue

            annotations = OrderedSet([])

            for scheme_node in graph.scheme_nodes.values():
                for premise, claim in itertools.product(
                    graph.incoming_nodes(scheme_node), graph.outgoing_nodes(scheme_node)
                ):
                    # rephrases are ignored
                    if (
                        (label := arguebuf_label.get(type(scheme_node.scheme)))
                        and isinstance(premise, arguebuf.AtomNode)
                        and isinstance(claim, arguebuf.AtomNode)
                    ):
                        annotations.add(
                            Annotation(premise.plain_text, claim.plain_text, label)
                        )

            for atom_node in graph.atom_nodes.values():
                for sibling in graph.sibling_nodes(atom_node):
                    if isinstance(sibling, arguebuf.AtomNode):
                        annotations.add(
                            Annotation(
                                atom_node.plain_text,
                                sibling.plain_text,
                                EntailmentLabel.NEUTRAL,
                            )
                        )

    return annotations


def _convert(
    input: Path,
    pattern: str,
    output: Path,
    convert_func: t.Callable[[t.Collection[Path]], OrderedSet[Annotation]],
):
    dataset = AnnotationDataset()
    typer.echo(f"Processing {input} with pattern {pattern}...")

    typer.echo("Converting training data")
    dataset.train = convert_func(list(input.glob("training/" + pattern)))

    typer.echo("Converting test data")
    dataset.test = convert_func(list(input.glob("test/" + pattern)))

    typer.echo("Converting validation data")
    dataset.validation = convert_func(list(input.glob("validation/" + pattern)))

    # with gzip.open(output.with_suffix(".json.gz"), "wt", encoding="utf-8") as f:
    #     json.dump(dataset.to_dict(), f)
    dataset.save(output)


@app.command()
def pheme(
    input: Path,
    pattern: str,
    output: Path,
):
    _convert(input, pattern, output, _pheme)


@app.command()
def argument_graph(
    input: Path,
    pattern: str,
    output: Path,
):
    _convert(input, pattern, output, _argument_graph)


arguebuf_patterns = {
    "kialo": "*.txt",
    "kialo-nilesc": "*.txt",
    "microtexts": "*.xml",
    "microtexts-v2": "*.xml",
    "persuasive-essays": "*.ann",
    "twitter-us2020": "*.json",
    "us-2016": "*.json",
}


@app.command()
def all_argument_graphs(input: Path, output: t.Optional[Path] = None):
    if output is None:
        output = input

    for name, pattern in arguebuf_patterns.items():
        _convert(input / name, pattern, output / f"{name}.pickle", _argument_graph)


if __name__ == "__main__":
    app()
