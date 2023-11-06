from __future__ import annotations

import itertools
import traceback
import typing as t
from pathlib import Path
from xml.etree import ElementTree as ET

import arguebuf
import immutables
import typer
from ordered_set import OrderedSet

from polarg.model import Annotation, AnnotationDataset, EntailmentLabel
from polarg.model.annotation import AnnotationContext, AnnotationContextType

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

arguebuf_label: dict[type[arguebuf.Scheme | None], EntailmentLabel] = {
    arguebuf.Support: EntailmentLabel.ENTAILMENT,
    arguebuf.Attack: EntailmentLabel.CONTRADICTION,
    type(None): EntailmentLabel.NEUTRAL,
}


def convert_pheme(files: t.Collection[Path]) -> OrderedSet[Annotation]:
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
                    annotations.add(
                        Annotation(
                            "premise",
                            "claim",
                            tuple(),
                            immutables.Map(premise=premise, claim=claim),
                            label,
                        )
                    )

    return annotations


def _max_text_length(nodes: t.Collection[arguebuf.AtomNode]) -> int | None:
    if len(nodes) == 0:
        return None

    return max(map(lambda n: len(n.plain_text), nodes))


def graph_context(
    g: arguebuf.Graph, premise: arguebuf.AtomNode, claim: arguebuf.AtomNode
) -> tuple[AnnotationContext, ...]:
    context: list[AnnotationContext] = []

    parents = g.outgoing_atom_nodes(claim)
    longest_parent = _max_text_length(parents)

    children = g.incoming_atom_nodes(premise)
    longest_child = _max_text_length(children)

    premise_siblings = [
        n
        for n in g.sibling_nodes(premise, max_levels=2)
        if isinstance(n, arguebuf.AtomNode)
    ]
    longest_premise_sibling = _max_text_length(premise_siblings)

    claim_siblings = [
        n
        for n in g.sibling_nodes(claim, max_levels=2)
        if isinstance(n, arguebuf.AtomNode)
    ]
    longest_claim_sibling = _max_text_length(claim_siblings)

    context.extend(
        AnnotationContext(parent.id, 1, AnnotationContextType.PARENT)  # noqa: F821
        for parent in parents
        if len(parent.plain_text) == longest_parent
    )
    context.extend(
        AnnotationContext(child.id, 1, AnnotationContextType.CHILD)
        for child in children
        if len(child.plain_text) == longest_child
    )
    context.extend(
        AnnotationContext(sibling.id, 1, AnnotationContextType.SIBLING)
        for sibling in premise_siblings
        if len(sibling.plain_text) == longest_premise_sibling
    )
    context.extend(
        AnnotationContext(sibling.id, 1, AnnotationContextType.SIBLING)
        for sibling in claim_siblings
        if len(sibling.plain_text) == longest_claim_sibling
    )

    return tuple(context)


def graph_annotations(graph: arguebuf.Graph) -> OrderedSet[Annotation]:
    annotations = OrderedSet()

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
                context = graph_context(graph, premise, claim)
                annotations.add(
                    Annotation(
                        premise.id,
                        claim.id,
                        context,
                        immutables.Map(
                            {premise.id: premise.plain_text, claim.id: claim.plain_text}
                            | {
                                c.adu_id: graph.atom_nodes[c.adu_id].plain_text
                                for c in context
                            }
                        ),
                        label,
                    )
                )

    for premise in graph.atom_nodes.values():
        # scheme nodes are also counted as a level, so it needs to be a multiple of 2
        for claim in graph.sibling_nodes(premise, max_levels=2):
            if isinstance(claim, arguebuf.AtomNode):
                context = graph_context(graph, premise, claim)
                annotations.add(
                    Annotation(
                        premise.plain_text,
                        claim.plain_text,
                        context,
                        immutables.Map(
                            {premise.id: premise.plain_text, claim.id: claim.plain_text}
                            | {
                                c.adu_id: graph.atom_nodes[c.adu_id].plain_text
                                for c in context
                            }
                        ),
                        EntailmentLabel.NEUTRAL,
                    )
                )

    return annotations


def convert_arguebuf(files: t.Collection[Path]) -> OrderedSet[Annotation]:
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

            annotations.update(graph_annotations(graph))

    return annotations


def convert_dataset(
    input: Path,
    pattern: str,
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

    return dataset


@app.command()
def pheme(
    input: Path,
    pattern: str,
    output: Path,
):
    dataset = convert_dataset(input, pattern, convert_pheme)
    dataset.save(output)


@app.command()
def graph(
    input: Path,
    pattern: str,
    output: Path,
):
    dataset = convert_dataset(input, pattern, convert_arguebuf)
    dataset.save(output)


arguebuf_patterns = {
    "kialo": "*.txt",
    "kialo-nilesc": "*.txt",
    "kialo-graphnli": "*.json",
    "microtexts": "*.xml",
    "microtexts-v2": "*.xml",
    "persuasive-essays": "*.ann",
    "twitter-us2020": "*.json",
    "us-2016": "*.json",
}


@app.command()
def all_graphs(input: Path, output: t.Optional[Path] = None):
    if output is None:
        output = input

    for name, pattern in arguebuf_patterns.items():
        dataset = convert_dataset(input / name, pattern, convert_arguebuf)
        dataset.save(output / name)


if __name__ == "__main__":
    app()
