from __future__ import annotations

import itertools
import traceback
import typing as t
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from xml.etree import ElementTree as ET

import arguebuf
import immutables
import typer
from arg_services.mining.v1beta import entailment_pb2
from mashumaro.mixins.orjson import DataClassORJSONMixin

from polarg.config import config


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
    # EntailmentLabel.NEUTRAL: entailment_pb2.EntailmentType.ENTAILMENT_TYPE_NEUTRAL,
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


@dataclass(frozen=True, slots=True)
class AnnotationContext(DataClassORJSONMixin):
    adu_id: str
    weight: float
    type: AnnotationContextType


# frozen cannot be used when pickling this class
@dataclass(frozen=True, slots=True)
class Annotation(DataClassORJSONMixin):
    premise_id: str
    claim_id: str
    context: tuple[AnnotationContext, ...]
    adus: immutables.Map[str, str]
    label: EntailmentLabel | None


@dataclass(frozen=True, slots=True)
class AnnotationDataset(DataClassORJSONMixin):
    train: tuple[Annotation, ...]
    test: tuple[Annotation, ...]
    validation: tuple[Annotation, ...]


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


def convert_pheme(files: t.Collection[Path]) -> list[Annotation]:
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
                    annotations.append(
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


def graph_annotations(graph: arguebuf.Graph) -> list[Annotation]:
    annotations = []

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
                annotations.append(
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

    if config.model.dataset.include_neutral:
        for premise in graph.atom_nodes.values():
            # scheme nodes are also counted as a level, so it needs to be a multiple of 2
            for claim in graph.sibling_nodes(premise, max_levels=2):
                if isinstance(claim, arguebuf.AtomNode):
                    context = graph_context(graph, premise, claim)
                    annotations.append(
                        Annotation(
                            premise.plain_text,
                            claim.plain_text,
                            context,
                            immutables.Map(
                                {
                                    premise.id: premise.plain_text,
                                    claim.id: claim.plain_text,
                                }
                                | {
                                    c.adu_id: graph.atom_nodes[c.adu_id].plain_text
                                    for c in context
                                }
                            ),
                            EntailmentLabel.NEUTRAL,
                        )
                    )

    return annotations


def convert_arguebuf(files: t.Collection[Path]) -> list[Annotation]:
    annotations = []
    file: Path

    with typer.progressbar(
        files, show_pos=True, item_show_func=lambda file: file.name if file else ""
    ) as batches:
        for file in batches:
            try:
                graph = arguebuf.load.file(file)
            except Exception:
                typer.echo(f"Skipping graph '{file}' because an error occured:")
                typer.echo(traceback.print_exc())
                continue

            annotations.extend(graph_annotations(graph))

    return annotations


def load_dataset(
    input: Path,
    pattern: str,
    convert_func: t.Callable[[t.Collection[Path]], list[Annotation]],
) -> tuple[list[Annotation], list[Annotation], list[Annotation]]:
    typer.echo(f"Processing {input} with pattern {pattern}...")

    typer.echo("Loading training data")
    train = convert_func(list(input.glob("training/" + pattern)))

    typer.echo("Loading test data")
    test = convert_func(list(input.glob("test/" + pattern)))

    typer.echo("Loading validation data")
    validation = convert_func(list(input.glob("validation/" + pattern)))

    return train, test, validation
