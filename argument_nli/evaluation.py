import csv
import itertools
import typing as t
from pathlib import Path

import arguebuf
import grpc
import typer
from arg_services.entailment.v1 import entailment_pb2, entailment_pb2_grpc

app = typer.Typer()


def get_prob(
    details: t.Iterable[entailment_pb2.Detail],
    entailment_type: entailment_pb2.Prediction.V,
) -> float:
    return list(filter(lambda x: x.prediction == entailment_type, details))[
        0
    ].probability


@app.command()
def adaptation(path: Path):
    retrieved_graphs: t.Dict[str, arguebuf.Graph] = {}
    adapted_graphs: t.Dict[str, arguebuf.Graph] = {}

    for retrieved_file in path.rglob("retrieved.json"):
        graph_name = retrieved_file.parent
        adapted_file = graph_name / "adapted.json"

        if retrieved_file.exists() and adapted_file.exists():
            retrieved_graphs[str(graph_name)] = arguebuf.Graph.from_file(retrieved_file)
            adapted_graphs[str(graph_name)] = arguebuf.Graph.from_file(adapted_file)

    channel = grpc.insecure_channel("127.0.0.1:6789")
    client = entailment_pb2_grpc.EntailmentServiceStub(channel)

    entailment_results: t.List[t.Dict[str, t.Any]] = []
    total_pairs = 0
    adapted_pairs = 0
    matching_pairs = 0

    for graph_name, retrieved_graph in retrieved_graphs.items():
        adapted_graph = adapted_graphs[graph_name]

        for retrieved_scheme in retrieved_graph.scheme_nodes.values():
            for retrieved_claim, retrieved_premise in itertools.product(
                retrieved_graph.outgoing_nodes(retrieved_scheme),
                retrieved_graph.incoming_nodes(retrieved_scheme),
            ):
                if isinstance(retrieved_claim, arguebuf.AtomNode) and isinstance(
                    retrieved_premise, arguebuf.AtomNode
                ):
                    adapted_claim = adapted_graph.atom_nodes[retrieved_claim.id]
                    adapted_premise = adapted_graph.atom_nodes[retrieved_premise.id]

                    retrieved_entailment = client.Entailment(
                        entailment_pb2.EntailmentRequest(
                            language="en",
                            premise=retrieved_premise.plain_text,
                            claim=retrieved_claim.plain_text,
                        )
                    )

                    adapted_entailment = client.Entailment(
                        entailment_pb2.EntailmentRequest(
                            language="en",
                            premise=adapted_premise.plain_text,
                            claim=adapted_claim.plain_text,
                        )
                    )

                    current_result = {
                        "graph_name": graph_name,
                        "source": retrieved_premise.plain_text,
                        "target": retrieved_claim.plain_text,
                        "retrieved_entailment": retrieved_entailment.prediction,
                        "adapted_entailment": adapted_entailment.prediction,
                        "entailment_prob": get_prob(
                            adapted_entailment.details,
                            entailment_pb2.PREDICTION_ENTAILMENT,
                        )
                        - get_prob(
                            retrieved_entailment.details,
                            entailment_pb2.PREDICTION_ENTAILMENT,
                        ),
                        "contradiction_prob": get_prob(
                            adapted_entailment.details,
                            entailment_pb2.PREDICTION_CONTRADICTION,
                        )
                        - get_prob(
                            retrieved_entailment.details,
                            entailment_pb2.PREDICTION_CONTRADICTION,
                        ),
                        "neutral_prob": get_prob(
                            adapted_entailment.details,
                            entailment_pb2.PREDICTION_NEUTRAL,
                        )
                        - get_prob(
                            retrieved_entailment.details,
                            entailment_pb2.PREDICTION_NEUTRAL,
                        ),
                    }

                    entailment_results.append(current_result)
                    total_pairs += 1

                    if (
                        retrieved_premise.plain_text != adapted_premise.plain_text
                        or retrieved_claim.plain_text != adapted_claim.plain_text
                    ):
                        adapted_pairs += 1

                    if retrieved_entailment.prediction == adapted_entailment.prediction:
                        matching_pairs += 1

    print(f"Adapted pairs: {adapted_pairs}/{total_pairs} ({adapted_pairs/total_pairs})")
    print(
        f"Matching predictions: {matching_pairs}/{total_pairs} ({matching_pairs/total_pairs})"
    )


@app.command()
def prediction(path: Path, pattern: str):
    graphs = [arguebuf.Graph.from_file(file) for file in path.glob(pattern)]

    channel = grpc.insecure_channel("127.0.0.1:6789")
    client = entailment_pb2_grpc.EntailmentServiceStub(channel)

    total_pairs = 0
    total_pairs_without_neutral = 0
    matching_pairs = 0

    for graph in graphs:
        for scheme in graph.scheme_nodes.values():
            for claim, premise in itertools.product(
                graph.outgoing_nodes(scheme),
                graph.incoming_nodes(scheme),
            ):
                if (
                    isinstance(claim, arguebuf.AtomNode)
                    and isinstance(premise, arguebuf.AtomNode)
                    and isinstance(scheme, arguebuf.SchemeNode)
                    and scheme.type
                    in (arguebuf.SchemeType.SUPPORT, arguebuf.SchemeType.ATTACK)
                ):
                    entailment = client.Entailment(
                        entailment_pb2.EntailmentRequest(
                            language="en",
                            premise=premise.plain_text,
                            claim=claim.plain_text,
                        )
                    )

                    total_pairs += 1

                    if entailment.prediction != entailment_pb2.PREDICTION_NEUTRAL:
                        total_pairs_without_neutral += 1

                    if (
                        entailment.prediction == entailment_pb2.PREDICTION_ENTAILMENT
                        and scheme.type == arguebuf.SchemeType.SUPPORT
                    ) or (
                        entailment.prediction == entailment_pb2.PREDICTION_CONTRADICTION
                        and scheme.type == arguebuf.SchemeType.ATTACK
                    ):
                        matching_pairs += 1

    typer.echo(
        f"Matching pairs: {matching_pairs}/{total_pairs} ({matching_pairs/total_pairs})"
    )
    typer.echo(
        f"Matching pairs without neutral: {matching_pairs}/{total_pairs_without_neutral} ({matching_pairs/total_pairs_without_neutral})"
    )


if __name__ == "__main__":
    app()
