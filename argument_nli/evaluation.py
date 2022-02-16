import csv
import itertools
import typing as t
from pathlib import Path

import arguebuf
import grpc
import networkx as nx
import typer
from arg_services.entailment.v1 import entailment_pb2, entailment_pb2_grpc
from sklearn import metrics

from argument_nli.config import config

app = typer.Typer()


def get_prob(
    details: t.Iterable[entailment_pb2.Detail],
    entailment_type: int,
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


scheme2prediction = {
    arguebuf.SchemeType.SUPPORT: entailment_pb2.Prediction.PREDICTION_ENTAILMENT,
    arguebuf.SchemeType.ATTACK: entailment_pb2.Prediction.PREDICTION_CONTRADICTION,
    None: entailment_pb2.Prediction.PREDICTION_NEUTRAL,
}


@app.command()
def prediction(path: Path, pattern: str):
    graphs = [arguebuf.Graph.from_file(file) for file in path.glob(pattern)]

    channel = grpc.insecure_channel("127.0.0.1:6789")
    client = entailment_pb2_grpc.EntailmentServiceStub(channel)

    predicted_labels = []
    true_labels = []

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
                    entailment: entailment_pb2.EntailmentResponse = client.Entailment(
                        entailment_pb2.EntailmentRequest(
                            language="en",
                            premise=premise.plain_text,
                            claim=claim.plain_text,
                        )
                    )

                    predicted_labels.append(entailment.prediction)
                    true_labels.append(
                        scheme2prediction.get(
                            scheme.type, entailment_pb2.Prediction.PREDICTION_NEUTRAL
                        )
                    )

        if config.convert.include_neutral:
            nx_graph = graph.to_nx().to_undirected()
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
                    entailment: entailment_pb2.EntailmentResponse = client.Entailment(
                        entailment_pb2.EntailmentRequest(
                            language="en",
                            premise=graph.atom_nodes[node1].plain_text,
                            claim=graph.atom_nodes[node2].plain_text,
                        )
                    )

                    predicted_labels.append(entailment.prediction)
                    true_labels.append(entailment_pb2.Prediction.PREDICTION_NEUTRAL)

    labels = sorted(set(predicted_labels).union(true_labels))
    # labels = [
    #     entailment_pb2.Prediction.PREDICTION_ENTAILMENT,
    #     entailment_pb2.Prediction.PREDICTION_CONTRADICTION,
    # ]

    typer.echo(f"Labels: {[entailment_pb2.Prediction.Name(label) for label in labels]}")
    typer.echo(f"Accuracy: {metrics.accuracy_score(true_labels, predicted_labels)}")
    typer.echo(
        f"Recall: {metrics.recall_score(true_labels, predicted_labels, average=None, labels=labels)}"
    )
    typer.echo(
        f"Precision: {metrics.precision_score(true_labels, predicted_labels, average=None, labels=labels)}"
    )


if __name__ == "__main__":
    app()
