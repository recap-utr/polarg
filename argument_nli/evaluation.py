import itertools
import typing as t
from pathlib import Path

import arguebuf
import grpc
import networkx as nx
import typer
from arg_services.mining.v1beta import entailment_pb2, entailment_pb2_grpc
from sklearn import metrics

from argument_nli.config import config

app = typer.Typer()


def get_prob(
    details: t.Iterable[entailment_pb2.EntailmentPrediction],
    entailment_type: int,
) -> float:
    return list(filter(lambda x: x.type == entailment_type, details))[0].probability


@app.command()
def adaptation(path: Path):
    retrieved_graphs: t.Dict[str, arguebuf.Graph] = {}
    adapted_graphs: t.Dict[str, arguebuf.Graph] = {}

    for retrieved_file in path.rglob("retrieved.json"):
        graph_name = retrieved_file.parent
        adapted_file = graph_name / "adapted.json"

        if retrieved_file.exists() and adapted_file.exists():
            retrieved_graphs[str(graph_name)] = arguebuf.load.file(retrieved_file)
            adapted_graphs[str(graph_name)] = arguebuf.load.file(adapted_file)

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
                        "retrieved_entailment": retrieved_entailment.entailment_type,
                        "adapted_entailment": adapted_entailment.entailment_type,
                        "entailment_prob": get_prob(
                            adapted_entailment.predictions,
                            entailment_pb2.ENTAILMENT_TYPE_ENTAILMENT,
                        ) - get_prob(
                            retrieved_entailment.predictions,
                            entailment_pb2.ENTAILMENT_TYPE_ENTAILMENT,
                        ),
                        "contradiction_prob": get_prob(
                            adapted_entailment.predictions,
                            entailment_pb2.ENTAILMENT_TYPE_CONTRADICTION,
                        ) - get_prob(
                            retrieved_entailment.predictions,
                            entailment_pb2.ENTAILMENT_TYPE_CONTRADICTION,
                        ),
                        "neutral_prob": get_prob(
                            adapted_entailment.predictions,
                            entailment_pb2.ENTAILMENT_TYPE_NEUTRAL,
                        ) - get_prob(
                            retrieved_entailment.predictions,
                            entailment_pb2.ENTAILMENT_TYPE_NEUTRAL,
                        ),
                    }

                    entailment_results.append(current_result)
                    total_pairs += 1

                    if (
                        retrieved_premise.plain_text != adapted_premise.plain_text
                        or retrieved_claim.plain_text != adapted_claim.plain_text
                    ):
                        adapted_pairs += 1

                    if (
                        retrieved_entailment.entailment_type
                        == adapted_entailment.entailment_type
                    ):
                        matching_pairs += 1

    print(f"Adapted pairs: {adapted_pairs}/{total_pairs} ({adapted_pairs/total_pairs})")
    print(
        "Matching predictions:"
        f" {matching_pairs}/{total_pairs} ({matching_pairs/total_pairs})"
    )


scheme2prediction: dict[
    t.Type[arguebuf.Scheme | None], entailment_pb2.EntailmentType.ValueType
] = {
    arguebuf.Support: entailment_pb2.EntailmentType.ENTAILMENT_TYPE_ENTAILMENT,
    arguebuf.Attack: entailment_pb2.EntailmentType.ENTAILMENT_TYPE_CONTRADICTION,
    type(None): entailment_pb2.EntailmentType.ENTAILMENT_TYPE_NEUTRAL,
}


@app.command()
def prediction(path: Path, pattern: str):
    graphs = [arguebuf.load.file(file) for file in path.glob(pattern)]

    channel = grpc.insecure_channel("127.0.0.1:6789")
    client = entailment_pb2_grpc.EntailmentServiceStub(channel)

    predicted_labels = []
    true_labels = []

    for graph in graphs:
        for scheme_node in graph.scheme_nodes.values():
            for claim, premise in itertools.product(
                graph.outgoing_nodes(scheme_node),
                graph.incoming_nodes(scheme_node),
            ):
                if (
                    isinstance(claim, arguebuf.AtomNode)
                    and isinstance(premise, arguebuf.AtomNode)
                    and isinstance(scheme_node, arguebuf.SchemeNode)
                    and type(scheme_node.scheme) in (arguebuf.Support, arguebuf.Attack)
                ):
                    entailment: entailment_pb2.EntailmentResponse = client.Entailment(
                        entailment_pb2.EntailmentRequest(
                            language="en",
                            premise=premise.plain_text,
                            claim=claim.plain_text,
                        )
                    )

                    predicted_labels.append(entailment.entailment_type)
                    true_labels.append(
                        scheme2prediction.get(
                            type(scheme_node.scheme),
                            entailment_pb2.EntailmentType.ENTAILMENT_TYPE_NEUTRAL,
                        )
                    )

        if config.convert.include_neutral:
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
                    entailment: entailment_pb2.EntailmentResponse = client.Entailment(
                        entailment_pb2.EntailmentRequest(
                            language="en",
                            premise=graph.atom_nodes[node1].plain_text,
                            claim=graph.atom_nodes[node2].plain_text,
                        )
                    )

                    predicted_labels.append(entailment.entailment_type)
                    true_labels.append(
                        entailment_pb2.EntailmentType.ENTAILMENT_TYPE_NEUTRAL
                    )

    labels = sorted(set(predicted_labels).union(true_labels))
    # labels = [
    #     entailment_pb2.Prediction.ENTAILMENT_TYPE_ENTAILMENT,
    #     entailment_pb2.Prediction.ENTAILMENT_TYPE_CONTRADICTION,
    # ]

    typer.echo(
        f"Labels: {[entailment_pb2.EntailmentType.Name(label) for label in labels]}"
    )
    typer.echo(f"Accuracy: {metrics.accuracy_score(true_labels, predicted_labels)}")
    typer.echo(
        "Recall:"
        f" {metrics.recall_score(true_labels, predicted_labels, average=None, labels=labels)}"
    )
    typer.echo(
        "Precision:"
        f" {metrics.precision_score(true_labels, predicted_labels, average=None, labels=labels)}"
    )


if __name__ == "__main__":
    app()
