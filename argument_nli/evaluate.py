import itertools
import typing as t
from pathlib import Path

import arguebuf
import grpc
import networkx as nx
import typer
from arg_services.mining.v1beta import adu_pb2, entailment_pb2, entailment_pb2_grpc
from sklearn import metrics

from argument_nli.config import config

app = typer.Typer()


scheme2prediction: dict[
    t.Type[arguebuf.Scheme | None], entailment_pb2.EntailmentType.ValueType
] = {
    arguebuf.Support: entailment_pb2.EntailmentType.ENTAILMENT_TYPE_ENTAILMENT,
    arguebuf.Attack: entailment_pb2.EntailmentType.ENTAILMENT_TYPE_CONTRADICTION,
    type(None): entailment_pb2.EntailmentType.ENTAILMENT_TYPE_NEUTRAL,
}


@app.command()
def main(address: str, path: Path, pattern: str):
    graphs = [arguebuf.load.file(file) for file in path.glob(pattern)]

    channel = grpc.insecure_channel(address)
    client = entailment_pb2_grpc.EntailmentServiceStub(channel)

    predicted_labels = []
    true_labels = []

    for graph in graphs:
        req = entailment_pb2.EntailmentsRequest(
            language="en",
            adus={
                node.id: adu_pb2.Segment(text=node.plain_text)
                for node in graph.atom_nodes.values()
            },
        )
        req.extras["model"] = "openai"

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
                    req.query.append(
                        entailment_pb2.EntailmentQuery(
                            premise_id=premise.id, claim_id=claim.id
                        )
                    )
                    true_labels.append(
                        scheme2prediction.get(
                            type(scheme_node.scheme),
                            entailment_pb2.EntailmentType.ENTAILMENT_TYPE_NEUTRAL,
                        )
                    )

        if config.model.include_neutral:
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
                    req.query.append(
                        entailment_pb2.EntailmentQuery(
                            premise_id=graph.atom_nodes[node1].id,
                            claim_id=graph.atom_nodes[node2].id,
                        )
                    )
                    true_labels.append(
                        entailment_pb2.EntailmentType.ENTAILMENT_TYPE_NEUTRAL
                    )

        res = client.Entailments(req)
        predicted_labels.extend(entailment.type for entailment in res.entailments)

    labels = sorted(set(predicted_labels).union(true_labels))

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
