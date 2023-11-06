import typing as t
from pathlib import Path

import arguebuf
import grpc
import typer
from arg_services.mining.v1beta import entailment_pb2, entailment_pb2_grpc
from sklearn import metrics

from polarg.model.annotation import (
    contexttype_to_proto,
    graph_annotations,
    label_to_proto,
)

app = typer.Typer()


@app.command()
def main(address: str, path: Path, pattern: str, openai_model: t.Optional[str] = None):
    graphs = [arguebuf.load.file(path) for path in path.glob(pattern)]

    channel = grpc.insecure_channel(address)
    client = entailment_pb2_grpc.EntailmentServiceStub(channel)

    true_labels: list[entailment_pb2.EntailmentType.ValueType] = []
    req = entailment_pb2.EntailmentsRequest(language="en")

    if openai_model:
        req.extras["openai_model"] = openai_model

    for i, graph in enumerate(graphs):
        for node in graph.atom_nodes.values():
            req.adus[f"{i}-{node.id}"].text = node.plain_text

        for annotation in graph_annotations(graph):
            req.query.append(
                entailment_pb2.EntailmentQuery(
                    premise_id=f"{i}-{annotation.premise_id}",
                    claim_id=f"{i}-{annotation.claim_id}",
                    context=[
                        entailment_pb2.EntailmentContext(
                            adu_id=f"{i}-{item.adu_id}",
                            type=contexttype_to_proto[item.type],
                            weight=item.weight,
                        )
                        for item in annotation.context
                    ],
                )
            )
            assert annotation.label is not None
            true_labels.append(label_to_proto[annotation.label])

    res = client.Entailments(req)
    predicted_labels = [entailment.type for entailment in res.entailments]

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
