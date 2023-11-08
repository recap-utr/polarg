import typing as t
from pathlib import Path

import arguebuf
import grpc
import typer
from arg_services.mining.v1beta import entailment_pb2, entailment_pb2_grpc
from sklearn import metrics

from polarg.config import config
from polarg.model.annotation import (
    EntailmentLabel,
    contexttype_to_proto,
    graph_annotations,
    label_to_proto,
)

app = typer.Typer()


@app.command()
def main(
    address: str,
    path: Path,
    pattern: str,
    openai_strategy: t.Optional[str] = None,
):
    graphs = [arguebuf.load.file(path) for path in sorted(path.glob(pattern))]

    channel = grpc.insecure_channel(address)
    client = entailment_pb2_grpc.EntailmentServiceStub(channel)

    predicted_labels: list[entailment_pb2.EntailmentType.ValueType] = []
    true_labels: list[entailment_pb2.EntailmentType.ValueType] = []

    with typer.progressbar(
        graphs, show_pos=True, item_show_func=lambda x: x.name if x else ""
    ) as batches:
        for graph in batches:
            req = entailment_pb2.EntailmentsRequest(language="en")

            if openai_strategy:
                req.extras["openai_strategy"] = openai_strategy

            for node in graph.atom_nodes.values():
                req.adus[node.id].text = node.plain_text

            for annotation in graph_annotations(graph):
                if (
                    annotation.label == EntailmentLabel.NEUTRAL
                    and not config.evaluate.include_neutral
                ):
                    continue

                context: list[entailment_pb2.EntailmentContext] | None = None

                if config.evaluate.include_context:
                    context = [
                        entailment_pb2.EntailmentContext(
                            adu_id=item.adu_id,
                            type=contexttype_to_proto[item.type],
                            weight=item.weight,
                        )
                        for item in annotation.context
                    ]

                req.query.append(
                    entailment_pb2.EntailmentQuery(
                        premise_id=annotation.premise_id,
                        claim_id=annotation.claim_id,
                        context=context,
                    )
                )
                assert annotation.label is not None
                true_labels.append(label_to_proto[annotation.label])

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
