import typing as t
from pathlib import Path

import arguebuf
import grpc
import typer
from arg_services.mining.v1beta import entailment_pb2, entailment_pb2_grpc
from google.protobuf.struct_pb2 import Struct
from sklearn import metrics

from polarg.model import llm
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
    llm_strategy: t.Optional[str] = None,
    llm_use_llama: bool = False,
    include_context: bool = False,
    include_neutral: bool = False,
):
    graphs = [arguebuf.load.file(path) for path in sorted(path.glob(pattern))]

    channel_options = {
        "grpc.lb_policy_name": "round_robin",
        "grpc.max_send_message_length": -1,
        "grpc.max_receive_message_length": -1,
        "grpc.max_metadata_size": 1024 * 1024,
    }
    channel = grpc.insecure_channel(
        address,
        options=list(channel_options.items()),
    )
    client = entailment_pb2_grpc.EntailmentServiceStub(channel)

    predicted_labels: list[entailment_pb2.EntailmentType.ValueType] = []
    true_labels: list[entailment_pb2.EntailmentType.ValueType] = []

    with typer.progressbar(
        graphs, show_pos=True, item_show_func=lambda x: x.name if x else ""
    ) as batches:
        for graph in batches:
            req = entailment_pb2.EntailmentsRequest(language="en")

            if llm_strategy is not None:
                llm_options: llm.Options = {
                    "strategy": t.cast(llm.Strategy, llm_strategy),
                    "use_llama": llm_use_llama,
                    "include_neutral": include_neutral,
                }
                llm_struct = Struct()
                llm_struct.update(llm_options)  # type: ignore
                req.extras["llm_options"] = llm_struct

            for node in graph.atom_nodes.values():
                req.adus[node.id].text = node.plain_text

            for annotation in graph_annotations(graph):
                if annotation.label == EntailmentLabel.NEUTRAL and not include_neutral:
                    continue

                context: list[entailment_pb2.EntailmentContext] | None = None

                if include_context:
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

    len_all_labels = len(true_labels)

    true_labels = [
        label
        for idx, label in enumerate(true_labels)
        if predicted_labels[idx]
        != entailment_pb2.EntailmentType.ENTAILMENT_TYPE_UNSPECIFIED
    ]

    predicted_labels = [
        label
        for idx, label in enumerate(predicted_labels)
        if predicted_labels[idx]
        != entailment_pb2.EntailmentType.ENTAILMENT_TYPE_UNSPECIFIED
    ]

    len_known_labels = len(true_labels)
    len_unknown_labels = len_all_labels - len_known_labels

    labels = sorted(set(predicted_labels).union(true_labels))

    typer.echo(
        f"Labels: {[entailment_pb2.EntailmentType.Name(label) for label in labels]}"
    )
    typer.echo(
        f"Unknown labels: {len_unknown_labels} ({len_unknown_labels / len_all_labels:.2%})"
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
