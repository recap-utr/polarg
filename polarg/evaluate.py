import sys
import typing as t
from pathlib import Path

import arguebuf
import grpc
import typer
from arg_services.mining.v1beta import entailment_pb2, entailment_pb2_grpc
from google.protobuf.struct_pb2 import Struct

from polarg import metrics
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
    export_file: t.Optional[Path] = None,
    llm_strategy: t.Optional[str] = None,
    llm_use_llama: bool = False,
    include_context: bool = False,
    include_neutral: bool = False,
    metrics_per_case: bool = False,
    start: int = 1,
    end: int = sys.maxsize,
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

    for i, graph in enumerate(graphs, 1):
        if i < start:
            continue
        if i > end:
            break

        print(f"Processing {graph.name} ({i}/{len(graphs)})")
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

        if metrics_per_case:
            metrics.show(metrics.serialize(predicted_labels, true_labels))
            print()

    print("Final metrics:")
    serialized_labels = metrics.serialize(predicted_labels, true_labels)
    metrics.show(serialized_labels)

    if export_file is not None:
        metrics.dump(serialized_labels, export_file)


if __name__ == "__main__":
    app()
