import csv
import itertools
import typing as t
from pathlib import Path

import arguebuf
import grpc
import typer
from arg_services.entailment.v1 import entailment_pb2, entailment_pb2_grpc

app = typer.Typer()


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
