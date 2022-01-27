import csv
import itertools
import typing as t
from pathlib import Path

import arguebuf as ag
import grpc
from arg_services.entailment.v1 import entailment_pb2, entailment_pb2_grpc


def get_prob(
    details: t.Iterable[entailment_pb2.Detail],
    entailment_type: entailment_pb2.Prediction,
) -> float:
    return list(filter(lambda x: x.prediction == entailment_type, details))[
        0
    ].probability


input_path = Path(__file__).parent / Path("data/prediction/persuasive-essays")
graphs = ag.Graph.from_folder(input_path)

channel = grpc.insecure_channel("127.0.0.1:6789")
client = entailment_pb2_grpc.EntailmentServiceStub(channel)

total_pairs = 0
total_pairs_without_neutral = 0
matching_pairs = 0

for graph in graphs:
    for snode_id, snode in graph.scheme_nodes.items():
        for claim, premise in itertools.product(
            graph.outgoing_nodes(snode),
            graph.incoming_nodes(snode),
        ):
            if (
                isinstance(claim, ag.AtomNode)
                and isinstance(premise, ag.AtomNode)
                and isinstance(snode, ag.SchemeNode)
                and snode.type in (ag.SchemeType.SUPPORT, ag.SchemeType.ATTACK)
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

                # if graph.name.endswith("nodeset6463"):
                #     print(
                #         f"{premise.plain_text} -> {claim.plain_text}: {entailment.prediction}"
                #     )

                if (
                    entailment.prediction == entailment_pb2.PREDICTION_ENTAILMENT
                    and snode.type == ag.SchemeType.SUPPORT
                ) or (
                    entailment.prediction == entailment_pb2.PREDICTION_CONTRADICTION
                    and snode.type == ag.SchemeType.ATTACK
                ):
                    matching_pairs += 1

print(f"Matching pairs: {matching_pairs}/{total_pairs} ({matching_pairs/total_pairs})")
print(
    f"Matching pairs without neutral: {matching_pairs}/{total_pairs_without_neutral} ({matching_pairs/total_pairs_without_neutral})"
)
