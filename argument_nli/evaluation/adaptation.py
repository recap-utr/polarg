import csv
import itertools
import typing as t
from pathlib import Path
from pprint import pprint

import arguebuf
import grpc
from arg_services.entailment.v1 import entailment_pb2, entailment_pb2_grpc


def get_prob(
    details: t.Iterable[entailment_pb2.Detail],
    entailment_type: entailment_pb2.Prediction,
) -> float:
    return list(filter(lambda x: x.prediction == entailment_type, details))[
        0
    ].probability


input_path = Path(__file__).parent / Path("input/adaptation")
retrieved_graphs: t.Dict[str, arguebuf.Graph] = {}
adapted_graphs: t.Dict[str, arguebuf.Graph] = {}

for retrieved_file in input_path.rglob("retrieved.json"):
    graph_name = retrieved_file.parent
    adapted_file = graph_name / "adapted.json"

    if retrieved_file.exists() and adapted_file.exists():
        retrieved_graphs[str(graph_name)] = arguebuf.Graph.from_file(retrieved_file)
        adapted_graphs[str(graph_name)] = arguebuf.Graph.from_file(adapted_file)

# graph: arguebuf.Graph

# for graph in itertools.chain(retrieved_graphs.values(), adapted_graphs.values()):
#     graph.strip_snodes()

channel = grpc.insecure_channel("127.0.0.1:6789")
client = entailment_pb2_grpc.EntailmentServiceStub(channel)

entailment_results: t.List[t.Dict[str, t.Any]] = []
total_pairs = 0
adapted_pairs = 0
matching_pairs = 0

for graph_name, retrieved_graph in retrieved_graphs.items():
    adapted_graph = adapted_graphs[graph_name]

    for snode_id, retrieved_snode in retrieved_graph.snode_mappings.items():
        for retrieved_claim, retrieved_premise in itertools.product(
            retrieved_graph.outgoing_nodes[retrieved_snode],
            retrieved_graph.incoming_nodes[retrieved_snode],
        ):
            if (
                retrieved_claim.category == arguebuf.NodeCategory.I
                and retrieved_premise.category == arguebuf.NodeCategory.I
                # True
            ):
                adapted_claim = adapted_graph.node_mappings[retrieved_claim.key]
                adapted_premise = adapted_graph.node_mappings[retrieved_premise.key]

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
                        adapted_entailment.details, entailment_pb2.PREDICTION_ENTAILMENT
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
                        adapted_entailment.details, entailment_pb2.PREDICTION_NEUTRAL
                    )
                    - get_prob(
                        retrieved_entailment.details, entailment_pb2.PREDICTION_NEUTRAL
                    ),
                }

                # pprint(current_result)
                entailment_results.append(current_result)
                total_pairs += 1

                # if graph_name.endswith(
                #     "microtexts-lorik/public_broadcasting_fees_on_demand/nodeset6463"
                # ):
                #     print(
                #         f"{retrieved_premise.plain_text} -> {retrieved_claim.plain_text}"
                #     )

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

# with Path("evaluation", "output.csv").open("w") as f:
#     writer = csv.DictWriter(
#         f,
#         [
#             "graph_name",
#             "source",
#             "target",
#             "retrieved_entailment",
#             "adapted_entailment",
#             "entailment_prob",
#             "contradiction_prob",
#             "neutral_prob",
#         ],
#     )
#     writer.writeheader()
#     writer.writerows(entailment_results)
