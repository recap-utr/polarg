import typing as t

import arg_services
import grpc
import typer
from arg_services.mining.v1beta import entailment_pb2, entailment_pb2_grpc

from .model import EntailmentClassifier

model = EntailmentClassifier()
app = typer.Typer()


class EntailmentService(entailment_pb2_grpc.EntailmentServiceServicer):
    def Entailment(
        self,
        req: entailment_pb2.EntailmentRequest,
        ctx: grpc.ServicerContext,
    ) -> t.Optional[entailment_pb2.EntailmentResponse]:
        prediction, probabilities = model.predict(req.premise, req.claim)

        res = entailment_pb2.EntailmentResponse(entailment_type=prediction)

        res.predictions.extend(
            [
                entailment_pb2.EntailmentPrediction(
                    probability=probability, type=prediction
                )
                for prediction, probability in probabilities.items()
            ]
        )

        return res

    def Entailments(
        self, request: entailment_pb2.EntailmentsRequest, context: grpc.ServicerContext
    ) -> entailment_pb2.EntailmentsResponse:
        raise NotImplementedError


def add_services(server: grpc.Server):
    entailment_pb2_grpc.add_EntailmentServiceServicer_to_server(
        EntailmentService(), server
    )


@app.command()
def main(address: str):
    arg_services.serve(
        address,
        add_services,
        [arg_services.full_service_name(entailment_pb2, "EntailmentService")],
    )


app()
