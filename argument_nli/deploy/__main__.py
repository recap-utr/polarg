import typing as t

import arg_services_helper
import grpc
import typer
from arg_services.entailment.v1 import entailment_pb2, entailment_pb2_grpc

from .model import EntailmentClassifier

model = EntailmentClassifier()
app = typer.Typer()


class EntailmentService(entailment_pb2_grpc.EntailmentServiceServicer):
    def Entailment(
        self,
        req: entailment_pb2.EntailmentRequest,
        ctx: grpc.ServicerContext,
    ) -> t.Optional[entailment_pb2.EntailmentResponse]:
        try:
            prediction, probabilities = model.predict(req.premise, req.claim)

            res = entailment_pb2.EntailmentResponse(prediction=prediction)

            res.details.extend(
                [
                    entailment_pb2.Detail(
                        probability=probability, prediction=prediction
                    )
                    for prediction, probability in probabilities.items()
                ]
            )

            return res

        except Exception as e:
            arg_services_helper.handle_except(e, ctx)

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
    arg_services_helper.serve(
        address,
        add_services,
        [arg_services_helper.full_service_name(entailment_pb2, "EntailmentService")],
    )


app()
