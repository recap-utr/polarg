import asyncio
import itertools
import typing as t

import arg_services
import grpc
import typer
from arg_services.mining.v1beta import entailment_pb2, entailment_pb2_grpc
from lightning import Trainer
from torch.utils.data import DataLoader

from argument_nli.config import config
from argument_nli.model import Annotation, EntailmentDataset, EntailmentModule, openai

ProbsType = dict[entailment_pb2.EntailmentType.ValueType, float]

app = typer.Typer()


class EntailmentService(entailment_pb2_grpc.EntailmentServiceServicer):
    def __init__(self):
        typer.echo("Loading model...")
        try:
            self.module = EntailmentModule.load_from_checkpoint(config.model.path)
            self.trainer = Trainer()
        except FileNotFoundError:
            typer.echo("Model not found, only OpenAI will be available.")

    def Entailments(
        self, req: entailment_pb2.EntailmentsRequest, ctx: grpc.ServicerContext
    ) -> entailment_pb2.EntailmentsResponse:
        res = entailment_pb2.EntailmentsResponse()

        if len(req.query) == 0:
            req.query.extend(
                entailment_pb2.EntailmentQuery(premise_id=premise, claim_id=claim)
                for premise, claim in itertools.product(req.adus, req.adus)
                if premise != claim
            )

        annotations: list[Annotation] = []
        predictions: list[ProbsType] = []

        for query in req.query:
            premise = req.adus[query.premise_id].text
            claim = req.adus[query.claim_id].text
            annotations.append(Annotation(premise, claim, None))

        try:
            model_type = req.extras["model"]
        except ValueError:
            model_type = None

        if model_type == "openai":
            predictions = asyncio.run(openai.predict(annotations))

        else:
            dataloader = DataLoader(
                EntailmentDataset(annotations),
                batch_size=config.model.batch_size,
            )
            predictions = t.cast(
                list[ProbsType],
                self.trainer.predict(self.module, dataloaders=dataloader),
            )

        assert len(req.query) == len(predictions)

        for query, probabilities in zip(req.query, predictions):
            prediction: entailment_pb2.EntailmentType.ValueType = max(
                probabilities, key=probabilities.get  # type: ignore
            )

            res.entailments.append(
                entailment_pb2.Entailment(
                    type=prediction,
                    predictions=[
                        entailment_pb2.EntailmentPrediction(
                            probability=probability, type=prediction
                        )
                        for prediction, probability in probabilities.items()
                    ],
                    premise_id=query.premise_id,
                    claim_id=query.claim_id,
                )
            )

        return res


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