import asyncio
import itertools
import typing as t

import arg_services
import grpc
import immutables
import typer
from arg_services.mining.v1beta import entailment_pb2, entailment_pb2_grpc
from lightning import Trainer
from torch.utils.data import DataLoader

from polarg.config import config
from polarg.model import EntailmentDataset, EntailmentModule, openai
from polarg.model.annotation import (
    Annotation,
    AnnotationContext,
    contexttype_from_proto,
)

ProbsType = dict[entailment_pb2.EntailmentType.ValueType, float]

app = typer.Typer()


class EntailmentService(entailment_pb2_grpc.EntailmentServiceServicer):
    def __init__(self):
        typer.echo("Loading model...")
        try:
            self.module = EntailmentModule.load_from_checkpoint(
                config.model.path, strict=False
            )
            self.trainer = Trainer(accelerator="gpu", logger=False)
        except FileNotFoundError:
            typer.echo("Model not found, only OpenAI will be available.")

    def Entailments(
        self, req: entailment_pb2.EntailmentsRequest, ctx: grpc.ServicerContext
    ) -> entailment_pb2.EntailmentsResponse:
        res = entailment_pb2.EntailmentsResponse()

        try:
            if len(req.query) == 0:
                req.query.extend(
                    entailment_pb2.EntailmentQuery(premise_id=premise, claim_id=claim)
                    for premise, claim in itertools.product(req.adus, req.adus)
                    if premise != claim
                )

            annotations: list[Annotation] = []
            predictions: list[ProbsType] = []

            for query in req.query:
                context = tuple(
                    AnnotationContext(
                        c.adu_id, c.weight, contexttype_from_proto[c.type]
                    )
                    for c in query.context
                )
                adus = immutables.Map(
                    {adu_id: adu.text for adu_id, adu in req.adus.items()}
                )
                annotations.append(
                    Annotation(
                        query.premise_id,
                        query.claim_id,
                        context,
                        adus,
                        None,
                    )
                )

            try:
                openai_strategy = t.cast(
                    openai.Strategy | None, req.extras["openai_strategy"]
                )
            except ValueError:
                openai_strategy = None

            if openai_strategy:
                predicted_values = asyncio.run(
                    openai.predict(annotations, openai_strategy)
                )
                predictions = [{val: 1.0} for val in predicted_values]

            else:
                dataloader = DataLoader(
                    EntailmentDataset(annotations),
                    batch_size=1,
                    num_workers=config.model.dataloader_workers,
                )
                predictions = t.cast(
                    list[ProbsType],
                    self.trainer.predict(self.module, dataloaders=[dataloader]),
                )

            assert predictions is not None
            assert len(req.query) == len(predictions)

            for query, probabilities in zip(req.query, predictions):
                prediction: entailment_pb2.EntailmentType.ValueType = max(
                    probabilities,
                    key=probabilities.get,  # type: ignore
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
        except Exception as ex:
            arg_services.handle_except(ex, ctx)

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
