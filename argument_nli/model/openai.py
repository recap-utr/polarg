import json
import typing as t
from pathlib import Path

import backoff
import openai
from arg_services.mining.v1beta.entailment_pb2 import EntailmentType

from argument_nli.model.annotation import Annotation


class ChatMessage(t.TypedDict):
    role: t.Literal["system", "user", "assistant"]
    content: str


type_map = {
    "support": EntailmentType.ENTAILMENT_TYPE_ENTAILMENT,
    "attack": EntailmentType.ENTAILMENT_TYPE_CONTRADICTION,
    "neutral": EntailmentType.ENTAILMENT_TYPE_NEUTRAL,
}


@backoff.on_exception(backoff.expo, openai.OpenAIError, max_tries=10)
async def _fetch_openai_chat(*args, **kwargs) -> t.Any:
    return await openai.ChatCompletion.acreate(*args, **kwargs)


async def predict(
    annotations: list[Annotation], model: str
) -> list[dict[EntailmentType.ValueType, float]]:
    predictions = []

    with Path("./openai_schema.json").open("r") as fp:
        schema = json.load(fp)

    for annotation in annotations:
        messages: list[ChatMessage] = [
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant that predicts the relation/entailment"
                    " between the premise and the claim of an argument. You shall"
                    " predict whether the premise supports or attacks the claim or has"
                    " a neutral relation to it."
                ),
            },
            {
                "role": "user",
                "content": f"""
                    Premise: {annotation.premise}.
                    Claim: {annotation.claim}.
                """,
            },
        ]

        response = await _fetch_openai_chat(
            model=model,
            messages=messages,
            functions=[{"name": "predict_entailment", "parameters": schema}],
            function_call={"name": "predict_entailment"},
        )

        result = json.loads(response.choices[0].message.function_call.arguments)
        probs: dict[EntailmentType.ValueType, float] = {
            type_map[result["entailment_type"]]: result["probability"]
            for result in result["entailments"]
        }
        predictions.append(probs)

    return predictions
