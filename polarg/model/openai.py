import json
from pathlib import Path

from arg_services.mining.v1beta.entailment_pb2 import EntailmentType
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam as ChatMessage

from polarg.model.annotation import Annotation

client = AsyncOpenAI()


type_map = {
    "support": EntailmentType.ENTAILMENT_TYPE_ENTAILMENT,
    "attack": EntailmentType.ENTAILMENT_TYPE_CONTRADICTION,
    "neutral": EntailmentType.ENTAILMENT_TYPE_NEUTRAL,
}


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
                    " a neutral relation to it. Provide exactly three predictions, one"
                    " for each entailment type."
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

        response = await client.chat.completions.create(
            model=model,
            messages=messages,
            functions=[{"name": "predict_entailment", "parameters": schema}],
            function_call={"name": "predict_entailment"},
        )

        function_call = response.choices[0].message.function_call
        assert function_call is not None

        result = json.loads(function_call.arguments)
        probs: dict[EntailmentType.ValueType, float] = {
            type_map[result["entailment_type"]]: result["probability"]
            for result in result["entailments"]
        }
        predictions.append(probs)

    return predictions
