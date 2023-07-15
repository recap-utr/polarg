import json
import typing as t
from pathlib import Path

import backoff
import openai
from arg_services.mining.v1beta.entailment_pb2 import EntailmentType

openai.api_key_path = "./openai_api_key.txt"


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


async def predict(premise: str, claim: str) -> t.Dict[EntailmentType.ValueType, float]:
    with Path("./openai_schema.json").open("r") as fp:
        schema = json.load(fp)

    messages: list[ChatMessage] = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant that predicts the relation/entailment"
                " between the premise and the claim of an argument. You shall predict"
                " whether the premise supports or attacks the claim or has a neutral"
                " relation to it."
            ),
        },
        {
            "role": "user",
            "content": f"""
                Premise: {premise}.
                Claim: {claim}.
            """,
        },
    ]

    response = await _fetch_openai_chat(
        model="gpt-3.5-turbo-0613",
        messages=messages,
        functions=[{"name": "predict_entailment", "parameters": schema}],
        function_call={"name": "predict_entailment"},
    )

    results = response.choices[0].message.function_call.arguments

    return {type_map[result.entailment_type]: result.probability for result in results}
