import asyncio
import json
from collections import abc
from copy import deepcopy
from pathlib import Path
from typing import Any, Literal, TypedDict

import httpx
from arg_services.mining.v1beta.entailment_pb2 import EntailmentType
from openai import AsyncOpenAI, OpenAIError
from openai.types.chat import ChatCompletionMessageParam as ChatMessage

from polarg.model.annotation import Annotation

client_openai = AsyncOpenAI(
    http_client=httpx.AsyncClient(
        http2=True,
        timeout=httpx.Timeout(timeout=30, connect=5),
        limits=httpx.Limits(max_connections=100, max_keepalive_connections=20),
    ),
    max_retries=3,
)
client_llama = AsyncOpenAI(
    http_client=httpx.AsyncClient(
        timeout=httpx.Timeout(timeout=120, connect=5),
        limits=httpx.Limits(max_connections=100, max_keepalive_connections=20),
    ),
    max_retries=3,
)


polarity_map = {
    "support": EntailmentType.ENTAILMENT_TYPE_ENTAILMENT,
    "attack": EntailmentType.ENTAILMENT_TYPE_CONTRADICTION,
    "neutral": EntailmentType.ENTAILMENT_TYPE_NEUTRAL,
    None: EntailmentType.ENTAILMENT_TYPE_UNSPECIFIED,
}

ModelName = Literal["gpt-3.5-turbo-1106", "gpt-4-1106-preview", "gpt-4-0613"]

token_limits: dict[ModelName, int] = {
    "gpt-3.5-turbo-1106": 16385,
    "gpt-4-0613": 8192,
    "gpt-4-1106-preview": 128000,
}

with Path("./openai_schema.json").open("r") as fp:
    schema_with_neutral: dict[str, Any] = json.load(fp)

schema_without_neutral = deepcopy(schema_with_neutral)
schema_without_neutral["properties"]["polarities"]["items"]["properties"][
    "polarity_type"
]["enum"].remove("neutral")


def generate_schema(include_neutral: bool) -> dict[str, Any]:
    if include_neutral:
        return schema_with_neutral
    else:
        return schema_without_neutral


BASE_SYSTEM_MESSAGE = """
You are a helpful assistant that predicts the relation/polarity between the premise and the claim of an argument.
You shall predict whether the premise supports or attacks the claim.
Answer either "support" or "attack".
"""

BASE_SYSTEM_MESSAGE_NEUTRAL_APPENDIX = """
The premise may also have a neutral relation to the claim.
In this case, you may answer "neutral".
"""


def generate_system_message(include_neutral: bool) -> str:
    if include_neutral:
        return BASE_SYSTEM_MESSAGE + BASE_SYSTEM_MESSAGE_NEUTRAL_APPENDIX
    else:
        return BASE_SYSTEM_MESSAGE


# sequential: send prompts one-by-one and append previous messages to the prompt to simulate memory. Does not use context.
# isolated: send prompts one-by-one without appending previous messages to the prompt. Uses context if `include_context` is true.
# batched: send all prompts at once and append previous messages to the prompt to simulate memory. Does not use context.
Strategy = Literal["isolated", "sequential", "batched"]


class Options(TypedDict):
    strategy: Strategy
    use_llama: bool
    include_neutral: bool


async def predict(
    annotations: abc.Sequence[Annotation], options: Options
) -> list[EntailmentType.ValueType]:
    predictions: list[EntailmentType.ValueType] | None = None

    if options["strategy"] == "isolated":
        predictions = await asyncio.gather(
            *(predict_isolated(annotation, options) for annotation in annotations)
        )
    elif options["strategy"] == "sequential":
        predictions = await predict_sequential(annotations, options)
    elif options["strategy"] == "batched":
        predictions = await predict_batched(annotations, options)

    assert predictions is not None

    return predictions


async def chat_response(
    messages: list[ChatMessage], use_llama: bool, openai_model: ModelName
) -> str:
    if use_llama:
        response = await client_llama.chat.completions.create(
            model="llama2",
            messages=messages,
        )
    else:
        response = await client_openai.chat.completions.create(
            model=openai_model,
            messages=messages,
        )

    response_msg = response.choices[0].message.content
    assert response_msg is not None
    return response_msg


async def predict_isolated(
    annotation: Annotation, options: Options, model: ModelName = "gpt-3.5-turbo-1106"
) -> EntailmentType.ValueType:
    premise = annotation.adus[annotation.premise_id]
    claim = annotation.adus[annotation.claim_id]

    user_msg = f"""
Premise: {premise}.
Claim: {claim}.
    """

    if len(annotation.context) > 0:
        annotation_context = "\n".join(
            annotation.adus[ctx.adu_id] for ctx in annotation.context
        )
        user_msg += f"""
The premise and the claim have the following neighbors in the conversation:
{annotation_context}
        """

    messages: list[ChatMessage] = [
        {
            "role": "system",
            "content": generate_system_message(options["include_neutral"]),
        },
        {
            "role": "user",
            "content": user_msg,
        },
    ]

    try:
        res = await chat_response(messages, options["use_llama"], model)
        prediction = res.lower()

        if "support" in prediction:
            return EntailmentType.ENTAILMENT_TYPE_ENTAILMENT
        elif "attack" in prediction:
            return EntailmentType.ENTAILMENT_TYPE_CONTRADICTION

    except OpenAIError:
        pass

    return EntailmentType.ENTAILMENT_TYPE_UNSPECIFIED


async def predict_sequential(
    annotations: abc.Sequence[Annotation],
    options: Options,
    model: ModelName = "gpt-3.5-turbo-1106",
) -> list[EntailmentType.ValueType]:
    predictions: list[EntailmentType.ValueType] = []
    messages: list[ChatMessage] = [
        {
            "role": "system",
            "content": generate_system_message(options["include_neutral"]),
        }
    ]

    for annotation in annotations:
        premise = annotation.adus[annotation.premise_id]
        claim = annotation.adus[annotation.claim_id]

        messages.append(
            {
                "role": "user",
                "content": f"""
Premise: {premise}.
Claim: {claim}.
""",
            }
        )

        try:
            res = await chat_response(messages, options["use_llama"], model)
            prediction = res.lower()

            if "support" in prediction:
                predictions.append(EntailmentType.ENTAILMENT_TYPE_ENTAILMENT)
            elif "attack" in prediction:
                predictions.append(EntailmentType.ENTAILMENT_TYPE_CONTRADICTION)
            else:
                predictions.append(EntailmentType.ENTAILMENT_TYPE_UNSPECIFIED)

            messages.append({"role": "assistant", "content": res})

        except OpenAIError:
            predictions.append(EntailmentType.ENTAILMENT_TYPE_UNSPECIFIED)

    return predictions


def batchify(seq: abc.Sequence[Any], n: int):
    for i in range(0, len(seq), n):
        yield seq[i : i + n]


async def predict_batched(
    annotations: abc.Sequence[Annotation],
    options: Options,
    model: ModelName = "gpt-3.5-turbo-1106",
    # model: ModelName = "gpt-4-1106-preview",
) -> list[EntailmentType.ValueType]:
    annotations_map = {
        (annotation.premise_id, annotation.claim_id): annotation
        for annotation in annotations
    }
    messages: list[ChatMessage] = [
        {
            "role": "system",
            "content": f"""
{generate_system_message(options["include_neutral"])},
You will be presented with a list of premise-claim pairs containing their text and id encoded as a JSON array.
Provide exactly one prediction for each of them.
""",
        }
    ]
    result_map: dict[tuple[str, str], str] = {}

    annotation_pairs = [
        {
            "premise_text": annotation.adus[annotation.premise_id],
            "premise_id": annotation.premise_id,
            "claim_text": annotation.adus[annotation.claim_id],
            "claim_id": annotation.claim_id,
        }
        for annotation in annotations
    ]

    messages.append(
        {
            "role": "user",
            "content": json.dumps(annotation_pairs),
        }
    )

    try:
        response = await client_openai.chat.completions.create(
            model=model,
            messages=messages,
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "predict_entailment",
                        "parameters": generate_schema(options["include_neutral"]),
                    },
                }
            ],
            tool_choice={
                "type": "function",
                "function": {"name": "predict_entailment"},
            },
        )
        tool_calls = response.choices[0].message.tool_calls
        assert tool_calls is not None

        result = json.loads(tool_calls[0].function.arguments)
        result_map.update(_result_to_dict(result))

        missing_pairs = set(annotations_map.keys()).difference(result_map.keys())

        if len(missing_pairs) > 0:
            messages.append(
                {"role": "assistant", "content": tool_calls[0].function.arguments}
            )
            messages.append(
                {
                    "role": "user",
                    "content": f"""
Some pairs are missing from your previous response.
Please provide predictions for the following pairs:
{json.dumps(list(missing_pairs))}
""",
                }
            )

            response = await client_openai.chat.completions.create(
                model=model,
                messages=messages,
                tools=[
                    {
                        "type": "function",
                        "function": {
                            "name": "predict_entailment",
                            "parameters": generate_schema(options["include_neutral"]),
                        },
                    }
                ],
                tool_choice={
                    "type": "function",
                    "function": {"name": "predict_entailment"},
                },
            )

            tool_calls = response.choices[0].message.tool_calls
            assert tool_calls is not None

            result = json.loads(tool_calls[0].function.arguments)
            result_map.update(_result_to_dict(result))

    except OpenAIError:
        pass

    return [polarity_map[result_map.get(key)] for key in annotations_map]


def _result_to_dict(result: Any) -> dict[tuple[str, str], str]:
    return {
        (item["premise_id"], item["claim_id"]): item["polarity_type"]
        for item in result["polarities"]
        if "polarity_type" in item and "premise_id" in item and "claim_id" in item
    }
