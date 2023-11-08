import asyncio
import json
from collections import abc
from copy import deepcopy
from pathlib import Path
from typing import Any, Literal

from arg_services.mining.v1beta.entailment_pb2 import EntailmentType
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam as ChatMessage

from polarg.config import config
from polarg.model.annotation import Annotation

client = AsyncOpenAI()


# TODO: Add optional memory by appending responses until context size is reached. Needs rework of async feature since responses are retrieved in parallel


polarity_map = {
    "support": EntailmentType.ENTAILMENT_TYPE_ENTAILMENT,
    "attack": EntailmentType.ENTAILMENT_TYPE_CONTRADICTION,
    "neutral": EntailmentType.ENTAILMENT_TYPE_NEUTRAL,
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


async def predict(
    annotations: abc.Sequence[Annotation], strategy: Strategy
) -> list[EntailmentType.ValueType]:
    predictions: list[EntailmentType.ValueType] | None = None

    if strategy == "isolated":
        predictions = await asyncio.gather(
            *(predict_isolated(annotation) for annotation in annotations)
        )
    elif strategy == "sequential":
        predictions = await predict_sequential(annotations)
    elif strategy == "batched":
        predictions = await predict_batched(annotations)

    assert predictions is not None

    return predictions


async def predict_isolated(
    annotation: Annotation, model: ModelName = "gpt-3.5-turbo-1106"
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
            "content": generate_system_message(config.evaluate.include_neutral),
        },
        {
            "role": "user",
            "content": user_msg,
        },
    ]

    response = await client.chat.completions.create(
        model=model,
        messages=messages,
    )

    assistant_content = response.choices[0].message.content
    assert assistant_content is not None

    prediction = assistant_content.lower()

    if "support" in prediction:
        return EntailmentType.ENTAILMENT_TYPE_ENTAILMENT
    elif "attack" in prediction:
        return EntailmentType.ENTAILMENT_TYPE_CONTRADICTION

    return EntailmentType.ENTAILMENT_TYPE_NEUTRAL


async def predict_sequential(
    annotations: abc.Sequence[Annotation], model: ModelName = "gpt-3.5-turbo-1106"
) -> list[EntailmentType.ValueType]:
    predictions: list[EntailmentType.ValueType] = []
    messages: list[ChatMessage] = [
        {
            "role": "system",
            "content": generate_system_message(config.evaluate.include_neutral),
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

        response = await client.chat.completions.create(
            model=model,
            messages=messages,
        )
        assistant_msg = response.choices[0].message
        assert assistant_msg.content is not None

        prediction = assistant_msg.content.lower()

        if "support" in prediction:
            predictions.append(EntailmentType.ENTAILMENT_TYPE_ENTAILMENT)
        elif "attack" in prediction:
            predictions.append(EntailmentType.ENTAILMENT_TYPE_CONTRADICTION)
        else:
            predictions.append(EntailmentType.ENTAILMENT_TYPE_NEUTRAL)

        messages.append({"role": assistant_msg.role, "content": assistant_msg.content})

    return predictions


def batchify(seq: abc.Sequence[Any], n: int):
    for i in range(0, len(seq), n):
        yield seq[i : i + n]


async def predict_batched(
    annotations: abc.Sequence[Annotation],
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
{generate_system_message(config.evaluate.include_neutral)},
You will be presented with a list of premise-claim pairs containing their text and id encoded as a JSON array.
Provide exactly one prediction for each of them.
""",
        }
    ]

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

    response = await client.chat.completions.create(
        model=model,
        messages=messages,
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "predict_entailment",
                    "parameters": generate_schema(config.evaluate.include_neutral),
                },
            }
        ],
        tool_choice={"type": "function", "function": {"name": "predict_entailment"}},
    )

    tool_calls = response.choices[0].message.tool_calls
    assert tool_calls is not None

    result = json.loads(tool_calls[0].function.arguments)
    result_map = _result_to_dict(result)

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

        response = await client.chat.completions.create(
            model=model,
            messages=messages,
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "predict_entailment",
                        "parameters": generate_schema(config.evaluate.include_neutral),
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

    return [polarity_map[result_map.get(key, "neutral")] for key in annotations_map]


def _result_to_dict(result: Any) -> dict[tuple[str, str], str]:
    return {
        (item["premise_id"], item["claim_id"]): item["polarity_type"]
        for item in result["polarities"]
        if "polarity_type" in item and "premise_id" in item and "claim_id" in item
    }
