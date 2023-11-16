import asyncio
import json
import math
from collections import abc
from copy import deepcopy
from pathlib import Path
from typing import Any, Literal, TypedDict

import httpx
from arg_services.mining.v1beta.entailment_pb2 import EntailmentType
from openai import AsyncOpenAI, OpenAIError
from openai._types import NOT_GIVEN, NotGiven
from openai.types.chat import ChatCompletionMessage
from openai.types.chat import ChatCompletionMessageParam as ChatMessage
from openai.types.chat.completion_create_params import Function, FunctionCall

from polarg.config import config
from polarg.model.annotation import Annotation

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


def batchify(seq: abc.Sequence[Any], n: int):
    for i in range(0, len(seq), n):
        yield seq[i : i + n]


# sequential: send prompts one-by-one and append previous messages to the prompt to simulate memory. Does not use context.
# isolated: send prompts one-by-one without appending previous messages to the prompt. Uses context if `include_context` is true.
# batched: send all prompts at once and append previous messages to the prompt to simulate memory. Does not use context.
Strategy = Literal["isolated", "sequential", "batched"]


class Options(TypedDict):
    strategy: Strategy
    use_llama: bool
    include_neutral: bool


openai_models: dict[Strategy, ModelName] = {
    "isolated": "gpt-3.5-turbo-1106",
    "sequential": "gpt-3.5-turbo-1106",
    "batched": "gpt-4-1106-preview",
}


async def predict(
    annotations: abc.Sequence[Annotation], options: Options
) -> list[EntailmentType.ValueType]:
    predictions: list[EntailmentType.ValueType] | None = None
    llm = Llm(options)

    if options["strategy"] == "isolated":
        predictions = await asyncio.gather(
            *(llm.predict_isolated(annotation) for annotation in annotations)
        )
    elif options["strategy"] == "sequential":
        predictions = await llm.predict_sequential(annotations)
    elif options["strategy"] == "batched":
        predictions = await llm.predict_batched(annotations)

    assert predictions is not None

    return predictions


class Llm:
    def __init__(self, options: Options):
        self.options = options
        # self.llama_client = httpx.AsyncClient(
        #     timeout=httpx.Timeout(timeout=120, connect=5),
        #     limits=httpx.Limits(max_connections=100, max_keepalive_connections=20),
        #     base_url=f"http://{config.llama_address}/api",
        # )

        self.client = AsyncOpenAI(
            http_client=httpx.AsyncClient(
                http2=False if options["use_llama"] else True,
                timeout=httpx.Timeout(timeout=120, connect=5),
                limits=httpx.Limits(max_connections=100, max_keepalive_connections=20),
            ),
            max_retries=2,
            base_url=config.openai_proxy_address if options["use_llama"] else None,
        )

    # async def fetch_llama(
    #     self, message: str, system_message: str, json: bool = False
    # ) -> str:
    #     response = await self.llama_client.post(
    #         "generate",
    #         json={
    #             "prompt": message,
    #             "system_prompt": system_message,
    #             "format": "json" if json else "text",
    #         },
    #     )
    #     return response.json()["response"]

    async def generate(
        self,
        messages: list[ChatMessage],
        functions: list[Function] | NotGiven = NOT_GIVEN,
        function_call: FunctionCall | NotGiven = NOT_GIVEN,
    ) -> ChatCompletionMessage:
        response = await self.client.chat.completions.create(
            model="ollama/llama2:13b"
            if self.options["use_llama"]
            else openai_models[self.options["strategy"]],
            messages=messages,
            functions=functions,
            function_call=function_call,
        )

        response_msg = response.choices[0].message
        return response_msg

    async def predict_isolated(
        self, annotation: Annotation
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
                "content": generate_system_message(self.options["include_neutral"]),
            },
            {
                "role": "user",
                "content": user_msg,
            },
        ]

        try:
            res = await self.generate(messages)
            assert res.content is not None
            prediction = res.content.lower()

            if "support" in prediction:
                return EntailmentType.ENTAILMENT_TYPE_ENTAILMENT
            elif "attack" in prediction:
                return EntailmentType.ENTAILMENT_TYPE_CONTRADICTION

        except OpenAIError as e:
            print(e)

        return EntailmentType.ENTAILMENT_TYPE_UNSPECIFIED

    async def predict_sequential(
        self, annotations: abc.Sequence[Annotation]
    ) -> list[EntailmentType.ValueType]:
        predictions: list[EntailmentType.ValueType] = []
        messages: list[ChatMessage] = [
            {
                "role": "system",
                "content": generate_system_message(self.options["include_neutral"]),
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
                res = await self.generate(messages)
                assert res.content is not None
                prediction = res.content.lower()

                if "support" in prediction:
                    predictions.append(EntailmentType.ENTAILMENT_TYPE_ENTAILMENT)
                elif "attack" in prediction:
                    predictions.append(EntailmentType.ENTAILMENT_TYPE_CONTRADICTION)
                else:
                    predictions.append(EntailmentType.ENTAILMENT_TYPE_UNSPECIFIED)

                messages.append({"role": "assistant", "content": res.content})

            except OpenAIError as e:
                print(e)
                predictions.append(EntailmentType.ENTAILMENT_TYPE_UNSPECIFIED)

        return predictions

    async def predict_batched(
        self,
        annotations: abc.Sequence[Annotation],
    ) -> list[EntailmentType.ValueType]:
        annotations_map = {
            (annotation.premise_id, annotation.claim_id): annotation
            for annotation in annotations
        }
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

        max_samples_per_batch = 20
        number_of_batches = math.ceil(len(annotation_pairs) / max_samples_per_batch)
        batch_size = math.ceil(len(annotation_pairs) / number_of_batches)

        for batch in batchify(annotation_pairs, batch_size):
            messages: list[ChatMessage] = [
                {
                    "role": "system",
                    "content": f"""
{generate_system_message(self.options["include_neutral"])},
You will be presented with a list of premise-claim pairs containing their text and id encoded as a JSON array.
Provide exactly one prediction for each of them using the function `predict_entailment`.
""",
                }
            ]
            messages.append(
                {
                    "role": "user",
                    "content": json.dumps(batch),
                }
            )

            try:
                res = await self.generate(
                    messages=messages,
                    functions=[
                        {
                            "name": "predict_entailment",
                            "parameters": generate_schema(
                                self.options["include_neutral"]
                            ),
                        }
                    ],
                    function_call={"name": "predict_entailment"},
                )
                assert res.function_call is not None

                result = json.loads(res.function_call.arguments)
                batch_result_map = _result_to_dict(result)
                batch_annotations = {
                    (item["premise_id"], item["claim_id"]) for item in batch
                }

                missing_pairs = batch_annotations.difference(batch_result_map.keys())

                if len(missing_pairs) > 0:
                    messages.append(
                        {"role": "assistant", "content": res.function_call.arguments}
                    )
                    messages.append(
                        {
                            "role": "user",
                            "content": f"""
Some pairs are missing from your previous response.
Please provide predictions for the following pairs using the function `predict_entailment`:
{json.dumps(list(missing_pairs))}
""",
                        }
                    )

                    res = await self.generate(
                        messages=messages,
                        functions=[
                            {
                                "name": "predict_entailment",
                                "parameters": generate_schema(
                                    self.options["include_neutral"]
                                ),
                            }
                        ],
                        function_call={"name": "predict_entailment"},
                    )

                    assert res.function_call is not None

                    result = json.loads(res.function_call.arguments)
                    batch_result_map.update(_result_to_dict(result))

                result_map.update(batch_result_map)

            except OpenAIError as e:
                print(e)

        return [polarity_map[result_map.get(key)] for key in annotations_map]


def _result_to_dict(result: Any) -> dict[tuple[str, str], str]:
    return {
        (item["premise_id"], item["claim_id"]): item["polarity_type"]
        for item in result.get("polarities", [])
        if "polarity_type" in item and "premise_id" in item and "claim_id" in item
    }
