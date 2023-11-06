import json
from pathlib import Path

from arg_services.mining.v1beta.entailment_pb2 import EntailmentType
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam as ChatMessage

from polarg.model.annotation import Annotation

client = AsyncOpenAI()


# TODO: Add optional memory by appending responses until context size is reached
# TODO: Respect "config.evaluateinclude_neutral" setting by altering the prompt and the function calling scheme


type_map = {
    "support": EntailmentType.ENTAILMENT_TYPE_ENTAILMENT,
    "attack": EntailmentType.ENTAILMENT_TYPE_CONTRADICTION,
    "neutral": EntailmentType.ENTAILMENT_TYPE_NEUTRAL,
}

# token_limits = {
#     "gpt-4": 8192,
#     "gpt-4-32k": 32768,
#     "gpt-3.5-turbo": 4096,
#     "gpt-3.5-turbo-16k": 16385,
# }

with Path("./openai_schema.json").open("r") as fp:
    schema = json.load(fp)


async def predict(
    annotations: list[Annotation], model: str
) -> list[dict[EntailmentType.ValueType, float]]:
    predictions = []

    for annotation in annotations:
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
                "content": user_msg,
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
