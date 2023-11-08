# https://github.com/huggingface/transformers/issues/25147#issuecomment-1707812777
import os
from typing import cast

from openai.types.chat import ChatCompletionMessageParam as Message
from transformers import AutoTokenizer, Conversation, ConversationalPipeline, pipeline

model = "meta-llama/Llama-2-13b-chat-hf"
hf_token = os.environ["HUGGINGFACE_API_KEY"]

_pipeline: ConversationalPipeline | None = None


def init_chatbot() -> ConversationalPipeline:
    global _pipeline

    if _pipeline is None:
        tokenizer = AutoTokenizer.from_pretrained(model, token=hf_token)
        tokenizer.use_default_system_prompt = False  # type: ignore

        _pipeline = cast(
            ConversationalPipeline,
            pipeline(
                "conversational",
                model=model,
                tokenizer=tokenizer,
                token=hf_token,
                device=0,
            ),
        )

    return _pipeline


def generate(messages: list[Message], max_length: int = 512) -> str:
    chatbot = init_chatbot()

    conversation = cast(
        Conversation,
        chatbot(cast(list[dict[str, str]], messages), max_length=max_length),
    )

    return conversation.generated_responses[-1]
