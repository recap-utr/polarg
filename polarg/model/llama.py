# https://github.com/huggingface/transformers/issues/25147#issuecomment-1707812777
import os
from typing import cast

from openai.types.chat import ChatCompletionMessageParam as Message
from torch.cuda import is_available as is_cuda_available
from transformers import AutoTokenizer, Conversation, ConversationalPipeline, pipeline

torch_device = "cuda" if is_cuda_available() else "cpu"
print(f"Using torch device '{torch_device}'.")

model = "meta-llama/Llama-2-13b-chat-hf"
hf_token = os.environ["HUGGINGFACE_API_KEY"]

pipeline_cache: dict[str, ConversationalPipeline] = {}
DEFAULT_CACHE = "default"


def init_chatbot() -> ConversationalPipeline:
    if DEFAULT_CACHE not in pipeline_cache:
        print("Loading llama tokenizer...")

        print("Loading llama model...")
        # TODO
        print("Finished loading llama.")

    return pipeline_cache[DEFAULT_CACHE]


chatbot = None

if torch_device == "cuda":
    tokenizer = AutoTokenizer.from_pretrained(model, token=hf_token)
    tokenizer.use_default_system_prompt = False  # type: ignore
    chatbot = cast(
        ConversationalPipeline,
        pipeline(
            "conversational",
            model=model,
            tokenizer=tokenizer,
            token=hf_token,
            # device=torch_device,
            device_map="auto",
        ),
    )


def generate(messages: list[Message], max_length: int = 512) -> str:
    # chatbot = init_chatbot()
    print("Generating response...")
    assert chatbot is not None

    conversation = cast(
        Conversation,
        chatbot(cast(list[dict[str, str]], messages), max_length=max_length),
    )
    print("Finished generating response.")

    return conversation.generated_responses[-1]
