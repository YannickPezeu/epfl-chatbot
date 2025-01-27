from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any, Union
import uvicorn
import json
import time
import asyncio
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm import AsyncLLMEngine, SamplingParams
import torch.multiprocessing as mp
import atexit
import re
import traceback

app = FastAPI()

# Global variable for the engine
engine = None


def create_engine():
    global engine
    if engine is None:
        engine_args = AsyncEngineArgs(
            model="/data/mistral_models/Mistral-Nemo-Instruct-12B",
            tokenizer_mode="mistral",
            load_format="mistral",
            config_format="mistral",
            max_model_len=8192,
            tensor_parallel_size=2,
            gpu_memory_utilization=0.85
        )
        engine = AsyncLLMEngine.from_engine_args(engine_args)
    return engine


class Tool(BaseModel):
    type: str
    function: Dict[str, Any]


class Message(BaseModel):
    role: str
    content: str
    name: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_call_id: Optional[str] = None


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Message]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 2048
    stream: Optional[bool] = False
    tools: Optional[List[Tool]] = None
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None


def convert_tool_to_dict(tool: Tool) -> dict:
    """Convert a Tool object to a dictionary."""
    return {
        "type": tool.type,
        "function": tool.function
    }


async def should_use_tool(messages: List[Message], tools: Optional[List[Tool]] = None) -> bool:
    """Determine if we should use a tool for the current query."""
    if not tools:
        return False

    # Create a decision prompt
    tool_descriptions = [
        f"{tool.function['name']}: {tool.function['description']}"
        for tool in tools
    ]
    tools_text = "\n".join(tool_descriptions)

    # Get the user's question (last message)
    user_question = next((msg.content for msg in reversed(messages) if msg.role == "user"), None)
    if not user_question:
        return False

    decision_prompt = f"<s>[INST] Tu as accès aux outils suivants:\n{tools_text}\n\nPour la question: \"{user_question}\"\n\nRéponds seulement par Oui ou Non: As-tu besoin d'utiliser un outil pour répondre à cette question ? [/INST]"

    print(f"\nDecision prompt:\n{decision_prompt}")

    engine = create_engine()
    sampling_params = SamplingParams(
        temperature=0.1,  # Low temperature for more deterministic response
        max_tokens=10,  # We only need a short response
        stop=["</s>", "[INST]"]
    )

    try:
        generator = engine.generate(decision_prompt, sampling_params, f"decision_{int(time.time())}")
        response_text = ""
        async for request_output in generator:
            for output in request_output.outputs:
                response_text = output.text.strip().lower()

        print(f"Tool decision response: {response_text}")
        return "oui" in response_text or "yes" in response_text or 'non' in response_text

    except Exception as e:
        print(f"Error in tool decision: {e}")
        traceback.print_exc()
        return False


def format_normal_messages(messages: List[Message]) -> str:
    """Format messages for normal conversation without tools."""
    prompt = "<s>"
    for message in messages:
        if message.role in ["user", "human", "system"]:
            prompt += f"[INST] {message.content} [/INST]\n"
        elif message.role == "assistant":
            prompt += f"{message.content}\n"
        elif message.role == "tool":
            prompt += f"Tool {message.name} returned: {message.content}\n"
    return prompt


async def generate_normal_stream(prompt: str, request_id: str, sampling_params: SamplingParams):
    """Generate a normal streaming response without tools."""
    print("Starting normal response generation")
    print(f"Normal prompt: {prompt}")

    engine = create_engine()
    results_generator = engine.generate(prompt, sampling_params, request_id)
    last_text_length = 0
    content_sent = False

    # Send initial role chunk
    initial_chunk = {
        "id": f"chatcmpl-{request_id}",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": "local-model",
        "choices": [{
            "index": 0,
            "delta": {"role": "assistant"},
            "finish_reason": None
        }]
    }
    yield f"data: {json.dumps(initial_chunk)}\n\n"

    try:
        # Collect complete response
        complete_text = ""
        async for request_output in results_generator:
            for output in request_output.outputs:
                current_text = output.text
                new_text = current_text[last_text_length:]
                last_text_length = len(current_text)
                complete_text = current_text

                if new_text.strip():
                    content_sent = True
                    chunk = {
                        "id": f"chatcmpl-{request_id}",
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": "local-model",
                        "choices": [{
                            "index": 0,
                            "delta": {"content": new_text},
                            "finish_reason": None
                        }]
                    }
                    yield f"data: {json.dumps(chunk)}\n\n"

        # If no content was sent, send a default greeting
        if not content_sent:
            default_chunk = {
                "id": f"chatcmpl-{request_id}",
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": "local-model",
                "choices": [{
                    "index": 0,
                    "delta": {"content": "Bonjour! Comment puis-je vous aider aujourd'hui?"},
                    "finish_reason": None
                }]
            }
            yield f"data: {json.dumps(default_chunk)}\n\n"

        # Send finish chunk
        finish_chunk = {
            "id": f"chatcmpl-{request_id}",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": "local-model",
            "choices": [{
                "index": 0,
                "delta": {},
                "finish_reason": "stop"
            }]
        }
        yield f"data: {json.dumps(finish_chunk)}\n\n"
        yield "data: [DONE]\n\n"

    except Exception as e:
        print(f"Error in stream generation: {e}")
        traceback.print_exc()
        error_chunk = {
            "id": f"chatcmpl-{request_id}",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": "local-model",
            "choices": [{
                "index": 0,
                "delta": {"content": "Error occurred during generation"},
                "finish_reason": "error"
            }]
        }
        yield f"data: {json.dumps(error_chunk)}\n\n"
        yield "data: [DONE]\n\n"


async def generate_tool_stream(request_id: str, user_message: str):
    """Generate a streaming response for tool usage."""
    print("Starting tool response generation")
    print(f"User message for tool: {user_message}")

    # Send initial role chunk
    initial_chunk = {
        "id": f"chatcmpl-{request_id}",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": "local-model",
        "choices": [{
            "index": 0,
            "delta": {"role": "assistant"},
            "finish_reason": None
        }]
    }
    yield f"data: {json.dumps(initial_chunk)}\n\n"

    # Send tool call chunk
    tool_call_chunk = {
        "id": f"chatcmpl-{request_id}",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": "local-model",
        "choices": [{
            "index": 0,
            "delta": {
                "tool_calls": [{
                    "type": "function",
                    "function": {
                        "name": "search_engine_tool",
                        "arguments": json.dumps({"question": user_message}, ensure_ascii=False)
                    }
                }]
            },
            "finish_reason": None
        }]
    }
    yield f"data: {json.dumps(tool_call_chunk)}\n\n"

    # Send finish chunk
    finish_chunk = {
        "id": f"chatcmpl-{request_id}",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": "local-model",
        "choices": [{
            "index": 0,
            "delta": {},
            "finish_reason": "tool_calls"
        }]
    }
    yield f"data: {json.dumps(finish_chunk)}\n\n"
    yield "data: [DONE]\n\n"


@app.post("/v1/chat/completions")
async def chat_completion(request: ChatCompletionRequest):
    print("\n=== Starting new chat completion request ===")
    print(f"Messages: {json.dumps([msg.dict() for msg in request.messages], indent=2)}")
    print(f"Tools available: {bool(request.tools)}")

    # First, decide if we should use a tool
    use_tool = await should_use_tool(request.messages, request.tools)
    print(f"Decision to use tool: {use_tool}")

    request_id = f"req_{int(time.time())}"

    if use_tool:
        # Get the last user message for the tool query
        last_user_message = next((msg.content for msg in reversed(request.messages) if msg.role == "user"), "")
        print(f"Using tool for message: {last_user_message}")

        return StreamingResponse(
            generate_tool_stream(request_id, last_user_message),
            media_type="text/event-stream"
        )
    else:
        # Regular conversation without tools
        prompt = format_normal_messages(request.messages)
        print(f"Using normal conversation for prompt: {prompt}")

        sampling_params = SamplingParams(
            temperature=1.0,  # Slightly higher temperature for more varied responses
            max_tokens=request.max_tokens,
            top_p=0.9,  # Add top_p sampling
            presence_penalty=0.6,  # Encourage new content
            frequency_penalty=0.6,  # Discourage repetition
            stop=["</s>", "[INST]"]
        )

        return StreamingResponse(
            generate_normal_stream(prompt, request_id, sampling_params),
            media_type="text/event-stream"
        )


@app.on_event("startup")
async def startup_event():
    create_engine()


def cleanup():
    global engine
    if engine:
        del engine


atexit.register(cleanup)

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    uvicorn.run("server:app", host="0.0.0.0", port=8001, reload=False)