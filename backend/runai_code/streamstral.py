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


def format_messages(messages: List[Message]) -> str:
    prompt = "<s>"
    for message in messages:
        if message.role in ["user", "human", "system"]:
            prompt += f"[INST] {message.content} [/INST]\n"
        elif message.role == "assistant":
            if message.tool_calls:
                # Format tool calls according to Mistral's expectations
                tool_call = message.tool_calls[0]
                prompt += f"I will use the {tool_call['function']['name']} function with these arguments: {tool_call['function']['arguments']}</s><s>[INST] "
            else:
                prompt += f"{message.content} </s><s>[INST] "
        elif message.role == "tool":
            # Format tool response
            prompt += f"Tool {message.name} returned: {message.content} [/INST]\n"
    return prompt


async def generate_stream(prompt: str, request_id: str, sampling_params: SamplingParams,
                          tools: Optional[List[Tool]] = None):
    engine = create_engine()
    results_generator = engine.generate(prompt, sampling_params, request_id)
    last_text_length = 0

    async for request_output in results_generator:
        for output in request_output.outputs:
            current_text = output.text
            new_text = current_text[last_text_length:]
            last_text_length = len(current_text)

            if new_text:
                response_chunk = {
                    "id": f"chatcmpl-{request_id}",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": "local-model",
                    "choices": [{
                        "index": 0,
                        "delta": {},
                        "finish_reason": None
                    }]
                }

                # Check if the text appears to be a function call
                if "I will use the" in new_text and "function with these arguments:" in new_text:
                    try:
                        # Extract function name and arguments
                        pattern = r'I will use the (\w+) function with these arguments: ({.*})'
                        match = re.search(pattern, new_text)
                        if match:
                            func_name, args_str = match.groups()
                            response_chunk["choices"][0]["delta"]["tool_calls"] = [{
                                "type": "function",
                                "function": {
                                    "name": func_name,
                                    "arguments": args_str
                                }
                            }]
                        else:
                            response_chunk["choices"][0]["delta"]["content"] = new_text
                    except Exception as e:
                        print(f"Error parsing function call: {e}")
                        response_chunk["choices"][0]["delta"]["content"] = new_text
                else:
                    response_chunk["choices"][0]["delta"]["content"] = new_text

                yield f"data: {json.dumps(response_chunk)}\n\n"

    yield f"data: [DONE]\n\n"


async def get_full_response(generator):
    response_text = ""
    async for request_output in generator:
        for output in request_output.outputs:
            response_text = output.text
    return response_text


@app.post("/v1/chat/completions")
async def chat_completion(request: ChatCompletionRequest):
    engine = create_engine()
    prompt = format_messages(request.messages)
    request_id = f"req_{int(time.time())}"

    sampling_params = SamplingParams(
        temperature=request.temperature,
        max_tokens=request.max_tokens,
        stop=["</s>", "[INST]"]
    )

    if request.stream:
        return StreamingResponse(
            generate_stream(prompt, request_id, sampling_params, request.tools),
            media_type="text/event-stream"
        )

    # For non-streaming
    generator = engine.generate(prompt, sampling_params, request_id)
    response_text = await get_full_response(generator)

    # Check if response appears to be a function call
    try:
        if "I will use the" in response_text and "function with these arguments:" in response_text:
            pattern = r'I will use the (\w+) function with these arguments: ({.*})'
            match = re.search(pattern, response_text)
            if match:
                func_name, args_str = match.groups()
                return {
                    "id": f"chatcmpl-{request_id}",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": "local-model",
                    "choices": [{
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [{
                                "type": "function",
                                "function": {
                                    "name": func_name,
                                    "arguments": args_str
                                }
                            }]
                        },
                        "finish_reason": "tool_calls",
                        "index": 0
                    }]
                }
    except Exception as e:
        print(f"Error parsing function call: {e}")

    # Default response format
    return {
        "id": f"chatcmpl-{request_id}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": "local-model",
        "choices": [{
            "message": {
                "role": "assistant",
                "content": response_text
            },
            "finish_reason": "stop",
            "index": 0
        }]
    }


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