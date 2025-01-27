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
import torch


# Working code
app = FastAPI()

# Global variable for the engine
engine = None


# def create_engine():
#     global engine
#     if engine is None:
#         engine_args = AsyncEngineArgs(
#             # model="/data/mistral_models/Mistral-Nemo-Instruct-12B",
#             model="/data/models2/llama3.3-70B/AWQ_4bit_quantization",
#             tokenizer_mode="mistral",
#             load_format="mistral",
#             config_format="mistral",
#             max_model_len=50000,
#             tensor_parallel_size=max(1, torch.cuda.device_count()),
#             gpu_memory_utilization=0.85
#         )
#         engine = AsyncLLMEngine.from_engine_args(engine_args)
#     return engine

def create_engine():
    global engine
    if engine is None:
        engine_args = AsyncEngineArgs(
            model="/data/models2/llama3.3-70B/AWQ_4bit_quantization",
            tokenizer=None,
            tokenizer_mode="auto",
            trust_remote_code=True,
            quantization="awq_marlin",  # Using AWQ-Marlin for better performance
            load_format="auto",
            dtype="auto",
            max_model_len=50000,
            tensor_parallel_size=4,  # Use all 4 GPUs
            gpu_memory_utilization=0.85,
            max_num_batched_tokens=2048,  # Control memory usage
            max_num_seqs=256,  # Limit concurrent sequences
            enable_chunked_prefill=True  # Enable chunked prefill for memory efficiency
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

    # Check the message sequence
    last_messages = list(reversed(messages))

    # Get the last user message and check what comes after it
    for i, msg in enumerate(last_messages):
        if msg.role == "user":
            # If this is the latest message, we can evaluate it for tool use
            if i == 0:
                break
            # If the messages after the user message include tool or assistant responses,
            # we're in the middle of a tool-using conversation
            for earlier_msg in last_messages[:i]:
                if earlier_msg.role in ["tool", "assistant"] and earlier_msg.tool_calls:
                    return False
            break
    else:
        # No user message found
        return False

    # Get the last user message
    user_question = last_messages[i].content

    # Create tool descriptions
    tool_descriptions = [
        f"{tool.function['name']}: {tool.function['description']}"
        for tool in tools
    ]
    tools_text = "\n".join(tool_descriptions)

    decision_prompt = f"<s>[INST]Tu es en charge de décider si l'on va utiliser un outil pour répondre à la question de l'utilisateur, si tu as un doute, utilise l'outil, il vaut mieux l'utiliser trop que pas assez. Mais si c'est clair qu'il n'y a pas besoin, par exemple si l'utilisateur dit juste bonjour, ne l'utilise pas. tu répond oui pour utiliser l'outil et non pour ne pas l'utiliser. Tu as accès aux outils suivants:\n{tools_text}\n\nPour la question: \"{user_question}\"\n\nRéponds seulement par Oui ou Non: As-tu besoin d'utiliser un outil pour répondre à cette question ? [/INST]"

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
        return "oui" in response_text or "yes" in response_text

    except Exception as e:
        print(f"Error in tool decision: {e}")
        traceback.print_exc()
        return False


# def format_normal_messages(messages: List[Message]) -> str:
#     """Format messages for normal conversation without tools."""
#     prompt = "<s>"
#     for message in messages:
#         if message.role in ["user", "human", "system"]:
#             prompt += f"[INST] {message.content} [/INST]\n"
#         elif message.role == "assistant":
#             prompt += f"{message.content}\n"
#         elif message.role == "tool":
#             prompt += f"Tool {message.name} returned: {message.content}\n"
#     return prompt


def format_normal_messages(messages: List[Message]) -> str:
    """Format messages for normal conversation without tools."""
    # For Llama models
    prompt = "<s>[INST] "
    conversation_start = True

    for message in messages:
        if message.role in ["user", "human"]:
            if not conversation_start:
                prompt += "[/INST]</s>[INST] "
            prompt += f"{message.content} "
            conversation_start = False
        elif message.role == "assistant":
            prompt += f"[/INST]{message.content}</s>"
            conversation_start = True
        elif message.role == "system":
            # For system messages in Llama, we include them in the first user message
            if conversation_start:
                prompt += f"<<SYS>>\n{message.content}\n<</SYS>>\n\n"
        elif message.role == "tool":
            prompt += f"Tool {message.name} returned: {message.content}\n"

    # If the last message was from a user, close the instruction
    if not conversation_start:
        prompt += "[/INST]"

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


async def generate_tool_query(request_id: str, user_messages: List[Message], tools: List[Tool]) -> tuple[str, str]:
    """
    Generate a query for the appropriate tool based on the user's message.

    Args:
        request_id (str): Unique identifier for the request
        user_messages (List[Message]): List of conversation messages
        tools (List[Tool]): List of available tools

    Returns:
        tuple[str, str]: A tuple containing (tool_name, tool_query)
    """
    # Get the last user message
    last_user_message = next((msg.content for msg in reversed(user_messages) if msg.role == "user"), "")

    # Create a prompt that asks the model to select a tool and generate a query
    tools_description = "\n".join([
        f"- {tool.function['name']}: {tool.function['description']}"
        for tool in tools
    ])

    tool_names = [tool.function['name'] for tool in tools]
    tool_names_str = ", ".join(tool_names)

    prompt = f"""<s>[INST] Tu as accès aux outils suivants:
{tools_description}

Pour la question de l'utilisateur: "{last_user_message}"

1. Choisis l'outil le plus approprié parmi: {tool_names_str}
2. Génère une requête précise pour cet outil

Réponds exactement dans ce format:
TOOL: <nom_de_l_outil>
QUERY: <ta_requête>
[/INST]"""

    engine = create_engine()
    sampling_params = SamplingParams(
        temperature=0.1,  # Low temperature for more deterministic response
        max_tokens=200,
        stop=["</s>", "[INST]"]
    )

    try:
        # Generate the response
        response_text = ""
        generator = engine.generate(prompt, sampling_params, f"tool_query_{request_id}")
        async for request_output in generator:
            for output in request_output.outputs:
                response_text = output.text.strip()

        # Parse the response to extract tool name and query
        tool_match = re.search(r"TOOL:\s*(\w+)", response_text)
        query_match = re.search(r"QUERY:\s*(.+)", response_text, re.DOTALL)

        if tool_match and query_match:
            tool_name = tool_match.group(1).strip()
            tool_query = query_match.group(1).strip()

            # Verify that the selected tool exists
            if tool_name in tool_names:
                return tool_name, tool_query

        # Default to first tool if parsing fails
        return tools[0].function['name'], last_user_message

    except Exception as e:
        print(f"Error generating tool query: {e}")
        traceback.print_exc()
        # Fallback to first tool and original message
        return tools[0].function['name'], last_user_message


async def generate_tool_stream(request_id: str, messages: List[Message], tools: List[Tool]):
    """Generate a streaming response for tool usage."""
    print("Starting tool response generation")

    # Generate the tool query
    tool_name, tool_query = await generate_tool_query(request_id, messages, tools)
    print(f"Selected tool: {tool_name}")
    print(f"Generated query: {tool_query}")

    # Generate a unique tool call ID
    tool_call_id = f"call_{request_id}_{int(time.time())}"

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

    # Send tool call chunk with tool_call_id
    tool_call_chunk = {
        "id": f"chatcmpl-{request_id}",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": "local-model",
        "choices": [{
            "index": 0,
            "delta": {
                "tool_calls": [{
                    "id": tool_call_id,  # Add this
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "arguments": json.dumps({"question": tool_query}, ensure_ascii=False)
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
        print("Using tool for conversation")
        return StreamingResponse(
            generate_tool_stream(request_id, request.messages, request.tools),
            media_type="text/event-stream"
        )
    else:
        # Regular conversation without tools
        prompt = format_normal_messages(request.messages)
        print(f"Using normal conversation for prompt: {prompt}")

        sampling_params = SamplingParams(
            temperature=1.0,
            max_tokens=request.max_tokens,
            top_p=0.9,
            presence_penalty=0.6,
            frequency_penalty=0.6,
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