from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any, Union
import uvicorn
import json
import time
import httpx
import asyncio
import re

app = FastAPI()

# Configuration
LLM_SERVICE_URL = "http://localhost:8002/generate"


# Pydantic models for request/response structure
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


def get_model_type(model_path: str) -> str:
    """Determine if the model is Llama or Mistral based on the path."""
    model_path = model_path.lower()
    if "llama" in model_path:
        return "llama"
    elif "mistral" in model_path:
        return "mistral"
    else:
        return "auto"


def format_messages(messages: List[Message], model_type: str) -> str:
    """Format messages based on model type."""
    if model_type == "mistral":
        prompt = ""
        conversation_start = True

        for message in messages:
            if conversation_start:
                prompt += "<s>"
                conversation_start = False

            if message.role in ["user", "human"]:
                prompt += f"[INST] {message.content} [/INST]"
            elif message.role == "assistant":
                prompt += f"{message.content}"
                if not any(m.role in ["user", "human"] for m in messages[messages.index(message) + 1:]):
                    prompt += "</s>"
                    conversation_start = True
            elif message.role == "system":
                if len(prompt) == 0:
                    prompt = f"<s>[INST] {message.content}\n"
            elif message.role == "tool":
                prompt += f"\nTool {message.name} returned: {message.content}\n"

        if not prompt.endswith("</s>"):
            prompt += "</s>"

        return prompt

    else:  # Llama formatting
        prompt = "<s>"
        conversation_start = True

        for message in messages:
            if message.role in ["user", "human"]:
                if not conversation_start:
                    prompt += "</s><s>"
                prompt += f"[INST] {message.content} [/INST]"
                conversation_start = False
            elif message.role == "assistant":
                prompt += f"{message.content}"
                conversation_start = True
            elif message.role == "system":
                if conversation_start:
                    prompt += f"[INST] <<SYS>>\n{message.content}\n<</SYS>>\n\n"
            elif message.role == "tool":
                prompt += f"Tool {message.name} returned: {message.content}\n"

        if not conversation_start:
            prompt += "</s>"
        else:
            prompt += "</s>"

        return prompt


async def should_use_tool(messages: List[Message], tools: Optional[List[Tool]] = None) -> bool:
    if not tools:
        return False

    conversation = ''
    for msg in messages:
        conversation += f"{msg.role}:{msg.content}\n"

    last_user_message = next((msg.content for msg in reversed(messages) if msg.role == "user"), "")

    tool_descriptions = [
        f"{tool.function['name']}: {tool.function['description']}"
        for tool in tools
    ]
    tools_text = "\n".join(tool_descriptions)

    decision_prompt = f"<s>[INST]Tu es en charge de décider si l'on va utiliser un outil pour répondre à la question de l'utilisateur, si tu as un doute, utilise l'outil, il vaut mieux l'utiliser trop que pas assez. Mais si c'est clair qu'il n'y a pas besoin, par exemple si l'utilisateur dit juste bonjour, ne l'utilise pas. tu répond oui pour utiliser l'outil et non pour ne pas l'utiliser. Tu as accès aux outils suivants:\n{tools_text}\n\nPour la question: \"{last_user_message}\"\n dans le contexte {conversation}\n\nRéponds seulement par Oui ou Non: As-tu besoin d'utiliser un outil pour répondre à cette question ? [/INST]"

    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            response = await client.post(
                LLM_SERVICE_URL,
                json={
                    "prompt": decision_prompt,
                    "sampling_params": {
                        "temperature": 0.1,
                        "max_tokens": 10,
                        "stop": ["</s>", "[INST]"]
                    },
                    "request_id": f"decision_{int(time.time())}"
                }
            )

            if response.status_code == 200:
                result = response.json()
                response_text = result["text"].strip().lower()
                return "oui" in response_text or "yes" in response_text
            else:
                print(f"Error in tool decision: {response.text}")
                return False
        except httpx.TimeoutError:
            print("Timeout while connecting to LLM service")
            return False
        except Exception as e:
            print(f"Error connecting to LLM service: {e}")
            return False


async def generate_tool_query(request_id: str, user_messages: List[Message], tools: List[Tool]) -> tuple[str, str]:
    conversation = ''
    for msg in user_messages:
        conversation += f"{msg.role}:{msg.content}\n"

    last_user_message = next((msg.content for msg in reversed(user_messages) if msg.role == "user"), "")

    tools_description = "\n".join([
        f"- {tool.function['name']}: {tool.function['description']}"
        for tool in tools
    ])

    tool_names = [tool.function['name'] for tool in tools]
    tool_names_str = ", ".join(tool_names)

    model_type = get_model_type("llama")  # You might want to make this configurable

    if model_type == "mistral":
        prompt = f"""<s>[INST] Tu as accès aux outils suivants:
{tools_description}

Pour la question de l'utilisateur: "{last_user_message}"
Dans le contexte: "{conversation}"

1. Choisis l'outil le plus approprié parmi: {tool_names_str}
2. Crée une requête pour l'outil

Réponds exactement dans ce format:
TOOL: <tool_name>
QUERY: <your_query> [/INST]"""
    else:
        prompt = f"""<s>[INST] <<SYS>>
You are a helpful AI assistant tasked with selecting and using tools appropriately.
<</SYS>>

You have access to the following tools:
{tools_description}

For the user question: "{last_user_message}"
In the context: "{conversation}"

1. Choose the most appropriate tool from: {tool_names_str}
2. Create a query for the tool

Answer in this exact format:
TOOL: <tool_name>
QUERY: <your_query> [/INST]"""

    async with httpx.AsyncClient() as client:
        response = await client.post(
            LLM_SERVICE_URL,
            json={
                "prompt": prompt,
                "sampling_params": {
                    "temperature": 0.1,
                    "max_tokens": 200,
                    "stop": ["</s>", "[INST]"]
                },
                "request_id": f"tool_query_{request_id}"
            }
        )

        if response.status_code == 200:
            result = response.json()
            response_text = result["text"].strip()

            tool_match = re.search(r"TOOL:\s*(\w+)", response_text)
            query_match = re.search(r"QUERY:\s*(.+)", response_text, re.DOTALL)

            if tool_match and query_match:
                tool_name = tool_match.group(1).strip()
                tool_query = query_match.group(1).strip()

                if tool_name in tool_names:
                    return tool_name, tool_query

        # Default to first tool if anything fails
        return tools[0].function['name'], last_user_message


async def generate_normal_stream(prompt: str, request_id: str, sampling_params: Dict[str, Any]):
    """Generate a normal streaming response."""
    print("Starting normal response generation")

    try:
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

        # Make streaming request to LLM service
        async with httpx.AsyncClient() as client:
            async with client.stream(
                    'POST',
                    LLM_SERVICE_URL,
                    json={
                        "prompt": prompt,
                        "sampling_params": sampling_params,
                        "request_id": request_id
                    },
                    timeout=60.0
            ) as response:
                if response.status_code != 200:
                    raise HTTPException(status_code=response.status_code, detail="LLM service error")

                # Process SSE stream
                buffer = ""
                async for chunk in response.aiter_bytes():
                    buffer += chunk.decode()

                    while "\n\n" in buffer:
                        message, buffer = buffer.split("\n\n", 1)
                        if message.startswith("data: "):
                            data = message[6:]  # Remove "data: " prefix

                            if data == "[DONE]":
                                continue

                            try:
                                event_data = json.loads(data)
                                if event_data.get("finished", False):
                                    continue

                                # Create content chunk
                                content_chunk = {
                                    "id": f"chatcmpl-{request_id}",
                                    "object": "chat.completion.chunk",
                                    "created": int(time.time()),
                                    "model": "local-model",
                                    "choices": [{
                                        "index": 0,
                                        "delta": {"content": event_data.get("text", "")},
                                        "finish_reason": None
                                    }]
                                }
                                yield f"data: {json.dumps(content_chunk)}\n\n"
                            except json.JSONDecodeError:
                                print(f"Failed to parse SSE data: {data}")
                                continue

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
        error_chunk = {
            "id": f"chatcmpl-{request_id}",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": "local-model",
            "choices": [{
                "index": 0,
                "delta": {"content": f"Error occurred during generation: {str(e)}"},
                "finish_reason": "error"
            }]
        }
        yield f"data: {json.dumps(error_chunk)}\n\n"
        yield "data: [DONE]\n\n"


async def generate_tool_stream(request_id: str, messages: List[Message], tools: List[Tool]):
    """Generate a streaming response for tool usage."""
    print("Starting tool response generation")

    # Generate the tool query
    tool_name, tool_query = await generate_tool_query(request_id, messages, tools)
    print(f"Selected tool: {tool_name}")
    print(f"Generated query: {tool_query}")

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
                    "id": tool_call_id,
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

    use_tool = await should_use_tool(request.messages, request.tools)
    print(f"Decision to use tool: {use_tool}")

    request_id = f"req_{int(time.time())}"
    model_type = get_model_type(request.model)

    if use_tool:
        print("Using tool for conversation")
        return StreamingResponse(
            generate_tool_stream(request_id, request.messages, request.tools),
            media_type="text/event-stream"
        )
    else:
        prompt = format_messages(request.messages, model_type)
        print(f"Using normal conversation for prompt: {prompt}")

        sampling_params = {
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
            "top_p": 0.9,
            "presence_penalty": 0.6,
            "frequency_penalty": 0.6,
            "stop": ["</s>", "[INST]"]
        }

        return StreamingResponse(
            generate_normal_stream(prompt, request_id, sampling_params),
            media_type="text/event-stream"
        )


if __name__ == "__main__":
    uvicorn.run("formatting_service:app", host="0.0.0.0", port=8001, reload=True)