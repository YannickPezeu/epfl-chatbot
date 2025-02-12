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
import uuid

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


print('LLM_SERVCICE_URL', LLM_SERVICE_URL.split('generate')[0])
print(f"{LLM_SERVICE_URL.split('generate')[0]}model_type")


def get_model_type() -> str:
    """Get model type from LLM service."""

    try:
        import requests
        response = requests.get(f"{LLM_SERVICE_URL.split('generate')[0]}model_type")
        if response.status_code == 200:
            return response.json()["type"]
        else:
            print(f"Error getting model type: {response.status_code}")
            return "auto"
    except Exception as e:
        print(f"Error querying model type: {e}")
        return "auto"


# model_type = get_model_type()
model_type = 'mistral'
print('model_type', model_type)


def format_messages(messages: List[Message], model_type: str, use_tool: bool, skip_system_msg: bool = False) -> str:
    """Format messages based on model type."""
    if model_type == "mistral":
        prompt = ""

        for i, message in enumerate(messages):
            if message.role == "system":
                # For Mistral, system message should be included in first user message
                continue
            elif message.role in ["user", "human"]:
                prefix = "[INST]"
                if i == 0 and any(m.role == "system" for m in messages):
                    # Add system message to first user message
                    system_msg = next((m.content for m in messages if m.role == "system"), "")
                    prefix = f"[INST] {system_msg}\n\n"
                prompt += f"{prefix} {message.content} [/INST]"
            elif message.role == "assistant":
                prompt += f"{message.content}"
            elif message.role == "tool":
                prompt += f"\nTool {message.name} returned: {message.content}\n"

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
    print('start should use tool')
    if not tools:
        return False

    conversation = ''
    for msg in messages:
        print('msg_role', msg.role)
        print('msg', msg.content[:50] + ' ... ' + msg.content[-50:])
        print('-' * 100)
        if msg.role == 'tool':
            return False
        if msg.role == 'system':
            continue
        conversation += f"{msg.role}:{msg.content}\n"

    tool_descriptions = [
        f"{tool.function['name']}: {tool.function['description']}"
        for tool in tools
    ]
    tools_text = "\n".join(tool_descriptions)

    if model_type == 'mistral':
        decision_prompt = f"[INST] You need to decide if the search_engine should be used to answer the user's question. If in doubt, use the search_engine - it's better to use it too much than not enough. But if it's clearly not needed, like if the user just says hello, don't use it. Answer only with 'yes' or 'no', nothing else: Do you need to use a search_engine to answer this question? [/INST]"
        decision_prompt = decision_prompt + '\n The conversation is the following:' + conversation + decision_prompt
    else:
        decision_prompt = f"<s>[INST]Basé sur la discussion précédente: Tu es en charge de décider si l'on va utiliser un outil pour répondre à la question de l'utilisateur, si tu as un doute, utilise l'outil, il vaut mieux l'utiliser trop que pas assez. Mais si c'est clair qu'il n'y a pas besoin, par exemple si l'utilisateur dit juste bonjour, ne l'utilise pas. tu répond seulement par 'oui' ou 'non', rien d'autre: As-tu besoin d'utiliser un outil pour répondre à cette question ? [/INST]"
        decision_prompt = decision_prompt + '\n The conversation is the following:' + conversation

    print('should_use_tool decision prompt', decision_prompt)
    print('-' * 100)

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                LLM_SERVICE_URL,
                json={
                    "prompt": decision_prompt,
                    "sampling_params": {
                        "temperature": 0.1,
                        "max_tokens": 100,
                        "stop": ["</s>", "[INST]"],
                        "top_p": 0.9,
                        "frequency_penalty": 0.1
                    },
                    "request_id": f"decision_{int(time.time())}"
                }
            )

            if response.status_code != 200:
                print(f"Error in tool decision: Status {response.status_code}")
                return False

            # Process the streaming response
            complete_response = ""
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    try:
                        data = line.removeprefix("data: ")
                        if data == "[DONE]":
                            continue

                        chunk_data = json.loads(data)
                        if isinstance(chunk_data, dict) and "text" in chunk_data:
                            complete_response += chunk_data["text"]
                    except json.JSONDecodeError:
                        continue

            # Clean up and check the complete response
            complete_response = complete_response.lower().strip()
            print(f"Complete response: {complete_response}")  # For debugging
            print('end should use tool')
            return "oui" in complete_response or "yes" in complete_response

    except httpx.RequestError as e:
        print(f"Request error in tool decision: {e}")
        return False
    except Exception as e:
        print(f"Unexpected error in tool decision: {e}")
        return False


async def generate_tool_query(request_id: str, user_messages: List[Message], tools: List[Tool]) -> tuple[str, str]:
    conversation = ''
    for msg in user_messages:
        if msg.role in ['system', 'tool']:
            continue
        conversation += f"{msg.role}:{msg.content}\n"

    last_user_message = next((msg.content for msg in reversed(user_messages) if msg.role == "user"), "")

    tools_description = "\n".join([
        f"- {tool.function['name']}: {tool.function['description']}"
        for tool in tools
    ])

    tool_names = [tool.function['name'] for tool in tools]
    tool_names_str = ", ".join(tool_names)

    if model_type == "mistral":
        prompt = f"""<s>[INST] You have access to the following tools:
            {tools_description}

            For the user question: "{last_user_message}"
            In the context: "{conversation}"

            1. Choose the most appropriate tool from: {tool_names_str}
            2. Create a query for the tool

            Output your response as a JSON object containing the tool name and query.
            [/INST]"""
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

            Output your response as a JSON object containing the tool name and query.
            [/INST]"""

    async with httpx.AsyncClient() as client:
        response = await client.post(
            LLM_SERVICE_URL,
            json={
                "prompt": prompt,
                "sampling_params": {
                    "temperature": 0.1,
                    "max_tokens": 200,
                    "stop": ["</s>", "[INST]", "[/INST]"]
                },
                "request_id": f"tool_query_{request_id}"
            }
        )

        if response.status_code == 200:
            complete_response = ""
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data = line.removeprefix("data: ")
                    if data == "[DONE]":
                        continue
                    try:
                        chunk_data = json.loads(data)
                        if isinstance(chunk_data, dict) and "text" in chunk_data:
                            complete_response += chunk_data["text"]
                    except json.JSONDecodeError:
                        continue

            response_text = complete_response.strip()

            try:
                # Extract JSON from markdown code block if present
                json_match = re.search(r'```(?:json)?\s*({[^}]+})\s*```', response_text)
                if json_match:
                    response_text = json_match.group(1)

                # Clean up any potential whitespace or newlines
                response_text = response_text.strip()

                # Parse the JSON response
                response_json = json.loads(response_text)
                tool_name = response_json.get("tool", "").strip()
                tool_query = response_json.get("query", "").strip()

                if tool_name in tool_names:
                    return tool_name, tool_query

            except json.JSONDecodeError:
                print(f"Failed to parse JSON response: {response_text}")

        # If anything fails, generate an appropriate query for the first tool
        first_tool = tools[0].function['name']
        return first_tool, generate_default_query(first_tool, last_user_message)


def generate_default_query(tool_name: str, user_message: str) -> str:
    """Generate a default query when JSON parsing fails."""
    return f"Search for information about: {user_message}"


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
    print('request:', request)
    print('=-'*100)
    msgs = request.messages
    user_msgs = [msg for msg in msgs if msg.role =='user']
    for msg in request.messages:
        if msg.role == 'tool':
            dico_tool_result = json.loads(msg.content)
            msg.content = json.dumps(dico_tool_result['data'])

    msgs = [{
        "role": msg.role,
        "content": msg.content[0:50] + ' ... ' + msg.content[-50:],
        "name": msg.name,
        "tool_calls": msg.tool_calls,
        "tool_calls_id": msg.tool_call_id
    } for msg in msgs]
    # print(f'Messages:', msg)
    print(f"Messages:")
    for msg in msgs:
        print(msg)
        print('---')
    print(f"Tools available: {bool(request.tools)}")
    print('=========================')
    print('=========================')
    print('=========================')

    use_tool = await should_use_tool(request.messages, request.tools)
    print(f"Decision to use tool for msg {user_msgs[-1].content}: {use_tool}")


    request_id = f"req_{uuid.uuid4()}"

    if use_tool:
        print("Using tool for conversation")
        return StreamingResponse(
            generate_tool_stream(request_id, request.messages, request.tools),
            media_type="text/event-stream"
        )
    else:
        prompt = format_messages(request.messages, model_type, use_tool)
        print(
            f"Using normal conversation for prompt: {prompt[:200] + ' ... ' + prompt[len(prompt) // 2:len(prompt) // 2 + 100] + ' ... ' + prompt[-2000:]}")

        sampling_params = {
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
            "top_p": 0.9,
            "presence_penalty": 0.6,
            "frequency_penalty": 0.3,
            "stop": ["</s>", "[INST]"]
        }
        print('sampling_params', sampling_params)

        return StreamingResponse(
            generate_normal_stream(prompt, request_id, sampling_params),
            media_type="text/event-stream"
        )


class ChatCompletionRequestSync(BaseModel):
    model: str
    messages: List[Message]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 2048
    tools: Optional[List[Tool]] = None
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None


async def generate_normal_sync(prompt: str, sampling_params: Dict[str, Any]):
    """Generate a normal synchronous response."""
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{LLM_SERVICE_URL.replace('generate', 'generate_sync')}",
                json={
                    "prompt": prompt,
                    "sampling_params": sampling_params
                }
            )

            if response.status_code != 200:
                raise HTTPException(status_code=response.status_code, detail="LLM service error")

            result = response.json()
            return result["generated_text"]

    except Exception as e:
        print(f"Error in sync generation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def generate_tool_sync(messages: List[Message], tools: List[Tool]):
    """Generate a synchronous response for tool usage."""
    request_id = f"req_{int(time.time())}"
    tool_name, tool_query = await generate_tool_query(request_id, messages, tools)
    tool_call_id = f"call_{request_id}_{int(time.time())}"

    return {
        "id": f"chatcmpl-{request_id}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": "local-model",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "tool_calls": [{
                    "id": tool_call_id,
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "arguments": json.dumps({"question": tool_query}, ensure_ascii=False)
                    }
                }]
            },
            "finish_reason": "tool_calls"
        }]
    }


@app.post("/v1/chat/completions/sync")
async def chat_completion_sync(request: ChatCompletionRequestSync):
    """Non-streaming version of chat completion"""
    print("\n=== Starting new synchronous chat completion request ===")

    # Process tool results in messages
    msgs = request.messages
    for msg in msgs:
        if msg.role == 'tool':
            dico_tool_result = json.loads(msg.content)
            msg.content = json.dumps(dico_tool_result['data'])

    # Log messages
    msgs_log = [{
        "role": msg.role,
        "content": msg.content[0:50] + ' ... ' + msg.content[-50:],
        "name": msg.name,
        "tool_calls": msg.tool_calls,
        "tool_calls_id": msg.tool_call_id
    } for msg in msgs]
    print(f"Messages:")
    for msg in msgs_log:
        print(msg)
        print('---')
    print(f"Tools available: {bool(request.tools)}")

    # use_tool = await should_use_tool(request.messages, request.tools)
    # print(f"Decision to use tool: {use_tool}")

    # if use_tool and request.tools:
    #     print("Using tool for conversation")
    #     return await generate_tool_sync(request.messages, request.tools)
    # else:
    prompt = format_messages(request.messages, model_type, use_tool)
    print(f"Using normal conversation with prompt: {prompt[:200]}...")

    sampling_params = {
        "temperature": request.temperature,
        "max_tokens": request.max_tokens,
        "top_p": 0.9,
        "presence_penalty": 0.6,
        "frequency_penalty": 0.3,
        "stop": ["</s>", "[INST]"]
    }

    generated_text = await generate_normal_sync(prompt, sampling_params)

    return {
        "id": f"chatcmpl-{int(time.time())}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": "local-model",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": generated_text
            },
            "finish_reason": "stop"
        }]
    }

if __name__ == "__main__":
    uvicorn.run("formatting_service:app", host="0.0.0.0", port=8001, reload=True)