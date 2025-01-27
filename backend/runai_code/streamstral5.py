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
import os
from huggingface_hub import snapshot_download
from pathlib import Path

app = FastAPI()

mymodelpath = 'mistralai/Mistral-Small-Instruct-2409'
# mymodelpath = 'mistralai/Mixtral-8x22B-Instruct-v0.1'
# mymodelpath= '/data/models2/llama3.3-70B/AWQ_4bit_quantization'

# Global variables
engine = None
current_model_path = None
current_model_type = None


def get_model_type(model_path: str) -> str:
    """Determine if the model is Llama or Mistral based on the path."""
    model_path = model_path.lower()
    if "llama" in model_path:
        return "llama"
    elif "mistral" in model_path:
        return "mistral"
    else:
        # Default to auto detection
        return "auto"

current_model_type = get_model_type(mymodelpath)

def is_awq_model(model_path: str) -> bool:
    """
    Check if the model at the given path is an AWQ quantized model.
    
    Args:
        model_path (str): Path to the model directory
        
    Returns:
        bool: True if the model is AWQ quantized, False otherwise
    """
    try:
        # Check if path exists
        if not os.path.exists(model_path):
            return False
            
        # Check for AWQ indicators in the path
        if any(awq_indicator in model_path.lower() 
               for awq_indicator in ['awq', '4bit', 'quantized', 'quantization']):
            return True
            
        # Check config.json for AWQ configurations
        config_path = os.path.join(model_path, 'config.json')
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    # Check for AWQ indicators in config
                    if any(key for key in config.keys() if 'awq' in key.lower()):
                        return True
                    if config.get('quantization_config', {}).get('method', '').lower() == 'awq':
                        return True
            except json.JSONDecodeError:
                print(f"Warning: Could not parse config.json at {config_path}")
                
        # Check for AWQ model files
        model_files = os.listdir(model_path)
        return any('awq' in f.lower() for f in model_files)
        
    except Exception as e:
        print(f"Error checking for AWQ model: {str(e)}")
        return False

def get_engine_args(model_path: str, is_fallback: bool = False) -> dict:
    """
    Get the appropriate engine arguments based on model type and fallback status.
    
    Args:
        model_path (str): Path to the model
        is_fallback (bool): Whether this is a fallback configuration
        
    Returns:
        dict: Engine arguments dictionary
    """
    # Base configuration
    base_args = {
        "model": model_path,
        "trust_remote_code": True,
        "tokenizer_mode": "auto",
        "enforce_eager": True
    }
    
    # Add non-fallback specific settings
    if not is_fallback:
        base_args.update({
            "max_model_len": 2*8192,
            "tensor_parallel_size": max(1, torch.cuda.device_count()),
            "gpu_memory_utilization": 0.85,
        })
    else:
        base_args.update({
            "max_model_len": 2048,
            "tensor_parallel_size": 1,
            "gpu_memory_utilization": 0.7,
        })
    
    # Add AWQ specific settings if needed
    if is_awq_model(model_path):
        base_args.update({
            "quantization": "awq",
            "load_format": "auto",
            # "max_context_len_to_capture": 2048 if is_fallback else 4096,
        })
        
    return base_args

def create_engine():
    """Create and return the AsyncLLMEngine instance with automatic AWQ detection."""
    global engine
    if engine is not None:
        return engine
        
    try:
        # Ensure model path is valid
        local_model_path = ensure_local_model(mymodelpath)
        print(f"Using model from: {local_model_path}")
        
        # First attempt with full settings
        try:
            print("Attempting to create engine with initial settings...")
            engine_args_dict = get_engine_args(local_model_path, is_fallback=False)
            print(f"Using engine arguments: {json.dumps(engine_args_dict, indent=2)}")
            
            engine_args = AsyncEngineArgs(**engine_args_dict)
            engine = AsyncLLMEngine.from_engine_args(engine_args)
            
            if engine is None:
                raise RuntimeError("Engine creation returned None")
                
            print("Engine created successfully!")
            return engine
            
        except Exception as first_error:
            print(f"First attempt failed: {str(first_error)}")
            print("Trying fallback configuration...")
            
            # Fallback attempt with minimal settings
            engine_args_dict = get_engine_args(local_model_path, is_fallback=True)
            print(f"Using fallback engine arguments: {json.dumps(engine_args_dict, indent=2)}")
            
            engine_args = AsyncEngineArgs(**engine_args_dict)
            engine = AsyncLLMEngine.from_engine_args(engine_args)
            
            if engine is None:
                raise RuntimeError("Fallback engine creation returned None")
                
            print("Engine created successfully with fallback configuration!")
            return engine
            
    except Exception as e:
        print(f"Fatal error in create_engine: {str(e)}")
        traceback.print_exc()
        raise

def ensure_local_model(model_path: str) -> str:
    """
    Ensures the model is available locally. If it's a Hugging Face path,
    downloads it to a corresponding local directory.
    """
    try:
        # Check if it's a local path
        if os.path.exists(model_path):
            # Verify it's a directory containing model files
            if os.path.isdir(model_path) and any(file.endswith(('.bin', '.safetensors', '.model')) 
                                               for file in os.listdir(model_path)):
                return model_path
            else:
                print(f"Warning: {model_path} exists but might not be a valid model directory")
        
        # If it's a Hugging Face URL, convert it to model ID
        if model_path.startswith(("http://huggingface.co/", "https://huggingface.co/")):
            model_path = model_path.replace("https://huggingface.co/", "")
            model_path = model_path.replace("http://huggingface.co/", "")
        
        # Clean up the model path
        model_path = model_path.strip("/")
        
        # If it's a Hugging Face model ID (e.g., "mistralai/Mistral-7B-v0.1")
        if "/" in model_path and not model_path.startswith(("/", ".")):
            # Create a base directory for downloaded models
            base_dir = Path("/data/models")
            model_name = model_path.split("/")[-1]
            local_path = base_dir / model_name
            
            # Download if not already present or incomplete
            if not local_path.exists() or not any(file.endswith(('.bin', '.safetensors', '.model')) 
                                                for file in os.listdir(local_path) if os.path.exists(local_path)):
                print(f"Downloading model from Hugging Face: {model_path}")
                try:
                    local_path = snapshot_download(
                        repo_id=model_path,
                        local_dir=str(local_path),
                        local_dir_use_symlinks=False,
                        # ignore_regex=r'.+\.h5|.+\.ot|.+\.msgpack',  # Ignore unnecessary files
                        resume_download=True,  # Resume interrupted downloads
                        # max_retries=3
                    )
                    print(f"Model downloaded successfully to {local_path}")
                except Exception as e:
                    print(f"Error during download: {str(e)}")
                    if os.path.exists(local_path):
                        print(f"Cleaning up failed download directory: {local_path}")
                        import shutil
                        shutil.rmtree(local_path, ignore_errors=True)
                    raise
            
            return str(local_path)
        
        raise ValueError(f"Model path '{model_path}' is neither a valid local path nor a Hugging Face model ID")
    
    except Exception as e:
        print(f"Error in ensure_local_model: {str(e)}")
        traceback.print_exc()
        raise



# def create_engine():
#     """Create and return the AsyncLLMEngine instance with better error handling."""
#     global engine
#     if engine is not None:
#         return engine
        
#     try:
#         # Ensure model path is valid
#         local_model_path = ensure_local_model(mymodelpath)
#         print(f"Using model from: {local_model_path}")
        
#         # Basic engine arguments - keeping it simple first
#         engine_args_dict = {
#             "model": local_model_path,
#             "trust_remote_code": True,
#             "tokenizer_mode": "auto",
#             "max_model_len": 2*8192,
#             "tensor_parallel_size": max(1, torch.cuda.device_count()),
#             "gpu_memory_utilization": 0.85,
#             "enforce_eager": True
#         }
        
#         try:
#             print("Attempting to create engine with initial settings...")
#             engine_args = AsyncEngineArgs(**engine_args_dict)
#             engine = AsyncLLMEngine.from_engine_args(engine_args)
#             if engine is None:
#                 raise RuntimeError("Engine creation returned None")
#             print("Engine created successfully!")
#             return engine
            
#         except Exception as first_error:
#             print(f"First attempt failed: {str(first_error)}")
#             print("Trying fallback configuration...")
            
#             # Fallback with minimal settings
#             fallback_args = AsyncEngineArgs(
#                 model=local_model_path,
#                 trust_remote_code=True,
#                 tokenizer_mode="auto",
#                 max_model_len=2048,
#                 tensor_parallel_size=1,
#                 gpu_memory_utilization=0.7,
#                 enforce_eager=True
#             )
            
#             engine = AsyncLLMEngine.from_engine_args(fallback_args)
#             if engine is None:
#                 raise RuntimeError("Fallback engine creation returned None")
#             print("Engine created successfully with fallback configuration!")
#             return engine
            
#     except Exception as e:
#         print(f"Fatal error in create_engine: {str(e)}")
#         traceback.print_exc()
#         raise  # Re-raise the exception instead of returning None


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

    conversation = ''

    for i, msg in enumerate(messages):
        conversation += msg.role + ':' + msg.content + '\n'

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

    decision_prompt = f"<s>[INST]Tu es en charge de décider si l'on va utiliser un outil pour répondre à la question de l'utilisateur, si tu as un doute, utilise l'outil, il vaut mieux l'utiliser trop que pas assez. Mais si c'est clair qu'il n'y a pas besoin, par exemple si l'utilisateur dit juste bonjour, ne l'utilise pas. tu répond oui pour utiliser l'outil et non pour ne pas l'utiliser. Tu as accès aux outils suivants:\n{tools_text}\n\nPour la question: \"{user_question}\"\n dans le contexte {conversation}\n\nRéponds seulement par Oui ou Non: As-tu besoin d'utiliser un outil pour répondre à cette question ? [/INST]"

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

def format_messages(messages: List[Message], model_type: str) -> str:
    """Format messages based on model type."""
    print('model_type', model_type)
    for msg in messages:
        print("msg.role", msg.role)
        print("msg.name", msg.name)
        print("msg.content[:100]", msg.content[:100])

    if model_type == "mistral":
        # Mistral formatting
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
                if not any(m.role in ["user", "human"] for m in messages[messages.index(message)+1:]):
                    prompt += "</s>"
                    conversation_start = True
            elif message.role == "system":
                # For Mistral, system message should be included in first [INST] block
                if len(prompt) == 0:
                    prompt = f"<s>[INST] {message.content}\n"
            elif message.role == "tool":
                # Include tool response within the current instruction block
                prompt += f"\nTool {message.name} returned: {message.content}\n"
                
        # Ensure the prompt ends properly
        if not prompt.endswith("</s>"):
            prompt += "</s>"
                
        return prompt

    else:
        # Llama formatting
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


async def generate_normal_stream(prompt: str, request_id: str, sampling_params: SamplingParams):
    """Generate a normal streaming response with better error handling."""
    print("Starting normal response generation")
    print(f"Normal prompt: {prompt}")

    try:
        engine = create_engine()
        if engine is None:
            raise RuntimeError("Failed to create engine")
            
        results_generator = engine.generate(prompt, sampling_params, request_id)
        if results_generator is None:
            raise RuntimeError("Engine.generate returned None")
            
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
    
    except:
        print('error in generate normal stream')

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

        # [Previous code remains the same until the generate_normal_stream function]
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

    conversation = ''

    for i, msg in enumerate(user_messages):
        conversation += msg.role + ':' + msg.content + '\n'

    # Get the last user message
    last_user_message = next((msg.content for msg in reversed(user_messages) if msg.role == "user"), "")

    # Create a prompt that asks the model to select a tool and generate a query
    tools_description = "\n".join([
        f"- {tool.function['name']}: {tool.function['description']}"
        for tool in tools
    ])

    tool_names = [tool.function['name'] for tool in tools]
    tool_names_str = ", ".join(tool_names)

    prompt = f"""<s>[INST] you have access to the following tools
{tools_description}

For the user question: "{last_user_message}" in the context "{conversation}"

1. please choose the most appropriate tool {tool_names_str}
2. Create a query for the tool

answer in this exact format:
TOOL: <tool_name>
QUERY: <your_query>
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

    # Get model type for proper formatting
    engine = create_engine()
    use_tool = await should_use_tool(request.messages, request.tools)
    print(f"Decision to use tool: {use_tool}")
    print(f"Using model type: {current_model_type}")

    request_id = f"req_{int(time.time())}"

    if use_tool:
        print("Using tool for conversation")
        return StreamingResponse(
            generate_tool_stream(request_id, request.messages, request.tools),
            media_type="text/event-stream"
        )
    else:
        # Regular conversation without tools
        prompt = format_messages(request.messages, current_model_type)
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