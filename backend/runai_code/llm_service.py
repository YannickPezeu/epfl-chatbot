from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import uvicorn
import json
import time
import asyncio
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm import AsyncLLMEngine, SamplingParams
import torch.multiprocessing as mp
import atexit
import traceback
import torch
import os
from huggingface_hub import snapshot_download, login
from pathlib import Path

app = FastAPI()

# Global variables
engine = None
model_cache_dir = Path("/data/model_cache")
model_cache_file = model_cache_dir / "model_cache.mmap"

# Ensure cache directory exists
model_cache_dir.mkdir(parents=True, exist_ok=True)


class GenerateRequest(BaseModel):
    prompt: str
    sampling_params: Dict[str, Any]
    request_id: str


def is_awq_model(model_path: str) -> bool:
    try:
        if not os.path.exists(model_path):
            return False

        if any(awq_indicator in model_path.lower()
               for awq_indicator in ['awq', '4bit', 'quantized', 'quantization']):
            return True

        config_path = os.path.join(model_path, 'config.json')
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    if any(key for key in config.keys() if 'awq' in key.lower()):
                        return True
                    if config.get('quantization_config', {}).get('method', '').lower() == 'awq':
                        return True
            except json.JSONDecodeError:
                print(f"Warning: Could not parse config.json at {config_path}")

        model_files = os.listdir(model_path)
        return any('awq' in f.lower() for f in model_files)

    except Exception as e:
        print(f"Error checking for AWQ model: {str(e)}")
        return False


def get_engine_args(model_path: str, is_fallback: bool = False) -> dict:
    base_args = {
        "model": model_path,
        "trust_remote_code": True,
        "tokenizer_mode": "auto",
        "enforce_eager": True,
        "gpu_memory_utilization": 0.75,
        "disable_log_stats": True,
        "max_model_len": 15000,
        "max_num_batched_tokens": 15000,
        "disable_custom_all_reduce": False,  # Changed to False
        "enable_chunked_prefill": True,      # Added this
        "use_v2_block_manager": True,        # Added this
        "swap_space": 4,                      # Added this
        # "pipeline_parallel_size": 2,        # Try adding pipeline parallelism
        "tensor_parallel_size": 2,          # Reduce tensor parallelism
    }

    if is_awq_model(model_path):
        base_args.update({
            "quantization": "awq_marlin",
            "load_format": "auto",
        })

    # Environment variables for better GPU load balancing
    os.environ["NCCL_ALGO"] = "Ring"
    os.environ["NCCL_NET_GDR_LEVEL"] = "5"
    os.environ["NCCL_IB_TIMEOUT"] = "22"
    os.environ["NCCL_MIN_NCHANNELS"] = "1"
    os.environ["TORCH_NCCL_BLOCKING_WAIT"] = "1"

    return base_args


def ensure_local_model(model_path: str) -> str:
    """Ensures the model is available locally and properly cached."""
    try:
        if os.path.exists(model_path):
            if os.path.isdir(model_path) and any(file.endswith(('.bin', '.safetensors', '.model'))
                                                 for file in os.listdir(model_path)):
                return model_path
            else:
                print(f"Warning: {model_path} exists but might not be a valid model directory")

        if model_path.startswith(("http://huggingface.co/", "https://huggingface.co/")):
            model_path = model_path.replace("https://huggingface.co/", "")
            model_path = model_path.replace("http://huggingface.co/", "")
        model_path = model_path.strip("/")

        if "/" in model_path and not model_path.startswith(("/", ".")):
            base_dir = Path("/data/models")
            model_name = model_path.split("/")[-1]
            local_path = base_dir / model_name

            if not local_path.exists() or not any(file.endswith(('.bin', '.safetensors', '.model'))
                                                  for file in os.listdir(local_path) if os.path.exists(local_path)):
                print(f"Downloading model from Hugging Face: {model_path}")
                try:
                    local_path = snapshot_download(
                        repo_id=model_path,
                        local_dir=str(local_path),
                        local_dir_use_symlinks=False,
                        resume_download=True,
                    )
                    print(f"Model downloaded successfully to {local_path}")

                    os.system(f"chmod -R 755 {local_path}")

                except Exception as e:
                    print(f"Error during download: {str(e)}")
                    if os.path.exists(local_path):
                        print(f"Cleaning up failed download directory: {local_path}")
                        import shutil
                        shutil.rmtree(local_path, ignore_errors=True)
                    raise

            # Pre-load files into page cache
            print("Pre-loading model files into system cache...")
            for root, _, files in os.walk(local_path):
                for file in files:
                    if file.endswith(('.bin', '.safetensors', '.model')):
                        filepath = os.path.join(root, file)
                        try:
                            with open(filepath, 'rb') as f:
                                while f.read(1024 * 1024):  # 1MB chunks
                                    pass
                        except Exception as e:
                            print(f"Warning: Failed to pre-cache {filepath}: {e}")

            return str(local_path)

        raise ValueError(f"Model path '{model_path}' is neither a valid local path nor a Hugging Face model ID")

    except Exception as e:
        print(f"Error in ensure_local_model: {str(e)}")
        traceback.print_exc()
        raise


def create_engine(model_path: str):
    """Create and return the AsyncLLMEngine instance with caching."""
    global engine
    if engine is not None:
        return engine

    try:
        local_model_path = ensure_local_model(model_path)
        print(f"Using model from: {local_model_path}")

        try:
            print("Attempting to create engine...")
            engine_args_dict = get_engine_args(local_model_path, is_fallback=False)
            print(f"Using engine arguments: {json.dumps(engine_args_dict, indent=2)}")

            torch.cuda.set_per_process_memory_fraction(0.95)
            torch.cuda.init()

            engine_args = AsyncEngineArgs(**engine_args_dict)
            engine = AsyncLLMEngine.from_engine_args(engine_args)

            if engine is None:
                raise RuntimeError("Engine creation returned None")

            print("Engine created successfully!")
            return engine

        except Exception as first_error:
            print(f"Engine creation failed: {str(first_error)}")
            traceback.print_exc()
            raise

    except Exception as e:
        print(f"Fatal error in create_engine: {str(e)}")
        traceback.print_exc()
        raise


async def _generate_stream(prompt: str, sampling_params: SamplingParams, request_id: str):
    """Stream the generated text token by token with performance logging."""
    try:
        print('Starting generation with prompt:', prompt[:1000], '...', prompt[-1000:])
        start_time = time.time()
        token_count = 0
        last_token_time = start_time

        output_texts = {}

        async for request_output in engine.generate(prompt, sampling_params, request_id):
            for output in request_output.outputs:
                current_time = time.time()
                token_count += 1
                if token_count == 1:
                    start_time = time.time()

                # Log performance metrics every token
                token_latency = current_time - last_token_time
                overall_tokens_per_sec = token_count / (current_time - start_time)

                output_index = output.index
                current_text = output.text

                # Get previous text for this output index
                previous_text = output_texts.get(output_index, "")

                # Calculate new text portion
                if len(current_text) > len(previous_text):
                    new_text = current_text[len(previous_text):]
                    output_texts[output_index] = current_text

                    # Remove the strip() check to preserve whitespace tokens
                    chunk = {
                        "text": new_text,
                        "request_id": request_id,
                        "finished": False
                    }
                    if token_count%20==0:

                        print(
                            f"Token {token_count}: Latency={token_latency:.3f}s, Overall tokens/sec={overall_tokens_per_sec:.2f}, this_token/s = {1/token_latency}, text={new_text}"
                        )
                    chunk_str = f"data: {json.dumps(chunk)}\n\n"
                    yield chunk_str

                last_token_time = current_time

        total_time = time.time() - start_time
        print(f"\nGeneration complete:")
        print(f"Total tokens: {token_count}")
        print(f"Total time: {total_time:.2f}s")
        print(f"Average tokens/sec: {token_count / total_time:.2f}")

        yield f"data: {json.dumps({'text': '', 'request_id': request_id, 'finished': True})}\n\n"
        yield "data: [DONE]\n\n"

    except Exception as e:
        print(f"Error in stream generation: {str(e)}")
        traceback.print_exc()
        error_chunk = {
            "error": str(e),
            "request_id": request_id,
            "finished": True
        }
        yield f"data: {json.dumps(error_chunk)}\n\n"
        yield "data: [DONE]\n\n"

async def generate_text(prompt: str, sampling_params: Dict[str, Any], request_id: str):
    try:
        print('received_prompt in generate function', prompt[:500]+ ' ... '+ prompt[-500:])

        if engine is None:
            raise HTTPException(status_code=500, detail="LLM engine not initialized")

        sampling_params_obj = SamplingParams(**sampling_params)

        response = StreamingResponse(
            _generate_stream(prompt, sampling_params_obj, request_id),
            media_type="text/event-stream"
        )
        response.headers["X-Accel-Buffering"] = "no"  # Disable buffering in nginx
        return response

    except Exception as e:
        print(f"Error in generate_text: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate")
async def generate(request: GenerateRequest):
    """Generate endpoint that returns a streaming response."""
    response = await generate_text(request.prompt, request.sampling_params, request.request_id)
    # Forward the streaming response directly
    response.headers["Content-Type"] = "text/event-stream"
    response.headers["Cache-Control"] = "no-cache"
    response.headers["Connection"] = "keep-alive"
    return response


model_path = None


@app.get("/model_type")
def get_model_type():
    """Return the type of the loaded model."""
    global model_path
    try:

        print(f"Model path: {model_path}")

        if "llama" in model_path.lower():
            return {"type": "llama"}
        elif "mistral" in model_path.lower():
            return {"type": "mistral"}
        else:
            return {"type": "auto"}

    except Exception as e:
        print(f"Error in get_model_type endpoint: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error getting model type: {str(e)}")


@app.on_event("startup")
async def startup_event():
    global model_path, engine
    # model_path = 'meta-llama/Llama-3.3-70B-Instruct'
    # model_path = '/data/models2/llama3.3-70B/AWQ_4bit_quantization'
    model_path = 'mistralai/Mistral-Small-24B-Instruct-2501'
    # model_path = 'mistralai/Mistral-Nemo-Instruct-2407'

    # Pre-initialize CUDA with optimal settings
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    # Verify CUDA availability
    assert torch.cuda.is_available(), "CUDA not available!"
    print(f"Available GPUs: {torch.cuda.device_count()}")

    # Clear cache before initialization
    torch.cuda.empty_cache()

    # Initialize engine
    engine = create_engine(model_path)


def cleanup():
    global engine
    if engine:
        del engine
        torch.cuda.empty_cache()


atexit.register(cleanup)


# Add this after your existing GenerateRequest class
class SyncGenerateRequest(BaseModel):
    prompt: str
    sampling_params: Dict[str, Any]


async def generate_text_sync(prompt: str, sampling_params: Dict[str, Any]):
    try:
        if engine is None:
            raise HTTPException(status_code=500, detail="LLM engine not initialized")

        sampling_params_obj = SamplingParams(**sampling_params)

        # Get the full response
        result = ""
        async for request_output in engine.generate(prompt, sampling_params_obj):
            # Take the last output as it contains the complete text
            result = request_output.outputs[0].text

        return {"generated_text": result}

    except Exception as e:
        print(f"Error in generate_text_sync: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate_sync")
async def generate_sync(request: SyncGenerateRequest):
    """Generate endpoint that returns a complete response."""
    try:
        response = await generate_text_sync(request.prompt, request.sampling_params)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # test if generation works

    mp.set_start_method('spawn', force=True)
    uvicorn.run("llm_service:app", host="0.0.0.0", port=8002, reload=False)