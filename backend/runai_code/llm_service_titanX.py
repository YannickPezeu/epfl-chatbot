import os

# Force older CUDA compatibility
os.environ["VLLM_CUDA_COMPAT"] = "1"
os.environ["TORCH_CUDA_ARCH_LIST"] = "5.2"  # Specific to TitanX Maxwell
os.environ["CUDA_MODULE_LOADING"] = "LAZY"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"  # Only use 4 GPUs for Mistral 7B

# Basic NCCL settings
os.environ["NCCL_DEBUG"] = "INFO"
os.environ["NCCL_IB_DISABLE"] = "1"
os.environ["NCCL_SOCKET_IFNAME"] = "eth0"

# Force older NCCL operations
os.environ["NCCL_ALGO"] = "RING"
os.environ["NCCL_PROTO"] = "LL"  # Changed to LL (Latency-Limited)
os.environ["NCCL_MAX_NRINGS"] = "1"
os.environ["NCCL_BUFFSIZE"] = "4194304"
os.environ["NCCL_NET_GDR_READ"] = "1"
os.environ["NCCL_LAUNCH_MODE"] = "PARALLEL"

# Memory settings
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512,garbage_collection_threshold:0.6"
os.environ["CUDA_FORCE_PTX_JIT"] = "1"

# Import torch and set configurations
import torch
torch._dynamo.config.suppress_errors = True
torch._dynamo.config.verbose = False
torch.backends.cuda.preferred_linalg_library = 'cusolver'
torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.enable_flash_sdp = False
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = False

from fastapi import FastAPI, HTTPException
# ... rest of your imports
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
import os
from huggingface_hub import snapshot_download, login
from pathlib import Path
import dotenv



# Rest of your imports and code...

dotenv.load_dotenv()

app = FastAPI()

# Global variables
engine = None
model_cache_dir = Path("/data/model_cache")
model_cache_file = model_cache_dir / "model_cache.mmap"

# Ensure cache directory exists
model_cache_dir.mkdir(parents=True, exist_ok=True)

hf_token = os.getenv("HUGGINGFACE_TOKEN")
login(token=hf_token)


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

import logging
import sys

def setup_logging():
    # Create a formatter that includes worker info
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(processName)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # File handler for llm.log
    file_handler = logging.FileHandler('/var/log/app/llm.log')
    file_handler.setFormatter(formatter)

    # Stream handler for stdout
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)

    # Configure root logger to capture everything
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(stream_handler)

    # Configure vllm logger specifically
    vllm_logger = logging.getLogger('vllm')
    vllm_logger.setLevel(logging.INFO)
    vllm_logger.addHandler(file_handler)
    vllm_logger.addHandler(stream_handler)

    logging.info("Logging setup completed")

# Add at the start of your llm_service.py
setup_logging()


def get_engine_args(model_path: str, is_fallback: bool = False) -> dict:
    base_args = {
        "model": model_path,
        "trust_remote_code": True,
        "tokenizer_mode": "auto",
        "enforce_eager": True,
        "gpu_memory_utilization": 0.85,  # Can be higher with 7B model
        "disable_log_stats": True,
        "max_model_len": 2048,
        "max_num_batched_tokens": 2048,
        "disable_custom_all_reduce": True,
        "enable_chunked_prefill": False,
        "use_v2_block_manager": False,
        "swap_space": 4,
        "tensor_parallel_size": 4,      # Using 4 GPUs for 7B model
        "dtype": "float16",
        "pipeline_parallel_size": 1
    }
    return base_args


def get_engine_args_bu(model_path: str, is_fallback: bool = False) -> dict:
    base_args = {
        "model": model_path,
        "trust_remote_code": True,
        "tokenizer_mode": "auto",
        "enforce_eager": True,
        "gpu_memory_utilization": 0.85,
        "disable_log_stats": True,
        "max_model_len": 15000,
        "max_num_batched_tokens": 15000,
        "disable_custom_all_reduce": False,  # Changed to False
        "enable_chunked_prefill": True,      # Added this
        "use_v2_block_manager": True,        # Added this
        "swap_space": 4,                      # Added this
        # "pipeline_parallel_size": 2,        # Try adding pipeline parallelism
        "tensor_parallel_size": 8,          # Reduce tensor parallelism
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
        if not model_path:
            raise ValueError("Model path is empty or None")

        # Convert to string if it's a Path object
        model_path = str(model_path)

        # Check if it's already a local path containing model files
        if os.path.exists(model_path):
            if os.path.isdir(model_path) and any(file.endswith(('.bin', '.safetensors', '.model'))
                                                 for file in os.listdir(model_path)):
                return model_path
            else:
                print(f"Warning: {model_path} exists but might not be a valid model directory")

        # Handle HuggingFace model IDs
        if "/" in model_path and not model_path.startswith(("/", ".")):
            base_dir = Path("/data/models")
            model_name = model_path.split("/")[-1]
            local_path = base_dir / model_name

            # Create base directory if it doesn't exist
            base_dir.mkdir(parents=True, exist_ok=True)

            if not local_path.exists():
                print(f"Downloading model from Hugging Face: {model_path}")
                local_path = snapshot_download(
                    repo_id=model_path,
                    local_dir=str(local_path),
                    local_dir_use_symlinks=False,
                    resume_download=True,
                )
                print(f"Model downloaded successfully to {local_path}")
                os.system(f"chmod -R 755 {local_path}")

            return str(local_path)

        raise ValueError(f"Invalid model path format: {model_path}")

    except Exception as e:
        print(f"Error in ensure_local_model: {str(e)}")
        traceback.print_exc()
        raise
def create_engine_bu2(model_path: str):
    logging.info("Starting engine creation...")
    try:
        local_model_path = ensure_local_model(model_path)
        logging.info(f"Using model from: {local_model_path}")

        try:
            engine_args_dict = get_engine_args(local_model_path, is_fallback=False)
            logging.info(f"Using engine arguments: {json.dumps(engine_args_dict, indent=2)}")

            logging.info("Creating AsyncLLMEngine...")
            engine_args = AsyncEngineArgs(**engine_args_dict)

            engine = AsyncLLMEngine.from_engine_args(engine_args)
            logging.info("AsyncLLMEngine created successfully!")

            if engine is None:
                raise RuntimeError("Engine creation returned None")

            return engine

        except Exception as first_error:
            logging.error(f"Engine creation failed: {str(first_error)}", exc_info=True)
            raise

    except Exception as e:
        logging.error(f"Fatal error in create_engine: {str(e)}", exc_info=True)
        raise

def create_engine_bu(model_path: str):
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
            print("About to create AsyncLLMEngine...")
            try:
                engine = AsyncLLMEngine.from_engine_args(engine_args)
                print("AsyncLLMEngine created successfully")
            except Exception as e:
                print(f"Failed to create AsyncLLMEngine: {str(e)}")
                traceback.print_exc()
                raise

            if engine is None:
                raise RuntimeError("Engine creation returned None")

            print("Engine created successfully!")
            print("Model loaded successfully!")  # Add this message

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
    print('request prompt', request.prompt[:100]+ ' ... '+  request.prompt[-100:])
    print("request.request_id", request.request_id)
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


def log_gpu_memory():
    """Log GPU memory usage"""
    for i in range(torch.cuda.device_count()):
        memory_allocated = torch.cuda.memory_allocated(i) / 1024**3
        memory_reserved = torch.cuda.memory_reserved(i) / 1024**3
        print(f"GPU {i}: Allocated: {memory_allocated:.2f}GB, Reserved: {memory_reserved:.2f}GB")

# Add periodic memory logging
async def monitor_memory():
    while True:
        log_gpu_memory()
        await asyncio.sleep(1)  # Log every 30 seconds


def ensure_clean_runtime_dir():
    """Ensure clean runtime directory for NCCL"""
    try:
        shm_path = "/dev/shm"
        # Remove any existing NCCL files
        for f in os.listdir(shm_path):
            if f.startswith("nccl-"):
                try:
                    os.remove(os.path.join(shm_path, f))
                except Exception as e:
                    print(f"Warning: Could not remove {f}: {e}")

        # Verify permissions
        os.system(f"chmod 777 {shm_path}")

    except Exception as e:
        print(f"Warning: Runtime dir cleanup failed: {e}")


import resource


def configure_system_limits():
    """Configure system limits for better performance"""
    try:
        # Remove memory locks
        resource.setrlimit(resource.RLIMIT_MEMLOCK, (-1, -1))

        # Increase number of open files limit
        resource.setrlimit(resource.RLIMIT_NOFILE, (65536, 65536))

    except Exception as e:
        print(f"Warning: Could not set system limits: {e}")

@app.on_event("startup")
async def startup_event():
    global model_path, engine
    asyncio.create_task(monitor_memory())

    try:
        # Set model path
        configure_system_limits()

        ensure_clean_runtime_dir()
        model_path = 'mistralai/Mistral-Small-24B-Instruct-2501'
        model_path = 'mistralai/Mistral-7B-Instruct-v0.2'
        print(f"Initial model path: {model_path}")

        # Pre-initialize CUDA
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

        # Verify CUDA availability
        assert torch.cuda.is_available(), "CUDA not available!"
        print(f"Available GPUs: {torch.cuda.device_count()}")

        # Create engine with verified model path
        local_model_path = ensure_local_model(model_path)
        if not os.path.exists(local_model_path):
            raise ValueError(f"Model path does not exist after download: {local_model_path}")

        print(f"Using local model path: {local_model_path}")
        engine = create_engine(local_model_path)

        if not engine:
            raise RuntimeError("Engine creation failed")

        print("Server started and ready!")

    except Exception as e:
        print(f"Startup failed: {str(e)}")
        traceback.print_exc()
        raise

def create_engine(model_path: str):
    logging.info("Starting engine creation...")
    try:
        if not model_path:
            raise ValueError("Model path cannot be None")

        local_model_path = ensure_local_model(model_path)
        logging.info(f"Using model from: {local_model_path}")

        try:
            engine_args_dict = get_engine_args(local_model_path, is_fallback=False)
            logging.info(f"Using engine arguments: {json.dumps(engine_args_dict, indent=2)}")

            logging.info("Creating AsyncLLMEngine...")
            engine_args = AsyncEngineArgs(**engine_args_dict)

            engine = AsyncLLMEngine.from_engine_args(engine_args)
            logging.info("AsyncLLMEngine created successfully!")

            if engine is None:
                raise RuntimeError("Engine creation returned None")

            return engine

        except Exception as first_error:
            logging.error(f"Engine creation failed: {str(first_error)}", exc_info=True)
            raise

    except Exception as e:
        logging.error(f"Fatal error in create_engine: {str(e)}", exc_info=True)
        raise
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


    uvicorn.run("llm_service_titanX:app", host="0.0.0.0", port=8002, reload=False)