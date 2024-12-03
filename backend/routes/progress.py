from fastapi import APIRouter, Depends
import json
import asyncio
from sse_starlette.sse import EventSourceResponse
from myUtils.redisStateManager import RedisStateManager

redis_state_manager = RedisStateManager()

router = APIRouter(
    prefix="/progress",
    tags=["progress"]  # This will group your library endpoints in the FastAPI docs
)


def get_state_manager():
    """Dependency to get the Redis state manager"""
    return redis_state_manager

@router.get("/{task_id}")
async def progress(
    task_id: str,
    state_manager: RedisStateManager = Depends(get_state_manager)
):
    async def event_generator():
        while True:
            progress_data = state_manager.get_state(task_id)
            if progress_data:
                yield {
                    "event": "message",
                    "data": json.dumps(progress_data)
                }
            await asyncio.sleep(1)

    return EventSourceResponse(event_generator())