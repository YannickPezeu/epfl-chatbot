from langchain_openai.chat_models.base import ChatOpenAI
from langchain.callbacks.manager import CallbackManagerForLLMRun, AsyncCallbackManagerForLLMRun
from langchain_core.messages import (
    BaseMessage,
    AIMessage,
    HumanMessage,
    SystemMessage,
    FunctionMessage,
    ToolMessage,
    ChatMessage
)
from langchain_core.messages.base import BaseMessageChunk
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.messages import (
    AIMessageChunk,
    HumanMessageChunk,
    SystemMessageChunk,
    FunctionMessageChunk,
    ToolMessageChunk,
    ChatMessageChunk
)
from langchain.schema.output import GenerationChunk, ChatGenerationChunk
from typing import Any, List, Optional, Dict, AsyncIterator, Union, Tuple
from langchain_core.tools import BaseTool
import aiohttp
import json
import asyncio
import logging
from aiohttp import ClientTimeout, TCPConnector
from pydantic import Field

logger = logging.getLogger(__name__)


class LocalLLM(ChatOpenAI):
    """Custom LLM class for local vLLM server with improved async support and connection handling."""

    base_url: str = Field(default="http://localhost:8001")
    streaming: bool = Field(default=True)
    temperature: float = Field(default=0.7)
    max_tokens: int = Field(default=2048)

    def __init__(
            self,
            base_url: str = "http://host.docker.internal:8001",
            streaming: bool = True,
            temperature: float = 0.7,
            max_tokens: int = 2048,
            **kwargs: Any
    ) -> None:
        """Initialize LocalLLM."""
        super().__init__(
            model="local-model",
            api_key="dummy_key",
            base_url=base_url.rstrip("/"),
            streaming=streaming,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )
        self.streaming = streaming
        self.temperature = temperature
        self.max_tokens = max_tokens

    def _create_message_dicts(
            self, messages: List[BaseMessage], stop: Optional[List[str]]
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Create the message dictionaries to send to the model."""
        params = {
            "model": "local-model",
            "stream": self.streaming,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }
        if stop is not None:
            if "stop" in params:
                raise ValueError("`stop` found in both the input and default params.")
            params["stop"] = stop

        message_dicts = [self._convert_message_to_dict(m) for m in messages]
        return message_dicts, params

    def _convert_message_to_dict(self, message: BaseMessage) -> dict:
        """Convert a LangChain message to a dictionary for the API."""
        message_dict: Dict[str, Any] = {
            "content": message.content,
        }

        if (name := message.name) is not None:
            message_dict["name"] = name

        if isinstance(message, AIMessage):
            message_dict["role"] = "assistant"
            if "function_call" in message.additional_kwargs:
                message_dict["function_call"] = message.additional_kwargs["function_call"]
            if message.tool_calls or message.invalid_tool_calls:
                message_dict["tool_calls"] = [
                                                 self._convert_tool_call_to_dict(tc) for tc in message.tool_calls
                                             ] + [
                                                 self._convert_invalid_tool_call_to_dict(tc)
                                                 for tc in message.invalid_tool_calls
                                             ]
            elif "tool_calls" in message.additional_kwargs:
                message_dict["tool_calls"] = message.additional_kwargs["tool_calls"]
        elif isinstance(message, HumanMessage):
            message_dict["role"] = "user"
        elif isinstance(message, SystemMessage):
            message_dict["role"] = "system"
        elif isinstance(message, FunctionMessage):
            message_dict["role"] = "function"
        elif isinstance(message, ToolMessage):
            message_dict["role"] = "tool"
            message_dict["tool_call_id"] = message.tool_call_id
        else:
            message_dict["role"] = message.type

        return message_dict

    def _convert_tool_call_to_dict(self, tool_call: dict) -> dict:
        """Convert a tool call to the format expected by the API."""
        return {
            "type": "function",
            "id": tool_call.get("id", ""),
            "function": {
                "name": tool_call["name"],
                "arguments": json.dumps(tool_call["args"])
            }
        }

    def _convert_invalid_tool_call_to_dict(self, tool_call: dict) -> dict:
        """Convert an invalid tool call to the format expected by the API."""
        return {
            "type": "function",
            "id": tool_call.get("id", ""),
            "function": {
                "name": tool_call["name"],
                "arguments": tool_call["args"]
            }
        }

    def _convert_dict_to_message(self, raw_message: Dict[str, Any]) -> BaseMessage:
        """Convert a dictionary response to a LangChain message."""
        role = raw_message["role"]
        content = raw_message.get("content", "") or ""
        additional_kwargs: Dict = {}

        if role == "assistant":
            if function_call := raw_message.get("function_call"):
                additional_kwargs["function_call"] = dict(function_call)

            tool_calls = []
            invalid_tool_calls = []
            if raw_tool_calls := raw_message.get("tool_calls"):
                additional_kwargs["tool_calls"] = raw_tool_calls
                for raw_tool_call in raw_tool_calls:
                    try:
                        tool_calls.append(self._parse_tool_call(raw_tool_call))
                    except Exception as e:
                        invalid_tool_calls.append(
                            self._make_invalid_tool_call(raw_tool_call, str(e))
                        )

            return AIMessage(
                content=content,
                additional_kwargs=additional_kwargs,
                tool_calls=tool_calls,
                invalid_tool_calls=invalid_tool_calls,
            )
        elif role == "user":
            return HumanMessage(content=content)
        elif role == "system":
            return SystemMessage(content=content)
        elif role == "function":
            return FunctionMessage(
                content=content,
                name=raw_message["name"],
            )
        elif role == "tool":
            return ToolMessage(
                content=content,
                tool_call_id=raw_message["tool_call_id"],
            )
        else:
            return ChatMessage(content=content, role=role)

    # Fix 1: Update _parse_tool_call in LocalLLM class (paste-2.txt)
    def _parse_tool_call(self, raw_tool_call: Dict) -> dict:
        """Parse a raw tool call into the expected format."""
        function_data = raw_tool_call["function"]
        # Add tool_call_id to the output
        return {
            "id": raw_tool_call.get("id", ""),
            "type": "function",
            "name": function_data["name"],
            "args": json.loads(function_data["arguments"]),
            "tool_call_id": raw_tool_call.get("id", "")  # Add this line
        }

    def _make_invalid_tool_call(self, raw_tool_call: Dict, error: str) -> dict:
        """Create an invalid tool call record."""
        return {
            "id": raw_tool_call.get("id", ""),
            "type": "function",
            "name": raw_tool_call["function"]["name"],
            "args": raw_tool_call["function"]["arguments"],
            "error": error
        }

    async def _astream(
            self,
            messages: List[BaseMessage],
            stop: Optional[List[str]] = None,
            run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
            **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        """Stream the response with proper tool call handling."""
        message_dicts, params = self._create_message_dicts(messages, stop)
        params = {**params, **kwargs, "stream": True}

        timeout = ClientTimeout(total=60, connect=10, sock_read=30)
        connector = TCPConnector(force_close=True)
        headers = {"Content-Type": "application/json"}

        async with aiohttp.ClientSession(timeout=timeout, connector=connector, headers=headers) as session:
            async with session.post(
                    f"{self.base_url}/v1/chat/completions",
                    json={**params, "messages": message_dicts}
            ) as response:
                async for line in response.content:
                    if not line or line == b"data: [DONE]":
                        continue

                    line = line.decode('utf-8').strip()
                    if not line.startswith('data: '):
                        continue

                    try:
                        data = json.loads(line[6:])  # Remove 'data: ' prefix
                        choice = data["choices"][0]
                        delta = choice.get("delta", {})

                        if not delta:
                            continue

                        chunk = self._create_chat_generation_chunk(delta, choice)
                        if chunk is not None:
                            if run_manager:
                                await run_manager.on_llm_new_token(
                                    token=chunk.text,
                                    chunk=chunk,
                                )
                            yield chunk

                    except Exception as e:
                        logger.error(f"Error in stream processing: {str(e)}")
                        continue

    def _create_chat_generation_chunk(
            self, delta: Dict[str, Any], choice: Dict[str, Any]
    ) -> Optional[ChatGenerationChunk]:
        """Create a chat generation chunk from a delta update."""
        if not delta:
            return None

        content = delta.get("content", "")
        role = delta.get("role")
        tool_calls = delta.get("tool_calls")

        if tool_calls:
            # Handle tool calls in delta
            message = AIMessageChunk(
                content="",
                additional_kwargs={"tool_calls": tool_calls},
            )
        elif content or role:
            # Handle normal content or role updates
            message = AIMessageChunk(content=content)
        else:
            return None

        generation_info = {}
        if finish_reason := choice.get("finish_reason"):
            generation_info["finish_reason"] = finish_reason

        return ChatGenerationChunk(message=message, generation_info=generation_info or None)

    async def _agenerate(
            self,
            messages: List[BaseMessage],
            stop: Optional[List[str]] = None,
            run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
            **kwargs: Any,
    ) -> ChatResult:
        """Generate a chat response."""
        if self.streaming:
            stream_iter = self._astream(
                messages, stop=stop, run_manager=run_manager, **kwargs
            )
            return await self._astream_to_chat_result(stream_iter)

        message_dicts, params = self._create_message_dicts(messages, stop)
        params = {**params, **kwargs}

        timeout = ClientTimeout(total=60, connect=10)
        connector = TCPConnector(force_close=True)
        headers = {"Content-Type": "application/json"}

        async with aiohttp.ClientSession(timeout=timeout, connector=connector, headers=headers) as session:
            async with session.post(
                    f"{self.base_url}/v1/chat/completions",
                    json={**params, "messages": message_dicts}
            ) as response:
                response_data = await response.json()
                return self._create_chat_result(response_data)

    def _create_chat_result(self, response: Dict[str, Any]) -> ChatResult:
        """Create a ChatResult from the API response."""
        generations = []
        for choice in response["choices"]:
            message = self._convert_dict_to_message(choice["message"])
            gen = ChatGeneration(
                message=message,
                generation_info=dict(finish_reason=choice.get("finish_reason"))
            )
            generations.append(gen)

        token_usage = response.get("usage", {})
        llm_output = {
            "token_usage": token_usage,
            "model_name": "local-model",
        }

        return ChatResult(generations=generations, llm_output=llm_output)

    async def _astream_to_chat_result(
            self, stream_iter: AsyncIterator[ChatGenerationChunk]
    ) -> ChatResult:
        """Convert a stream of chat chunks into a ChatResult."""
        final_generation = None
        async for chunk in stream_iter:
            if chunk is not None:
                final_generation = chunk

        if final_generation is None:
            return ChatResult(generations=[], llm_output={})

        return ChatResult(
            generations=[
                ChatGeneration(
                    message=final_generation.message,
                    generation_info=final_generation.generation_info,
                )
            ],
            llm_output={"model_name": "local-model"},
        )