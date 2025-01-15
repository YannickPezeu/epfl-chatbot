from langchain_openai.chat_models.base import ChatOpenAI
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.schema import BaseMessage, ChatGeneration, ChatResult, AIMessage
from langchain.schema.messages import AIMessageChunk
from langchain.schema.output import GenerationChunk, ChatGenerationChunk
from typing import Any, List, Optional, Dict, AsyncIterator, Union
from langchain_core.tools import BaseTool
import aiohttp
import json
import asyncio
import logging
from aiohttp import ClientTimeout, TCPConnector
from pydantic import Field

logger = logging.getLogger(__name__)


class LocalLLM(ChatOpenAI):
    """Custom LLM class for local vLLM server with improved async support and connection handling"""

    base_url: str = Field(default="http://localhost:8001")
    streaming: bool = Field(default=True)
    temperature: float = Field(default=0.7)
    max_tokens: int = Field(default=2048)
    tools: Optional[List[Dict[str, Any]]] = Field(default=None)
    tool_choice: Optional[Union[str, Dict[str, Any]]] = Field(default="auto")

    def __init__(
            self,
            base_url: str = "http://host.docker.internal:8001",
            streaming: bool = True,
            temperature: float = 0.7,
            max_tokens: int = 2048,
            **kwargs: Any
    ) -> None:
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
        self.tools = kwargs.get('tools')
        self.tool_choice = kwargs.get('tool_choice', "auto")
        logger.info(f"Initialized LocalLLM with base_url: {self.base_url}")

    def _tools_to_functions(self, tools: List[BaseTool]) -> List[Dict[str, Any]]:
        """Convert LangChain tools to Mistral function format."""
        functions = []
        for tool in tools:
            function = {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "question": {
                                "type": "string",
                                "description": "The input to the tool",
                            }
                        },
                        "required": ["question"]
                    }
                }
            }
            functions.append(function)
        return functions

    def bind_tools(
            self,
            tools: Optional[List[BaseTool]] = None,
            **kwargs: Any,
    ) -> ChatOpenAI:
        """Bind tools to the LLM."""
        if tools:
            self.tools = self._tools_to_functions(tools)
            self.tool_choice = kwargs.get('tool_choice', 'auto')
        return self

    def _create_request_params(
            self,
            messages: List[BaseMessage],
            stop: Optional[List[str]] = None,
            **kwargs: Any,
    ) -> Dict[str, Any]:
        params = {
            "model": "local-model",
            "messages": [
                {
                    "role": "system" if message.type == "system"
                    else "assistant" if message.type == "ai"
                    else "tool" if message.type == "tool"
                    else "user",
                    "content": message.content,
                    **({"tool_calls": message.additional_kwargs["tool_calls"]}
                       if message.additional_kwargs.get("tool_calls") else {}),
                    **({"name": message.additional_kwargs.get("name")}
                       if message.type == "tool" and message.additional_kwargs.get("name") else {}),
                    **({"tool_call_id": message.additional_kwargs.get("tool_call_id")}
                       if message.type == "tool" and message.additional_kwargs.get("tool_call_id") else {})
                }
                for message in messages
            ],
            "stream": kwargs.get("stream", self.streaming),
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

        if self.tools:
            params["tools"] = self.tools
            if self.tool_choice:
                params["tool_choice"] = (
                    "auto" if self.tool_choice == "auto"
                    else "none" if self.tool_choice == "none"
                    else {"type": "function", "function": {"name": self.tool_choice}}
                    if isinstance(self.tool_choice, str)
                    else {"type": "function", "function": self.tool_choice}
                )

        if stop:
            params["stop"] = stop

        return params

    async def _astream(
            self,
            messages: List[BaseMessage],
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        """Stream the response with improved error handling and connection management."""
        timeout = ClientTimeout(total=60, connect=10, sock_read=30)
        connector = TCPConnector(force_close=True)
        headers = {"Content-Type": "application/json"}
        params = self._create_request_params(messages, stop, **kwargs)

        async with aiohttp.ClientSession(
                timeout=timeout,
                connector=connector,
                headers=headers
        ) as session:
            logger.info(f"Making request to {self.base_url}/v1/chat/completions")
            logger.info(f"Request params: {json.dumps(params, indent=2)}")

            async with session.post(
                    f"{self.base_url}/v1/chat/completions",
                    json=params
            ) as response:
                async for line in response.content:
                    if not line:
                        continue

                    line = line.decode('utf-8').strip()
                    if not line or not line.startswith('data: '):
                        continue

                    if line == 'data: [DONE]':
                        return

                    try:
                        data = json.loads(line[6:])  # Remove 'data: ' prefix
                        chunk = data['choices'][0]

                        if 'tool_calls' in chunk.get('delta', {}):
                            tool_calls = chunk['delta']['tool_calls']
                            content = json.dumps({
                                "name": tool_calls[0]["function"]["name"],
                                "arguments": tool_calls[0]["function"]["arguments"]
                            })
                        else:
                            content = chunk['delta'].get('content', '')

                        if content:
                            message = AIMessageChunk(content=content)
                            chunk = ChatGenerationChunk(message=message)
                            yield chunk

                            if run_manager:
                                await run_manager.on_llm_new_token(
                                    token=content,
                                    chunk=chunk
                                )
                    except json.JSONDecodeError:
                        logger.warning(f"Failed to parse line: {line}")
                        continue
                    except Exception as e:
                        logger.error(f"Error processing chunk: {str(e)}")
                        continue

    async def _agenerate(
            self,
            messages: List[BaseMessage],
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            **kwargs: Any,
    ) -> ChatResult:
        """Generate completion with improved error handling."""
        timeout = ClientTimeout(total=60, connect=10)
        connector = TCPConnector(force_close=True)
        headers = {"Content-Type": "application/json"}
        params = self._create_request_params(messages, stop, **kwargs)

        async with aiohttp.ClientSession(
                timeout=timeout,
                connector=connector,
                headers=headers
        ) as session:
            async with session.post(
                    f"{self.base_url}/v1/chat/completions",
                    json=params
            ) as response:
                response_data = await response.json()

                choices = response_data['choices'][0]
                if 'tool_calls' in choices['message']:
                    message = AIMessage(
                        content=None,
                        additional_kwargs={
                            'tool_calls': choices['message']['tool_calls']
                        }
                    )
                else:
                    message = AIMessage(content=choices['message'].get('content', ''))

                return ChatResult(generations=[ChatGeneration(message=message)])