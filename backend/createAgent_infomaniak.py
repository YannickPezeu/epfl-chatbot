import requests
from langchain.agents.output_parsers import ToolsAgentOutputParser
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.utils.function_calling import convert_to_openai_function
from langchain_groq import ChatGroq
from langchain_openai import AzureChatOpenAI, ChatOpenAI  # Changed to Azure-specific import
from dotenv import load_dotenv
import os
import tiktoken
from typing import List, Tuple, Union

load_dotenv()
infomaniak_token = os.getenv('INFOMANIAK_TOKEN2')

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = current_dir

from langchain.agents import AgentExecutor
from langchain.globals import set_debug
from myUtils.connect_acad2 import initialize_all_connection
print('test0')
from searchEngine.search_engines import create_search_engine_tool
print('test1')
from myUtils.get_prompt import get_prompt, MEMORY_KEY
print('test2')
from langchain.agents.format_scratchpad import format_to_tool_messages

set_debug(False)

# Database paths setup
lex_db_path = os.path.realpath(os.path.join(root_dir, 'data/LEXs/LEXs.db'))
hr_db_path = os.path.realpath(os.path.join(root_dir, 'data/HR/HR.db'))

db_paths = {
    'LEX': lex_db_path,
    'HR': hr_db_path
}
print('test3')

tiktoken_encoding = tiktoken.get_encoding("cl100k_base")

from typing import Any, List, Mapping, Optional
from langchain.llms.base import LLM
from langchain.callbacks.manager import CallbackManagerForLLMRun
import requests

from langchain.schema import AIMessage, Generation
from typing import Any, List, Mapping, Optional, Iterator
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM
import requests

from langchain.schema import  AIMessage, Generation
from typing import Any, List, Mapping, Optional
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM
import requests


from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.outputs import LLMResult, ChatGeneration, ChatResult
from typing import Any, List, Mapping, Optional, Dict
from langchain.callbacks.manager import CallbackManagerForLLMRun
import requests

class InfomaniakLLM(BaseChatModel):
    """Custom Chat Model wrapper for Infomaniak API."""

    infomaniak_token: str
    model: str = "mixtral"
    functions: Optional[List[dict]] = None

    @property
    def _llm_type(self) -> str:
        return "infomaniak"

    def _generate(
            self,
            messages: List[BaseMessage],
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            **kwargs: Any,
    ) -> ChatResult:
        """Generate a response using the Infomaniak API."""

        # Convert messages to Infomaniak format
        infomaniak_messages = []
        for message in messages:
            if isinstance(message, HumanMessage):
                infomaniak_messages.append({"role": "user", "content": message.content})
            elif isinstance(message, AIMessage):
                infomaniak_messages.append({"role": "assistant", "content": message.content})

        headers = {
            "Authorization": f"Bearer {self.infomaniak_token}",
            "Content-Type": "application/json",
        }

        data = {
            "messages": infomaniak_messages,
            "model": self.model
        }

        # Add functions if they exist
        if self.functions:
            data["functions"] = self.functions
            data["function_call"] = "auto"

        try:
            response = requests.post(
                "https://api.infomaniak.com/1/ai/101819/openai/chat/completions",
                headers=headers,
                json=data
            )

            if response.status_code != 200:
                raise ValueError(f"API call failed with status code: {response.status_code}")

            response_data = response.json()
            message = response_data["choices"][0]["message"]

            # Create generation based on response type
            if "function_call" in message:
                generation = ChatGeneration(
                    message=AIMessage(
                        content="",
                        additional_kwargs={
                            "function_call": {
                                "name": message["function_call"]["name"],
                                "arguments": message["function_call"]["arguments"]
                            }
                        }
                    )
                )
            else:
                generation = ChatGeneration(
                    message=AIMessage(content=message["content"])
                )

            # Return ChatResult with the generation
            return ChatResult(
                generations=[generation],
                llm_output={"model_name": self.model}
            )

        except Exception as e:
            print(f"Error in API call: {str(e)}")
            raise

    def bind_tools(self, tools: List[Any]) -> BaseChatModel:
        """Bind tools to the model."""
        functions = [convert_to_openai_function(t) for t in tools]
        return self.__class__(
            infomaniak_token=self.infomaniak_token,
            model=self.model,
            functions=functions
        )

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {"model": self.model}

    def _combine_llm_outputs(self, llm_outputs: List[Optional[dict]]) -> dict:
        """Combine multiple LLM outputs."""
        return {}

    def get_num_tokens(self, text: str) -> int:
        """Get the number of tokens in a text."""
        return len(text.split())  # Simple approximation

def convert_db_messages_to_langchain_messages(db_messages: List[Tuple]) -> List[Union[HumanMessage, AIMessage]]:
    """Convert database messages to LangChain message format"""
    messages = []
    for author_type, content, _ in db_messages:
        if author_type == "user":
            messages.append(HumanMessage(content=content))
        elif author_type == "ai_robot":
            messages.append(AIMessage(content=content))
    return messages


def get_chat_history(conversation_id):
    try:
        conn, cursor = initialize_all_connection()
        cursor.execute('''SELECT author_type, content, timestamp FROM messages WHERE conversation_id=%s''',
                       (conversation_id,))
        messages = cursor.fetchall()
        conn.close()
        return messages
    except Exception as e:
        print('error:', e)
        return []


def create_new_conversation(username):
    try:
        conn, cursor = initialize_all_connection()
        cursor.execute('''INSERT INTO conversations (username) VALUES (%s)''', (username,))
        conversation_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return conversation_id
    except Exception as e:
        print('error:', e)
        return None


def get_chat_history_for_agent(conversation_id):
    print('get_chat_history_for_agent:', conversation_id)
    msgs = get_chat_history(conversation_id)
    print('msgs:', msgs)
    return convert_db_messages_to_langchain_messages(msgs)


local_azure_key = os.getenv('AZURE_OPENAI_API_KEY')
openai_key = os.getenv('OPENAI_KEY')

def createAgent(
        username,
        model_name='mixtral',
        n_documents_searched=1,
        library='LEX',
        openai_key=None,
        azure_endpoint='https://testpezeu0.openai.azure.com/openai/deployments/gpt-4/chat/completions?api-version=2024-08-01-preview',
        mistral_key=None,
        embedding_model=None,
        groq_key=None,
        interaction_type='chat',
        rerank=False,
        special_prompt=None,
        conversation_id=None
):
    # Initialize the Infomaniak LLM
    llm = InfomaniakLLM(
        infomaniak_token=infomaniak_token,
        model=model_name
    )

    # Create tools
    tools = [
        create_search_engine_tool(
            username=username,
            library=library,
            model_name='openai',
            n_results=n_documents_searched,
            mistral_key='test',
            openai_key=openai_key,  # Updated to use Azure key
        )
    ]

    llm_with_tools = llm.bind_tools(tools=tools)

    # Handle conversation history
    if conversation_id:
        db_messages = get_chat_history(conversation_id)
        chat_messages = convert_db_messages_to_langchain_messages(db_messages)
    else:
        conversation_id = create_new_conversation(username)
        chat_messages = []

    # Create the agent based on interaction type
    if library == 'no_library':
        agent = (
                RunnablePassthrough.assign(
                    agent_scratchpad=lambda x: format_to_tool_messages(x["intermediate_steps"]),
                    chat_history=RunnableLambda(lambda x: get_chat_history_for_agent(conversation_id))
                )
                | get_prompt('no_library', special_prompt)
                | llm
                | ToolsAgentOutputParser()
        )
    elif interaction_type in ['chat', 'email']:
        agent = (
                RunnablePassthrough.assign(
                    agent_scratchpad=lambda x: format_to_tool_messages(x["intermediate_steps"]),
                    chat_history=RunnableLambda(lambda x: get_chat_history_for_agent(conversation_id))
                )
                | get_prompt(interaction_type, special_prompt)
                | llm_with_tools
                | ToolsAgentOutputParser()
        )
    else:
        raise ValueError('interaction_type must be chat or email')

    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        return_intermediate_steps=True
    )

    return agent_executor, conversation_id


# Updated model names to match Azure deployments
azure_model_names = [
    "gpt-4o",
    "gpt-4o-mini"
]

groq_model_names = [
    'llama3-8b-8192',
    'llama3-70b-8192',
    'mixtral-8x7b-32768'
]

if __name__ == '__main__':
    import time
    import traceback
    print('test4')


    def run_interactive_conversation():
        """
        Run an interactive conversation with the Infomaniak-powered agent.
        """
        load_dotenv()

        print("Initializing agent...")
        start = time.time()

        try:
            # Create the agent
            agent_executor, conv_id = createAgent(
                username="test_user",
                model_name="mixtral",
                interaction_type="chat",
                library="no_library"  # Start without the search tool for testing
            )

            print(f"\nAgent created! Time taken: {time.time() - start:.2f}s")
            print("\nStarting conversation (type 'quit' to exit)...")
            print("-" * 50)

            while True:
                try:
                    # Get user input
                    user_input = input("\nYou: ").strip()

                    # Check for exit command
                    if user_input.lower() in ['quit', 'exit', 'q']:
                        print("\nEnding conversation. Goodbye!")
                        break

                    if not user_input:
                        continue

                    # Get agent response
                    start_time = time.time()
                    response = agent_executor.invoke({
                        "input": user_input,
                        "chat_history": []  # Initialize empty chat history for first message
                    })

                    # Print response
                    print("\nAssistant:", response["output"])
                    print(f"\n(Response time: {time.time() - start_time:.2f}s)")
                    print("-" * 50)

                except KeyboardInterrupt:
                    print("\nEnding conversation. Goodbye!")
                    break
                except Exception as e:
                    print(f"\nError occurred: {str(e)}", traceback.format_exc())
                    print("Please try again.")

        except Exception as e:
            print(f"Error during initialization: {str(e)}")

    run_interactive_conversation()
