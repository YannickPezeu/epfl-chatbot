import requests
from langchain.agents.output_parsers import ToolsAgentOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.utils.function_calling import convert_to_openai_function
from langchain_groq import ChatGroq
from dotenv import load_dotenv


load_dotenv()
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = current_dir
from langchain.agents import AgentExecutor

from langchain.globals import set_debug
from myUtils.connect_acad import initialize_all_connection
set_debug(False)

import tiktoken

from searchEngine.search_engines import create_search_engine_tool
from myUtils.get_prompt import get_prompt, MEMORY_KEY

from langchain_openai import ChatOpenAI
from typing import List, Tuple, Union
lex_db_path = os.path.realpath(os.path.join(root_dir, 'data/LEXs/LEXs.db'))
hr_db_path = os.path.realpath(os.path.join(root_dir, 'data/HR/HR.db'))

db_paths = {
    'LEX': lex_db_path,
    'HR': hr_db_path
}

tiktoken_encoding = tiktoken.get_encoding("cl100k_base")

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
        cursor.execute('''SELECT author_type, content, timestamp FROM messages WHERE conversation_id=%s''', (conversation_id,))
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




from langchain.agents.format_scratchpad import format_to_tool_messages


chat_history = []
memory = []


def get_chat_history_for_agent(conversation_id):
    print('get_chat_history_for_agent:', conversation_id)
    msgs= get_chat_history(conversation_id)
    print('msgs:', msgs)
    return convert_db_messages_to_langchain_messages(msgs)

def createAgent(
        username,
        model_name='gpt-4o',
        n_documents_searched=1,
        library='LEX',
        openai_key=None,
        mistral_key=None,
        embedding_model=None,
        groq_key=None,
        interaction_type='chat',
        rerank=False,
        special_prompt=None,
        conversation_id=None
):

    if 'gpt' in model_name:
        llm = ChatOpenAI(
            model=model_name,
            temperature=0.1,
            openai_api_key=openai_key,
        )
    else:
        llm = ChatGroq(
            model=model_name,
            groq_api_key=groq_key,
        )

    tools = [
        create_search_engine_tool(
        username=username,
        library=library,
        model_name=embedding_model,
        n_results=n_documents_searched,
        mistral_key=mistral_key,
        openai_key=openai_key,
        rerank=rerank
        )
    ]
    llm_with_tools = llm.bind_tools(tools=tools)

    # add history
    if conversation_id is not None and conversation_id:
        db_messages = get_chat_history(conversation_id)
        print('db_messages:', db_messages)
        chat_messages = convert_db_messages_to_langchain_messages(db_messages)

    else:
        conversation_id = create_new_conversation(username)
        chat_messages = []

    print('chat_messages:', chat_messages)


    if library == 'no_library':
        agent = (
            RunnablePassthrough.assign(
                agent_scratchpad=lambda x: format_to_tool_messages(x["intermediate_steps"]),
                chat_history= RunnableLambda(lambda x: get_chat_history_for_agent(conversation_id))
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

    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, return_intermediate_steps=True)

    # print('time-4:', time.time()-start)
    return agent_executor, conversation_id

openai_model_names = ["gpt-4o",
"gpt-4o-mini"]
groq_model_names = [
    'llama3-8b-8192',
    'llama3-70b-8192',
    'mixtral-8x7b-32768']






if __name__ == '__main__':
    import time
    start = time.time()
    end = time.time()
    # print('time:', end-start)