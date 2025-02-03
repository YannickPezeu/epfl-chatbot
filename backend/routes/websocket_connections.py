import os
import time
import traceback
from uuid import uuid4

import httpx
from fastapi import APIRouter, Cookie, HTTPException, WebSocketDisconnect
from starlette.websockets import WebSocketState, WebSocket

from typing import Dict
from myUtils.connect_acad import initialize_all_connection
from createAgent import createAgent
from searchEngine.search_engines import search_engine
from myUtils.ask_chatGPT import ask_chatGPT
import json
import struct
from myUtils.read_pdf_online import read_pdf
from langchain.schema.messages import AIMessage, HumanMessage

import tempfile

from routes.auth import get_username_from_session_token, get_openai_key
from myUtils.handle_openai_errors import handle_openai_errors
from typing import List, Union, Tuple
agent_sessions = {}

router = APIRouter(
    prefix="/ws",
    tags=["ws"]
)




def add_message_to_conversation(conversation_id, author_type, content):
    try:
        conn, cursor = initialize_all_connection()
        cursor.execute('''INSERT INTO messages (conversation_id, author_type, content) VALUES (%s, %s, %s)''', (conversation_id, author_type, content))
        conn.commit()
        conn.close()
    except Exception as e:
        print('error:', e)
        return None

class AgentSession:
    '''AgenSession object is used to access the details of the chat agent and the chat history'''
    def __init__(self, username, agent, model_name, embedding_model, library, n_documents_searched_no_llm, n_documents_searched, mistral_key=None, openai_key=None, groq_key=None, interaction_type='chat', rerank=False, special_prompt=None, conversation_id=None):
        self.username = username
        self.agent = agent
        self.model_name = model_name
        self.chat_history = []
        self.embedding_model = embedding_model
        self.library = library
        self.n_documents_searched_no_llm = n_documents_searched_no_llm
        self.n_documents_searched = n_documents_searched
        self.mistral_key = mistral_key
        self.openai_key = openai_key
        self.groq_key = groq_key
        self.interaction_type = interaction_type
        self.rerank = rerank
        self.special_prompt = special_prompt
        self.conversation_id = conversation_id


def get_response_no_model(user_input, session):
    # print('session', session)
    # print('user_input', user_input, type(user_input))
    # user_input = json.loads(user_input)
    user_input = user_input['user_input']
    # print('user_input', user_input)
    response = search_engine(
        username=session.username,
        library=session.library,
        text=user_input,
        n_results=session.n_documents_searched_no_llm,
        model_name=session.embedding_model,
        mistral_key=session.mistral_key,
        openai_key=session.openai_key,
        rerank=session.rerank
    )
    sources_formated = [
        {
            'title': source['title'],
            'page_number': source['page_number'],
            'document_index': source['document_index'],
            'pdf_id': source['pdf_id'],
            'url': source['url']
        } for source in response
    ]
    response = {'type': 'No_Model', 'sources': sources_formated}

    return response

def classify_input(user_input, agent_session):
    try:
        library = agent_session.library
        conn, cursor = initialize_all_connection()
        cursor.execute(
            "SELECT library_summary FROM user_libraries WHERE (username=%s OR username='all_users') AND library_name=%s",
            (agent_session.username, library))
        library_summary = cursor.fetchone()[0]

        response = ask_chatGPT(
            prompt=f"""Classify the user input in three classes:
            user_input: {user_input}
            library_title: {library}
            library_summary: {library_summary}
            """,
            system_instruction=f"""you receive a user_input, a library title and a library summary.
            Classify the user input in three classes:
            class 0: the user input is a question related to the library
            class 1: the question is on a completely different subject
            class 2: the question represents a danger and should be handled as an emergency by a human being 

            your answer is a json object with the following structure:
            {{
                "class": <0,1,2>
            }}
            """,
            openai_key=agent_session.openai_key

        )

        return response.choices[0].message.content

    except Exception as e:
        return {"class": 0}

UPLOADED_FILES = {}
from myUtils.redisStateManager import RedisStateManager

redis_state_manager = RedisStateManager()

async def process_file(file_data, filename, ws_connection_id):
    # Create a temporary file to store the uploaded data
    # print('processing file:', filename)
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(filename)[1]) as temp_file:
        temp_file.write(file_data)
        temp_file_path = temp_file.name

    try:
        if filename.lower().endswith('.pdf'):
            # If it's a PDF, use read_pdf to extract the text
            pages = read_pdf(temp_file_path)
            if pages:
                # Combine all page contents into a single string
                extracted_text = "\n\n".join([f"Page {page.metadata['page']}:\n{page.page_content}" for page in pages])
                redis_state_manager.handle_uploaded_files(
                    ws_connection_id,
                    "add",
                    {
                        'filename': filename,
                        'text': extracted_text
                    }
                )
                # print('processed pdf file:', filename)
                return {
                    "message": f"PDF {filename} processed successfully",
                    "extracted_text": extracted_text,
                    "page_count": len(pages),
                    "type": 'file_processed',
                    'filename': filename
                }
            else:
                # print(f"Failed to extract text from PDF {filename}")
                return {"error": f"Failed to extract text from PDF {filename}", 'type': 'file_upload_error', 'filename': filename}
        else:
            # For other file types, you can implement different processing logic
            # or simply save the file

            return {"message": f"File {filename} not pdf"}
    finally:
        # Clean up the temporary file
        os.unlink(temp_file_path)

class WsConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}

    async def connect(self, websocket: WebSocket, ws_connection_id: str):
        print('connecting to {}'.format(ws_connection_id))
        print('active connections', self.active_connections)
        print('number of active connections:', len(self.active_connections))
        await websocket.accept()
        if ws_connection_id in self.active_connections:
            print('waiting disconnect0')
            await self.disconnect(ws_connection_id)
        self.active_connections[ws_connection_id] = websocket

    async def disconnect(self, ws_connection_id: str):
        print('disconnecting from {}'.format(ws_connection_id))
        print('active connections', self.active_connections)
        if ws_connection_id in self.active_connections:
            try:
                print('closing connection0')
                await self.active_connections[ws_connection_id].close()
            except RuntimeError as e:
                if "websocket.close" not in str(e).lower():
                    raise
            finally:
                del self.active_connections[ws_connection_id]

    async def send_message(self, message: dict, ws_connection_id: str):
        if ws_connection_id in self.active_connections:
            print('waiting sending message to:', ws_connection_id)
            await self.active_connections[ws_connection_id].send_json(message)

manager = WsConnectionManager()

def convert_db_messages_to_langchain_messages(db_messages: List[Tuple]) -> List[Union[HumanMessage, AIMessage]]:
    """Convert database messages to LangChain message format"""
    messages = []
    for author_type, content, _ in db_messages:
        if author_type == "human":
            messages.append(HumanMessage(content=content))
        elif author_type == "ai":
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

@router.websocket("/{ws_connection_id}")
async def websocket_endpoint(websocket: WebSocket, ws_connection_id: str):
    try:
        print('ws_connection_id', ws_connection_id)
        print('waiting connect')
        await manager.connect(websocket, ws_connection_id)

        agent_session = agent_sessions.get(ws_connection_id)
        conversation_id = agent_session.conversation_id

        if not agent_session:
            print(f"Session not found for connection {ws_connection_id}")
            print('waiting send message session not found')
            await manager.send_message({"error": "Session not found"}, ws_connection_id)
            print('waiting disconnect session not found')
            await manager.disconnect(ws_connection_id)
            return

        agent_session.chat_history = convert_db_messages_to_langchain_messages(get_chat_history(conversation_id))

        try:
            start = time.time()
            checkpoint = time.time()
            while True:
                new_checkpoint = time.time()

                print('time elapsed:', new_checkpoint - checkpoint, ws_connection_id)
                checkpoint = new_checkpoint

                if time.time() - start > 60:
                    print('timeout')
                    break
                if websocket.application_state == WebSocketState.DISCONNECTED:
                    print(f"Client disconnected from session {ws_connection_id}")
                    break

                print('waiting for message::', ws_connection_id)
                message = await websocket.receive()

                print('message::', ws_connection_id, message)

                if message['type'] == 'websocket.disconnect':
                    print(f"Client disconnected from session {ws_connection_id}")
                    break


                # Check if it's a text message or binary (file)
                if message['type'] == 'websocket.receive':
                    if 'bytes' in message:
                        data = message['bytes']
                        # Extract header length (first 4 bytes)
                        header_length = struct.unpack('!I', data[:4])[0]

                        # Extract and parse header
                        header_json = data[4:4 + header_length].decode('utf-8')
                        try:
                            header = json.loads(header_json)
                            # print('Received file header:', header)
                        except json.JSONDecodeError:
                            # print("Error decoding JSON header:", header_json)
                            continue

                        if header['type'] == 'file':
                            filename = header['filename']
                            file_size = header['size']

                            # Extract file content
                            file_data = data[4 + header_length:]

                            print('wait process file')
                            response = await process_file(file_data, filename, ws_connection_id)

                            print('wait send message file processed')
                            await manager.send_message(response, ws_connection_id)
                        else:
                            pass
                            # print("Unexpected message type:", header['type'])

                    if 'text' in message:
                        # Handle text message (existing logic)
                        data = json.loads(message['text'])
                        # print('data', data)

                        if data.get('type') == 'remove_file':
                            filename_to_remove = data.get('filename')
                            # UPLOADED_FILES[ws_connection_id] = [f for f in UPLOADED_FILES[ws_connection_id] if f['filename'] != filename_to_remove]
                            # print('UPLOADED_FILES UPDATED', UPLOADED_FILES)
                            redis_state_manager.handle_uploaded_files(
                                ws_connection_id,
                                "remove",
                                filename_to_remove=filename_to_remove
                            )
                            print('file removed:', filename_to_remove)
                            await manager.send_message({'type': 'file_removed', 'filename': filename_to_remove}, ws_connection_id)

                        elif agent_session.agent == 'No_Model':
                            print('no model')
                            response = get_response_no_model(data, agent_session)
                            print('waiting send message no model')
                            await manager.send_message(response, ws_connection_id)
                        else:
                            user_input = data['user_input']
                            # print('UPLOAD_FILES', UPLOADED_FILES)
                            # if UPLOADED_FILES.get(ws_connection_id):
                            #     for f in UPLOADED_FILES[ws_connection_id]:
                            #         user_input = f['text'] + '\n' + user_input
                            #     UPLOADED_FILES[ws_connection_id] = []
                            uploaded_files = redis_state_manager.handle_uploaded_files(ws_connection_id, "get")
                            if uploaded_files:
                                for f in uploaded_files:
                                    user_input = f['text'] + '\n' + user_input
                                redis_state_manager.handle_uploaded_files(ws_connection_id, "clear")

                            interaction_type = data.get('interaction_type')
                            reload_message = data.get('reload_message')
                            # print('reload_message', reload_message)

                            if reload_message:
                                chat_history = agent_session.chat_history
                                # remove last AIMessage and last HumanMessage
                                agent_session.chat_history = chat_history[:-2]

                            if interaction_type == 'email':
                                classification = classify_input(user_input, agent_session)
                                classification = json.loads(classification)['class']
                                if int(classification) in [1, 2]:
                                    response = 'This question is out of scope' if int(
                                        classification) == 1 else 'This requires immediate human attention'
                                    await manager.send_message({
                                        'type': 'final_response',
                                        'data': {'message_content': response},
                                        'sources': [],
                                        'n_tokens_input': 0,
                                        'run_id': 'test'
                                    }, ws_connection_id)
                                    continue

                            async for chunk in agent_session.agent.astream_events(
                                    {"input": user_input, 'chat_history': agent_session.chat_history},
                                    version="v1"
                            ):
                                chain_end = False
                                # # print('mychunk', chunk)
                                if chunk['event'] == 'on_tool_end' and chunk['name'] in ['search_engine_tool']:
                                    print('chunk tool end', chunk)
                                    run_id = chunk.get('run_id')
                                    if 'run_id' not in chunk:
                                        pass
                                        # print('run_id not in chunk data:', chunk)
                                    output = json.loads(chunk['data']['output'])
                                    # print('myoutput', output)
                                    # print('myoutput keys', output.keys())
                                    sources = output['sources']
                                    # print('sources', sources)
                                    n_tokens_input = output['n_tokens_input']
                                    chunk_data ={
                                        'type': 'tool_answer',
                                        'data': {'message_content': output['data']},
                                        'sources': sources,
                                        'n_tokens_input': n_tokens_input,
                                        'run_id': run_id
                                    }
                                    # add to historic table
                                    conn, cursor = initialize_all_connection()
                                    query = "INSERT INTO historic (username, action, detail) VALUES (%s, %s, %s)"
                                    cursor.execute(query,
                                                   (agent_session.username,
                                                    'search',
                                                   json.dumps({'llm': agent_session.model_name, 'n_tokens_input': n_tokens_input})
                                                   ))
                                    conn.commit()

                                    try:
                                        print('waiting send message tool answer json')
                                        await websocket.send_json(chunk_data)
                                    except Exception as e:
                                        raise e

                                elif chunk['event'] == 'on_chain_stream' and\
                                    chunk['name'] == 'AgentExecutor' and\
                                        chunk['data'].get('chunk') and\
                                        chunk['data'].get('chunk').get('steps'):

                                    chunk_data = {'type': 'tool_input',
                                                  'data': {
                                                      'tool_name':chunk['data']['chunk']['steps'][0].action.tool,
                                                      'tool_input': chunk['data']['chunk']['steps'][0].action.tool_input['question']
                                                              },
                                                  'run_id': run_id
                                                  }
                                    try:
                                        print('waiting send message chain stream json')
                                        await websocket.send_json(chunk_data)
                                    except Exception as e:
                                        raise e
                                        # print('error', e)
                                        # print('chunk_data', chunk_data)

                                elif chunk['event'] == 'on_chat_model_stream':
                                    run_id = chunk.get('run_id')
                                    #not using structured output
                                    if chunk['data']['chunk'].content:
                                        chunk_data = {
                                            'data': {
                                                'message_content' : chunk['data']['chunk'].content
                                            },
                                            'run_id': run_id,
                                            'type': 'response_without_tool'
                                        }

                                    elif chunk['data']['chunk'].additional_kwargs.get('function_call'):
                                        if chunk['data']['chunk'].additional_kwargs['function_call']['name']:
                                            function_called = chunk['data']['chunk'].additional_kwargs['function_call']['name']
                                        if function_called != 'Response':
                                            continue
                                        chunk_data = {
                                            'type': 'final_response',
                                            'data': {'message_content': chunk['data']['chunk'].additional_kwargs['function_call']['arguments']},
                                            'run_id': run_id
                                        }
                                    else:
                                        chunk_data = {'data': ''}

                                    try:
                                        print('waiting send message chat model stream json')
                                        await manager.send_message(chunk_data, ws_connection_id)
                                    except Exception as e:
                                        raise e
                                        # print('error', e)
                                        # print('chunk_data', chunk_data)

                                elif chunk['event'] == 'on_chain_end' and chunk['name'] == 'AgentExecutor':
                                    agent_session.chat_history.extend(
                                        [
                                            HumanMessage(content=user_input),
                                            AIMessage(content=str(chunk['data']['output'].get('output'))),
                                        ]
                                    )

                                    add_message_to_conversation(agent_session.conversation_id, 'user', user_input)
                                    add_message_to_conversation(agent_session.conversation_id, 'ai_robot', str(chunk['data']['output'].get('output')))
                                    # print("------")

                else:
                    print('message type not recognized:', message)



        except WebSocketDisconnect:
            print(f"Client disconnected from session {ws_connection_id}")

        except Exception as e:
            print(f"Error in WebSocket connection: {str(e)}, traceback: {traceback.format_exc()}")
            error_message = str(e)
            if "invalid_api_key" in error_message:
                try:
                    error_data = json.loads(error_message)
                    error_message = error_data.get('error', {}).get('message', 'Invalid API key')
                except json.JSONDecodeError:
                    error_message = 'Invalid API key'
            print('waiting send message')
            await manager.send_message({
                "type": "error",
                "message": error_message
            }, ws_connection_id)
        finally:
            print('waiting disconnect')
            await manager.disconnect(ws_connection_id)
            # UPLOADED_FILES.pop(ws_connection_id, None)
            redis_state_manager.handle_uploaded_files(ws_connection_id, "clear")

            # print(f"WebSocket connection closed for session {ws_connection_id}")
    except Exception as e:
        print(f"Detailed error: {str(e)}")
        print(f"Stack trace: {traceback.format_exc()}")
        # Re-raise if needed
        raise


from openai import AuthenticationError, APIError



from uuid import uuid4

@router.post("/new_ws_connection")
@handle_openai_errors
async def create_new_ws_connection(
        session_token: str = Cookie(None),
        model_name: str ='gpt-4o',
        library: str = None,
        mistral_key: str = None,
        groq_key: str = None,
        embedding_model: str = None,
        n_documents_searched: int = None,
        n_documents_searched_no_llm: int = None,
        interaction_type: str = 'chat',
        rerank: str = 'false',
        conversation_id: str = None
):
    print('new_ws_connection testetst')
    try:
        if rerank == 'true':
            rerank = True
        else:
            rerank = False

        # print('rerank', rerank)
        username = get_username_from_session_token(session_token)
        openai_key, openai_key_status = get_openai_key(username)

        # test openai key
        try:
            # ask_chatGPT(
            # prompt="This is a test",
            # openai_key=openai_key,
            # max_tokens=5,
            # max_retries=1
            # )
            print("ask_chatgpt successful")
        except Exception as e:
            print('ask_chatGPT error:', e)
            # Create a mock httpx Response object
            response = httpx.Response(
                status_code=401,
                headers={},
                content=b'{"error": {"message": "Invalid OpenAI API key"}}',
                request=httpx.Request('GET', 'https://api.openai.com/v1')
            )

            # update status of openai key
            conn, cursor = initialize_all_connection()
            cursor.execute(
                "UPDATE users SET openai_key_status='invalid' WHERE username=%s",
                (username,)
            )
            conn.commit()
            conn.close()

            print('updated openai key status to invalid for username', username)




            raise AuthenticationError(
                message='Invalid OpenAI API key',
                response=response,
                body={'error': {'message': 'Invalid OpenAI API key'}}
            )





        ws_connection_id = str(uuid4())
        if model_name == 'No_Model':
            # print('No_Model', 'library', library, 'embedding_model', embedding_model)
            agent_sessions[ws_connection_id] = AgentSession(
                username=username,
                agent='No_Model',
                model_name=model_name,
                embedding_model=embedding_model,
                library=library,
                mistral_key=mistral_key,
                openai_key=openai_key,
                groq_key=groq_key,
                n_documents_searched=n_documents_searched,
                n_documents_searched_no_llm=n_documents_searched_no_llm,
                interaction_type=interaction_type,
                rerank=rerank
                )

        else:
            # get special prompt
            conn, cursor = initialize_all_connection()
            cursor.execute("SELECT special_prompt FROM user_libraries WHERE username=%s AND library_name=%s", (username, library))
            special_prompt = cursor.fetchone()
            if special_prompt is not None:
                special_prompt = special_prompt[0]

            conn.close()

            agent, conversation_id = createAgent(
                username=username,
                model_name=model_name,
                n_documents_searched=n_documents_searched,
                library=library,
                openai_key=openai_key,
                mistral_key=mistral_key,
                embedding_model=embedding_model,
                groq_key=groq_key,
                interaction_type=interaction_type,
                rerank=rerank,
                special_prompt=special_prompt,
                conversation_id=conversation_id,
                use_local_llm=False
                )


            agent_sessions[ws_connection_id] = AgentSession(
                username=username,
                agent=agent,
                model_name=model_name,
                embedding_model=embedding_model,
                library=library,
                mistral_key=mistral_key,
                openai_key=openai_key,
                groq_key=groq_key,
                n_documents_searched = n_documents_searched,
                n_documents_searched_no_llm = n_documents_searched_no_llm,
                interaction_type=interaction_type,
                rerank=rerank,
                special_prompt=special_prompt,
                conversation_id=conversation_id
            )

        return {
            "ws_connection_id": ws_connection_id,
            'model_name': model_name,
            'embedding_model': embedding_model,
            'conversation_id': conversation_id
                }
    except (AuthenticationError, APIError) as e:
        print(traceback.format_exc())
        raise e
    except Exception as e:
        print(traceback.format_exc())
        error_message = str(e)
        if "openai_api_key" in error_message:
            error_message = "OpenAI API key is missing or invalid. Please check your API key settings."
        elif "mistral_api_key" in error_message:
            error_message = "Mistral API key is missing or invalid. Please check your API key settings."
        # Add more specific error messages as needed
        else:
            error_message = f"An unexpected error occurred: {error_message}"

        raise HTTPException(status_code=500, detail=error_message )