# routes/libraries.py
from fastapi import APIRouter, Cookie, BackgroundTasks
from myUtils.connect_acad2 import initialize_all_connection, reconnect_on_failure
from routes.auth import get_username_from_session_token

from myUtils.ask_chatGPT import ask_chatGPT
from routes.auth import get_openai_key
from myUtils.handle_openai_errors import handle_openai_errors

import os

import sqlite3


from library_creation._0_create_big_chunks_from_source_docs import insert_big_chunks_into_db
from library_creation._1_create_small_chunks_from_big_chunks import insert_small_chunks_into_db
from library_creation._2_embedd_small_chunks import dtypes_for_models, insert_embeddings_models_into_db, embedd_all_small_chunks
from library_creation._3_create_faiss_index import create_faiss_index


from fastapi import File, UploadFile, Cookie, HTTPException, Request
from typing import List
import tempfile
from fastapi import Form

router = APIRouter(
    prefix="/libraries",
    tags=["libraries"]  # This will group your library endpoints in the FastAPI docs
)


@router.delete("/{library_name}")  # RESTful way to delete a resource
def delete_library(library_name: str, session_token: str = Cookie(None)):
    username = get_username_from_session_token(session_token)

    # Database deletion
    conn, cursor = initialize_all_connection()
    cursor.execute("DELETE FROM user_libraries WHERE library_name=%s AND username=%s", (library_name, username))
    # cursor.execute("DELETE FROM source_docs WHERE library=%s AND username=%s", (library_name, username))
    cursor.execute("DELETE FROM big_chunks WHERE library=%s AND username=%s", (library_name, username))
    cursor.execute("DELETE FROM small_chunks WHERE library=%s AND username=%s", (library_name, username))
    cursor.execute("DELETE FROM embeddings WHERE library=%s AND username=%s", (library_name, username))

    # Query model_ids before deleting metadata
    cursor.execute(
        "SELECT DISTINCT model_id FROM faiss_index_metadata WHERE library=%s AND (username=%s OR username='all_users')",
        (library_name, username))
    model_ids = [row[0] for row in cursor.fetchall()]

    cursor.execute("DELETE FROM faiss_indexes WHERE library=%s AND username=%s", (library_name, username))
    cursor.execute("DELETE FROM faiss_index_metadata WHERE library=%s AND username=%s", (library_name, username))
    cursor.execute("DELETE FROM faiss_index_parts WHERE library=%s AND username=%s", (library_name, username))
    conn.commit()
    conn.close()

    # Delete from Redis - Using the same approach as your reference code
    # Initialize Redis managers for different data types
    string_redis = RedisStateManager(decode_responses=True)  # For JSON/string data
    binary_redis = RedisStateManager(decode_responses=False)  # For binary data

    # Delete FAISS indexes from Redis
    for model_id in model_ids:
        # Check both user-specific and all_users indexes
        for index_username in [username, "all_users"]:
            # Create the keys
            index_key = f"faiss:index:{model_id}:{library_name}:{index_username}"
            embedding_key = f"faiss:embeddings:{model_id}:{library_name}:{index_username}"

            # Check if keys exist and delete if they do
            index_exists = binary_redis.redis_client.exists(index_key)
            embedding_exists = string_redis.redis_client.exists(embedding_key)

            if index_exists:
                binary_redis.redis_client.delete(index_key)
                print(f"Deleted index key: {index_key}")

            if embedding_exists:
                string_redis.redis_client.delete(embedding_key)
                print(f"Deleted embedding key: {embedding_key}")

    # Handle uploaded files and agent sessions
    redis_manager = RedisStateManager()

    # Get all active connections for this user
    active_connections = redis_manager.get_state(f"user_connections:{username}") or []

    # For each active connection, clean up any library-related data
    for connection_id in active_connections:
        # Clear any uploaded files related to this library
        uploaded_files = redis_manager.get_state(f"uploaded_files:{connection_id}") or []
        updated_files = [f for f in uploaded_files if f.get('library') != library_name]

        if len(updated_files) != len(uploaded_files):
            redis_manager.set_state(f"uploaded_files:{connection_id}", updated_files)

        # Update any agent sessions using this library
        agent_session_key = f"agent_session:{connection_id}"
        session_data = redis_manager.get_state(agent_session_key)

        if session_data and session_data.get('library') == library_name:
            redis_manager.delete_state(agent_session_key)

    # Delete any library-specific Redis keys
    redis_manager.delete_state(f"library_metadata:{username}:{library_name}")

    return {"message": f"Library '{library_name}' deleted successfully"}



@reconnect_on_failure
def insert_user_library(username, library_name, special_prompt=None, cursor=None):

    if cursor is None:
        conn, cursor = initialize_all_connection()

    #retrieve source_docs content from the database and create a summary
    cursor.execute("SELECT page_content FROM big_chunks WHERE library=%s AND page_number=0", (library_name,))
    source_docs_content = cursor.fetchall()
    text_to_summarize = [pdf_content[0] for pdf_content in source_docs_content]
    text_to_summarize = '\n\n'.join(text_to_summarize)
    prompt = f"""
    Summarize the content of the database {library_name}, 
    which contains the following documents: {text_to_summarize}
    the summary should be approximately 300 words
    don't add any comments or personal opinions
    """
    try:
        summary = ask_chatGPT(
            prompt=prompt,
            model='gpt-4o-mini',
            openai_key=get_openai_key(username)[0],
            # openai_key='test',
        )
        # choices = summary.choices
        # if len(choices) > 0:
        #     summary = summary.choices[0].message.content
        # else:
        #     summary = 'no summary'
        summary = summary.choices[0].message.content
    except Exception as e:
        # print('error', e)
        summary = 'no summary'

    # Insert the new table for the user
    cursor.execute("INSERT IGNORE INTO user_libraries (username, library_name, library_summary, special_prompt) VALUES (%s, %s, %s, %s)", (username, library_name, summary, special_prompt))



@router.post("/create")
@handle_openai_errors
@reconnect_on_failure
async def create_library(
    request: Request,
    files: List[UploadFile] = File(...),
    library_name: str = Form(...),
    model_name: str = Form(default='openai'),
    special_prompt: str = Form(default=''),
    session_token: str = Cookie(None),
    background_tasks: BackgroundTasks = None
):


    username = get_username_from_session_token(session_token)
    task_id = str(username) + '_' + library_name

    # Save files to temporary directory
    temp_file_paths = []
    for file in files:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
            content = await file.read()
            tmp.write(content)
            temp_file_paths.append((tmp.name, file.filename))

    # Add the task to the BackgroundTasks object
    background_tasks.add_task(process_library_creation, task_id, username, library_name, temp_file_paths, model_name, special_prompt)

    return {"task_id": task_id, "status": "Started"}


import asyncio
from concurrent.futures import ThreadPoolExecutor

# from routes.progress import progress_data
from datetime import datetime

from myUtils.redisStateManager import RedisStateManager
redis_state_manager = RedisStateManager()



def process_pdf_files_into_source_docs_table(username, temp_file_paths, library_name: str, cursor):
    for temp_path, original_filename in temp_file_paths:
        print('Processing file:', original_filename)

        with open(temp_path, "rb") as f:
            file_content = f.read()

        cursor.execute(
            "INSERT IGNORE INTO source_docs (file, date_detected, date_extracted, url, title, breadCrumb, checksum, library, username) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)",
            (file_content, 'unknown', datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'unknown',
             original_filename, 'unknown',
             '', library_name, username)
        )




@handle_openai_errors
async def process_library_creation(task_id, username, library_name, temp_file_paths, model_name='openai', special_prompt=None):
    # print('Starting library creation process for:', library_name)

    redis_state_manager.set_state(task_id, {"status": "Started", "progress": 0})
    print('set_state', task_id, {"status": "Started", "progress": 0})
    # progress_data[task_id] = {"status": "Started", "progress": 0}

    try:
        # Use a thread pool for CPU-bound and blocking I/O operations
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor() as pool:
            conn, cursor = await loop.run_in_executor(pool, initialize_all_connection)

            # progress_data[task_id] = {"status": "Processing source_docs", "progress": 10}
            redis_state_manager.set_state(task_id, {"status": "Processing source_docs", "progress": 10})
            await loop.run_in_executor(pool, process_pdf_files_into_source_docs_table, username, temp_file_paths, library_name, cursor)
            await loop.run_in_executor(pool, conn.commit)

            # progress_data[task_id] = {"status": "Inserting big chunks", "progress": 30}
            redis_state_manager.set_state(task_id, {"status": "Inserting big chunks", "progress": 30})
            await loop.run_in_executor(pool, insert_big_chunks_into_db, library_name, username, cursor)
            await loop.run_in_executor(pool, conn.commit)

            # progress_data[task_id] = {"status": "Inserting small chunks", "progress": 50}
            redis_state_manager.set_state(task_id, {"status": "Inserting small chunks", "progress": 50})
            await loop.run_in_executor(pool, insert_small_chunks_into_db, library_name, username, cursor)
            await loop.run_in_executor(pool, conn.commit)

            if model_name == 'openai':
                # calculate cost
                sql_query = "SELECT SUM(n_token) FROM small_chunks WHERE library=%s AND username=%s"
                cursor.execute(sql_query, (library_name, username))
                n_tokens = cursor.fetchone()[0]
                price = n_tokens *1.3e-7
                print('price', price)

            for model_name, language in zip([model_name], ['fr']):
                # progress_data[task_id] = {"status": f"Processing model {model_name}", "progress": 70, "price": price}
                redis_state_manager.set_state(task_id, {"status": f"Processing model {model_name}", "progress": 70, "price": price})
                await loop.run_in_executor(pool, insert_embeddings_models_into_db, model_name, language, dtypes_for_models[model_name], cursor)
                await loop.run_in_executor(pool, conn.commit)

                openai_key, openai_key_status = await loop.run_in_executor(pool, get_openai_key, username)
                # progress_data[task_id] = {"status": "Embedding small chunks", "progress": 80, "price": price}
                redis_state_manager.set_state(task_id, {"status": "Embedding small chunks", "progress": 80, "price": price})
                await loop.run_in_executor(pool, embedd_all_small_chunks, library_name, model_name, language, username, cursor, conn, 50, None, openai_key)
                await loop.run_in_executor(pool, conn.commit)

                # progress_data[task_id] = {"status": "Creating FAISS index", "progress": 90, "price": price}
                redis_state_manager.set_state(task_id, {"status": "Creating FAISS index", "progress": 90, "price": price})
                await loop.run_in_executor(pool, create_faiss_index, library_name, model_name, language, username, cursor)
                await loop.run_in_executor(pool, conn.commit)

            await loop.run_in_executor(pool, insert_user_library, username, library_name, special_prompt, cursor)
            await loop.run_in_executor(pool, conn.commit)

        # progress_data[task_id] = {"status": "Completed", "progress": 100, "price": price}
        redis_state_manager.set_state(task_id, {"status": "Completed", "progress": 100, "price": price})
        # print(f"Library {library_name} created successfully for user {username}")

    except Exception as e:
        # progress_data[task_id] = {"status": "Error", "message": str(e)}
        redis_state_manager.set_state(task_id, {"status": "Error", "message": str(e)})
        # print(f"Error creating library {library_name} for user {username}: {str(e)}")
    finally:
        # Clean up temporary files
        for temp_path, _ in temp_file_paths:
            os.remove(temp_path)

        #update historic table
        query = "INSERT INTO historic (username, action, detail) VALUES (%s, %s, %s)"
        cursor.execute(query, (username, 'create_library', library_name))
        conn.commit()

        print('finished process_library_creation')

        if 'conn' in locals():
            conn.close()


@router.get("/check_user_libraries")
async def check_user_libraries(session_token: str = Cookie(None)):

    # print('check_users_databases, session_token', session_token)

    if session_token is None:
        raise HTTPException(status_code=401, detail="Not authenticated")

    conn, cursor = initialize_all_connection()

    cursor.execute('''SELECT username, session_token from users''')
    myUsers = cursor.fetchall()
    dico_users_session_tokens = {u[0]: u[1] for u in myUsers}

    username = None

    for username_temp in dico_users_session_tokens:
        # print('username', username_temp, 'token', dico_users_session_tokens[username_temp])
        if session_token == dico_users_session_tokens[username_temp]:
            username = username_temp
            break

    if username is None:
        raise HTTPException(status_code=401, detail="Not authenticated")

    try:
        # Query to get all tables belonging to the user
        cursor.execute("SELECT library_name FROM user_libraries WHERE (username = %s OR username='all_users')", (username,))
        user_libraries = cursor.fetchall()

        conn.close()

        # Convert the result to a list of table names
        library_list = [library[0] for library in user_libraries]

        # print('table_list', library_list)

        # put the 'no_library' at the end of the list
        if 'no_library' in library_list:
            library_list.remove('no_library')
            library_list.append('no_library')

        print('library_list', library_list)

        return {"username": username, "libraries": library_list}

    except sqlite3.Error as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

if __name__ == '__main__':
    def delete_library_admin(library_name: str, username):

        conn, cursor = initialize_all_connection()
        cursor.execute("DELETE FROM user_libraries WHERE library_name=%s AND username=%s", (library_name, username))
        cursor.execute("DELETE FROM source_docs WHERE library=%s AND username=%s", (library_name, username))
        cursor.execute("DELETE FROM big_chunks WHERE library=%s AND username=%s", (library_name, username))
        cursor.execute("DELETE FROM small_chunks WHERE library=%s AND username=%s", (library_name, username))
        cursor.execute("DELETE FROM embeddings WHERE library=%s AND username=%s", (library_name, username))
        cursor.execute("DELETE FROM faiss_indexes WHERE library=%s AND username=%s", (library_name, username))
        cursor.execute("DELETE FROM faiss_index_metadata WHERE library=%s AND username=%s", (library_name, username))
        cursor.execute("DELETE FROM faiss_index_parts WHERE library=%s AND username=%s", (library_name, username))
        conn.commit()
        conn.close()

        return {"message": f"Library '{library_name}' deleted successfully"}

    # delete_library_admin('LEX', 'all_users')
    delete_library_admin('LEX AND RH', 'all_users')