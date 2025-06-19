# routes/libraries.py
import traceback

from fastapi import APIRouter, Cookie, BackgroundTasks
from langsmith import expect

from myUtils.connect_acad2 import initialize_all_connection, reconnect_on_failure
from routes.auth import get_username_from_session_token

from myUtils.ask_chatGPT import ask_chatGPT
from routes.auth import get_openai_key
from myUtils.handle_openai_errors import handle_openai_errors

import os

import sqlite3
import re

from library_creation._0_create_big_chunks_from_source_docs import insert_big_chunks_into_db
from library_creation._1_create_small_chunks_from_big_chunks import insert_small_chunks_into_db
from library_creation._2_embedd_small_chunks import dtypes_for_models, insert_embeddings_models_into_db, embedd_all_small_chunks
from library_creation._3_create_faiss_index import create_faiss_index


from fastapi import File, UploadFile, Cookie, HTTPException, Request
from typing import List
import tempfile
from fastapi import Form

import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)
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
    text_to_summarize = [source_doc_content[0] for source_doc_content in source_docs_content]
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
    background_tasks: BackgroundTasks = None,
    doc_type: str = 'pdf'
):


    username = get_username_from_session_token(session_token)
    task_id = str(username) + '_' + library_name

    # Save files to temporary directory
    temp_file_paths = []
    suffix = ''
    if doc_type == 'pdf':
        suffix = '.pdf'
    elif doc_type == 'txt':
        suffix = '.txt'
    elif doc_type == 'json':
        suffix = '.json'
    else:
        print('doc_type not recognized')
        raise Exception

    for file in files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
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
import json

def process_source_docs_files_into_source_docs_table(username, temp_file_paths, library_name: str, cursor, doc_type,
                                                     urls_json=None):
    if urls_json is None:
        # Original functionality when no JSON metadata is provided
        for temp_path, original_filename in temp_file_paths:
            print('Processing file:', original_filename)

            with open(temp_path, "rb") as f:
                file_content = f.read()

            cursor.execute(
                "INSERT IGNORE INTO source_docs (file, date_detected, date_extracted, url, title, breadCrumb, checksum, library, username, doc_type) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
                (file_content, 'unknown', datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'unknown',
                 original_filename, 'unknown',
                 '', library_name, username, doc_type)
            )

    else:
        # Process with JSON metadata
        # urls_json is now a list of JSON file paths
        metadata_dict = {}
        total_metadata_count = 0

        # Load metadata from each JSON file in the list
        try:
            if isinstance(urls_json, str):
                # Handle case where a single path is provided
                urls_json = [urls_json]

            for json_path in urls_json:
                if os.path.exists(json_path):
                    with open(json_path, 'r', encoding='utf-8') as f:
                        metadata_list = json.load(f)

                    # Add to the combined metadata dictionary
                    for item in metadata_list:
                        # Store by title
                        metadata_dict[item['title']] = item

                        # Also store by sanitized title (filename without extension)
                        sanitized_title = sanitize_filename(item['title'])
                        if sanitized_title.endswith('.pdf'):
                            sanitized_title = sanitized_title[:-4]  # Remove .pdf extension
                        metadata_dict[sanitized_title] = item

                    total_metadata_count += len(metadata_list)
                    print(f"Loaded metadata for {len(metadata_list)} documents from {json_path}")
                else:
                    print(f"Warning: JSON file not found: {json_path}")

            print(f"Combined metadata for {total_metadata_count} documents from {len(urls_json)} JSON files")

            for temp_path, original_filename in temp_file_paths:
                print('Processing file:', original_filename)

                with open(temp_path, "rb") as f:
                    file_content = f.read()

                # Try to find matching metadata
                file_basename = os.path.basename(original_filename)
                if file_basename.endswith('.pdf'):
                    file_basename = file_basename[:-4]  # Remove .pdf extension

                # Try different ways to match the file with metadata
                metadata = None

                # Try direct match with filename
                if file_basename in metadata_dict:
                    metadata = metadata_dict[file_basename]
                else:
                    # Try to find a partial match (in case the filename is slightly different)
                    for key, value in metadata_dict.items():
                        if file_basename in key or key in file_basename:
                            metadata = value
                            break

                # If we still don't have a match, try using Levenshtein distance for fuzzy matching
                if metadata is None:
                    best_match = None
                    best_ratio = 0
                    import difflib

                    for key in metadata_dict.keys():
                        ratio = difflib.SequenceMatcher(None, file_basename, key).ratio()
                        if ratio > 0.8 and ratio > best_ratio:  # Use 0.8 as threshold
                            best_ratio = ratio
                            best_match = key

                    if best_match:
                        metadata = metadata_dict[best_match]
                        print(f"Found fuzzy match ({best_ratio:.2f}) for {file_basename}: {best_match}")

                if metadata:
                    print(f"Found metadata match for {file_basename}: {metadata['title']}")

                    # Extract date from lastUpdate if available
                    date_detected = 'unknown'
                    if 'lastUpdate' in metadata:
                        date_detected = metadata['lastUpdate'].split(' ')[0]  # Get just the date part

                    cursor.execute(
                        "INSERT IGNORE INTO source_docs (file, date_detected, date_extracted, url, title, breadCrumb, checksum, library, username, doc_type) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
                        (file_content, date_detected, datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                         metadata['url'], metadata['title'], 'unknown',
                         '', library_name, username, doc_type)
                    )
                else:
                    print(f"No metadata found for {file_basename}, using default values")

                    # Fall back to default values if no match found
                    cursor.execute(
                        "INSERT IGNORE INTO source_docs (file, date_detected, date_extracted, url, title, breadCrumb, checksum, library, username, doc_type) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
                        (file_content, 'unknown', datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'unknown',
                         original_filename, 'unknown',
                         '', library_name, username, doc_type)
                    )

        except Exception as e:
            print(f"Error processing JSON metadata: {e}")
            import traceback
            traceback.print_exc()

            # Fallback to processing without metadata
            for temp_path, original_filename in temp_file_paths:
                print('Processing file:', original_filename)

                with open(temp_path, "rb") as f:
                    file_content = f.read()

                cursor.execute(
                    "INSERT IGNORE INTO source_docs (file, date_detected, date_extracted, url, title, breadCrumb, checksum, library, username, doc_type) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
                    (file_content, 'unknown', datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'unknown',
                     original_filename, 'unknown',
                     '', library_name, username, doc_type)
                )



def sanitize_filename(filename):
    """Create a valid filename by removing invalid characters."""
    # Replace invalid characters with underscore
    invalid_chars = r'[<>:"/\\|?*]'
    sanitized = re.sub(invalid_chars, '_', filename)
    # Ensure the filename is not too long
    if len(sanitized) > 200:
        name, ext = os.path.splitext(sanitized)
        sanitized = name[:195] + ext
    # Ensure it has a .pdf extension
    if not sanitized.lower().endswith('.pdf'):
        sanitized += '.pdf'
    return sanitized




@handle_openai_errors
async def process_library_creation(task_id, username, library_name, temp_file_paths, model_name='openai',
                                             special_prompt=None, doc_type='pdf', remove_temp_files=True, urls_json=None):
    """
    Creates a new library or updates an existing one with minimal downtime.

    Args:
        task_id (str): Unique identifier for tracking progress
        username (str): Username for whom the library is being created/updated
        library_name (str): Name of the library
        temp_file_paths (list): List of tuples (file_path, original_filename)
        model_name (str): Model to use for embeddings
        special_prompt (str, optional): Special prompt for the library
        doc_type (str): Document type, defaults to 'pdf'
    """
    redis_state_manager.set_state(task_id, {"status": "Started", "progress": 0})
    print('set_state', task_id, {"status": "Started", "progress": 0})

    # Use a temp library name if updating
    temp_library = f"{library_name}_{username}_temp"

    try:
        # Use a thread pool for CPU-bound and blocking I/O operations
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor() as pool:
            conn, cursor = await loop.run_in_executor(pool, initialize_all_connection)

            # Check if library already exists for this user
            await loop.run_in_executor(
                pool,
                lambda: cursor.execute("SELECT COUNT(*) FROM user_libraries WHERE username = %s AND library_name = %s",
                                       (username, library_name))
            )
            library_exists = await loop.run_in_executor(pool, lambda: cursor.fetchone()[0] > 0)

            # If library exists, we'll use the update approach
            actual_library_name = temp_library if library_exists else library_name

            redis_state_manager.set_state(task_id, {
                "status": f"{'Updating' if library_exists else 'Creating'} library",
                "progress": 5
            })

            # Process files and insert into source_docs
            redis_state_manager.set_state(task_id, {"status": "Processing source_docs", "progress": 10})
            if temp_file_paths:
                await loop.run_in_executor(
                    pool,
                    process_source_docs_files_into_source_docs_table,
                    username, temp_file_paths, actual_library_name, cursor, doc_type, urls_json
                )
                await loop.run_in_executor(pool, conn.commit)

            # Insert big chunks
            redis_state_manager.set_state(task_id, {"status": "Inserting big chunks", "progress": 30})
            await loop.run_in_executor(
                pool,
                insert_big_chunks_into_db,
                actual_library_name, username, cursor, conn
            )
            await loop.run_in_executor(pool, conn.commit)

            # Insert small chunks
            redis_state_manager.set_state(task_id, {"status": "Inserting small chunks", "progress": 50})
            await loop.run_in_executor(
                pool,
                insert_small_chunks_into_db,
                actual_library_name, username, cursor
            )
            await loop.run_in_executor(pool, conn.commit)

            # Calculate cost for OpenAI
            if model_name == 'openai':
                sql_query = "SELECT SUM(n_token) FROM small_chunks WHERE library=%s AND username=%s"
                cursor.execute(sql_query, (actual_library_name, username))
                n_tokens = cursor.fetchone()[0]
                price = n_tokens * 1.3e-7
                print('price', price)
            else:
                price = 0

            # Process embeddings for the model
            for model_name, language in zip([model_name], ['fr']):
                print(f'starting embedding_process for model {model_name} in language {language}')
                redis_state_manager.set_state(task_id, {
                    "status": f"Processing model {model_name}",
                    "progress": 70,
                    "price": price
                })

                # Insert embeddings models
                await loop.run_in_executor(
                    pool,
                    insert_embeddings_models_into_db,
                    model_name, language, dtypes_for_models[model_name], cursor
                )
                await loop.run_in_executor(pool, conn.commit)

                # Get OpenAI key if needed
                if model_name == 'openai':
                    openai_key, openai_key_status = await loop.run_in_executor(
                        pool,
                        get_openai_key,
                        username
                    )
                else:
                    openai_key = None

                # Embed small chunks
                redis_state_manager.set_state(task_id, {
                    "status": "Embedding small chunks",
                    "progress": 80,
                    "price": price
                })

                try:
                    await loop.run_in_executor(
                        pool,
                        embedd_all_small_chunks,
                        actual_library_name, model_name, language, username, cursor, conn, 50, None, openai_key
                    )
                    await loop.run_in_executor(pool, conn.commit)
                except Exception as e:
                    print('Error embed all small chunks', e)

                # Create FAISS index
                redis_state_manager.set_state(task_id, {
                    "status": "Creating FAISS index",
                    "progress": 90,
                    "price": price
                })

                await loop.run_in_executor(
                    pool,
                    create_faiss_index,
                    actual_library_name, model_name, language, username, cursor
                )
                await loop.run_in_executor(pool, conn.commit)

            # If we're updating, perform the swap operation
            if library_exists:
                redis_state_manager.set_state(task_id, {
                    "status": "Swapping temporary and production libraries",
                    "progress": 95,
                    "price": price
                })

                # Start transaction for atomic update
                await loop.run_in_executor(pool, lambda: cursor.execute("START TRANSACTION"))

                # Find tables with library and username columns
                await loop.run_in_executor(pool, lambda: cursor.execute("""
                    SELECT t1.TABLE_NAME 
                    FROM INFORMATION_SCHEMA.COLUMNS t1
                    JOIN INFORMATION_SCHEMA.COLUMNS t2 
                    ON t1.TABLE_NAME = t2.TABLE_NAME AND t1.TABLE_SCHEMA = t2.TABLE_SCHEMA
                    WHERE t1.COLUMN_NAME = 'library' 
                    AND t2.COLUMN_NAME = 'username'
                    AND t1.TABLE_SCHEMA = DATABASE()
                """))

                tables_with_library_and_username = await loop.run_in_executor(
                    pool,
                    lambda: [row[0] for row in cursor.fetchall()]
                )

                # Find tables with model_id column
                await loop.run_in_executor(pool, lambda: cursor.execute("""
                    SELECT t1.TABLE_NAME 
                    FROM INFORMATION_SCHEMA.COLUMNS t1
                    JOIN INFORMATION_SCHEMA.COLUMNS t2 
                    ON t1.TABLE_NAME = t2.TABLE_NAME AND t1.TABLE_SCHEMA = t2.TABLE_SCHEMA
                    JOIN INFORMATION_SCHEMA.COLUMNS t3
                    ON t1.TABLE_NAME = t3.TABLE_NAME AND t1.TABLE_SCHEMA = t3.TABLE_SCHEMA
                    WHERE t1.COLUMN_NAME = 'library' 
                    AND t2.COLUMN_NAME = 'username'
                    AND t3.COLUMN_NAME = 'model_id'
                    AND t1.TABLE_SCHEMA = DATABASE()
                """))

                tables_with_model_id = await loop.run_in_executor(
                    pool,
                    lambda: [row[0] for row in cursor.fetchall()]
                )

                # Get model ID if needed
                model_id = None
                if model_name:
                    await loop.run_in_executor(
                        pool,
                        lambda: cursor.execute("SELECT id FROM embeddings_models WHERE model_name = %s", (model_name,))
                    )
                    result = await loop.run_in_executor(pool, lambda: cursor.fetchone())
                    if result:
                        model_id = result[0]

                # Sort tables by dependency
                tables_in_order = sorted(
                    tables_with_library_and_username,
                    key=lambda t: (
                        0 if "embedding" in t or "faiss" in t else
                        1 if "small" in t else
                        2 if "big" in t else
                        3 if t == "source_docs" else
                        4 if t == "user_libraries" else 5
                    )
                )

                # Delete existing library data
                for table in tables_in_order:
                    if table in tables_with_model_id and model_id:
                        await loop.run_in_executor(
                            pool,
                            lambda: cursor.execute(
                                f"DELETE FROM {table} WHERE library = %s AND username = %s AND model_id = %s",
                                (library_name, username, model_id)
                            )
                        )
                    else:
                        await loop.run_in_executor(
                            pool,
                            lambda: cursor.execute(
                                f"DELETE FROM {table} WHERE library = %s AND username = %s",
                                (library_name, username)
                            )
                        )

                # Update temp library to be the main library
                for table in tables_with_library_and_username:
                    if table in tables_with_model_id and model_id:
                        await loop.run_in_executor(
                            pool,
                            lambda: cursor.execute(
                                f"UPDATE {table} SET library = %s WHERE library = %s AND username = %s AND model_id = %s",
                                (library_name, temp_library, username, model_id)
                            )
                        )
                    else:
                        await loop.run_in_executor(
                            pool,
                            lambda: cursor.execute(
                                f"UPDATE {table} SET library = %s WHERE library = %s AND username = %s",
                                (library_name, temp_library, username)
                            )
                        )

                # Commit transaction
                await loop.run_in_executor(pool, lambda: cursor.execute("COMMIT"))
            else:
                # If creating new library, just insert into user_libraries
                await loop.run_in_executor(
                    pool,
                    insert_user_library,
                    username, library_name, special_prompt, cursor
                )
                await loop.run_in_executor(pool, conn.commit)

        # Mark task as completed
        redis_state_manager.set_state(task_id, {"status": "Completed", "progress": 100, "price": price})

    except Exception as e:
        # Handle errors
        logger.error(f'encontered error {e}, {traceback.format_exc()}')
        redis_state_manager.set_state(task_id, {"status": "Error", "message": str(e)})

    finally:
        # Clean up temporary files
        if remove_temp_files:
            for temp_path, _ in temp_file_paths:
                os.remove(temp_path)

        # Update historic table
        query = "INSERT INTO historic (username, action, detail) VALUES (%s, %s, %s)"
        cursor.execute(query, (username, 'create_or_update_library', library_name))
        conn.commit()

        print('finished process_library_creation_or_update')

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
    # delete_library_admin('LEX AND RH', 'all_users')

    # asyncio.run(process_library_creation, )

import asyncio
import os
from pathlib import Path
import time

async def main():
    # Define your parameters
    task_id = "task_" + str(int(time.time()))  # Create a unique task ID using timestamp
    username = "servicenow_user"
    library_name = "servicenow_finance"
    doc_type = 'pdf'

    # Get all PDF files from the specified directory
    pdf_dir = Path(r"C:\Dev\EPFL-chatbot-clean\backend\servicenow\scraping\epfl_finance_kb")
    pdf_files = list(pdf_dir.glob("*.pdf"))

    pdf_dir2 = Path(r"C:\Dev\EPFL-chatbot-clean\backend\epfl_hr_scraper\data\LEX_finances")
    pdf_files += list(pdf_dir2.glob("*.pdf"))

    # Create temp_file_paths list as expected by the function
    # The function expects tuples of (file_path, original_filename)
    temp_file_paths = [(str(pdf_file), pdf_file.name) for pdf_file in pdf_files]

    urls_json_paths = [
        r'C:\Dev\EPFL-chatbot-clean\backend\epfl_hr_scraper\data\LEX_finances\lex_finance_articles.json',
        r'C:\Dev\EPFL-chatbot-clean\backend\servicenow\scraping\epfl_finance_kb\articles.json'
    ]

    if not temp_file_paths:
        print("No PDF files found in the specified directory.")
        return

    print(f"Found {len(temp_file_paths)} PDF files to process")

    # Call the function with your parameters
    await process_library_creation(
        task_id=task_id,
        username=username,
        library_name=library_name,
        temp_file_paths=temp_file_paths,
        model_name='rcp',  # Using default model
        special_prompt=None,  # No special prompt
        doc_type=doc_type,
        remove_temp_files=False,
        urls_json=urls_json_paths,

    )

    print(f"Library creation process for {library_name} initiated with task ID: {task_id}")


if __name__ == "__main__":
    asyncio.run(main())