import math
import sqlite3
import time

import numpy as np

from myUtils.retrieve_embedding_from_db_online import retrieve_embedding_from_db_online
from myUtils.connect_acad import reconnect_on_failure
import faiss
import json
import tempfile
import os
from myUtils.redisStateManager import RedisStateManager


def split_file(file_path, chunk_size=20 * 1024 * 1024):  # 20MB in bytes
    file_size = os.path.getsize(file_path)
    total_chunks = math.ceil(file_size / chunk_size)

    chunks = []
    with open(file_path, 'rb') as f:
        for _ in range(total_chunks):
            chunk = f.read(chunk_size)
            chunks.append(chunk)

    return chunks

def reconstruct_file(base_path, num_parts, output_path):
    with open(output_path, 'wb') as outfile:
        for i in range(num_parts):
            part_path = f"{base_path}.part{i+1}"
            with open(part_path, 'rb') as infile:
                outfile.write(infile.read())
    # print(f"File reconstructed at {output_path}")

def init_local_db():
    conn = sqlite3.connect('local.db')
    cursor = conn.cursor()
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS faiss_index (
        model_id INTEGER,
        library TEXT,
        username TEXT,
        index_data BLOB,
        embedding_ids TEXT,
        PRIMARY KEY (model_id, library, username)
    )
    ''')
    cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_faiss_lookup 
            ON faiss_index(model_id, library, username)
        ''')
    conn.commit()
    conn.execute('PRAGMA journal_mode=WAL')
    conn.execute('PRAGMA cache_size=-2000000')  # 2GB cache
    conn.execute('PRAGMA synchronous=NORMAL')
    conn.execute('PRAGMA temp_store=MEMORY')
    conn.execute('PRAGMA mmap_size=30000000000')  # 30GB memory map
    return conn, cursor


@reconnect_on_failure
def create_faiss_index(library, model_name, language, username, cursor):

    # print('create_faiss_index, library:', library, 'model_name:', model_name, 'language:', language, 'username:', username)

    # print('execute_query')

    cursor.execute("SELECT id FROM embeddings_models WHERE model_name=%s AND language=%s ORDER BY id",
              (model_name, language))
    model_id = cursor.fetchone()[0]

    # print('model_id:', model_id)

    # print('retrieve_embedding_from_db')

    #get embedding_ids from db

    cursor.execute("""
        SELECT id FROM acad.embeddings 
        WHERE (model_id = %s OR %s IS NULL)
        AND (library = %s OR %s IS NULL)
        AND (username = %s OR %s IS NULL)
        ORDER BY id
    """, (model_id, model_id, library, library, username, username))

    # print('TEST')

    embedding_ids = [row[0] for row in cursor.fetchall()]
    print('embedding_ids:', embedding_ids)

    embeddings = retrieve_embedding_from_db_online(embedding_ids, cursor)

    # print('create_index')
    myIndex = faiss.IndexFlatL2(embeddings.shape[1])
    myIndex.add(embeddings)

    # Create a temporary file
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file_path = temp_file.name

    # Write the index to the temporary file
    faiss.write_index(myIndex, temp_file_path)

    # Split the file into chunks
    chunks = split_file(temp_file_path)
    # print(f"Index split into {len(chunks)} parts")

    # Remove the temporary file
    os.unlink(temp_file_path)

    # Insert the chunks into the database
    for i, chunk in enumerate(chunks):
        cursor.execute("""
                INSERT INTO faiss_index_parts (
                model_id, 
                part_number,
                total_parts,
                index_part, 
                library,
                username
                ) VALUES (%s, %s, %s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE
                index_part = VALUES(index_part),
                total_parts = VALUES(total_parts)
                """, (model_id, i + 1, len(chunks), chunk, library, username))

    # Store the metadata
    cursor.execute("""
            INSERT INTO faiss_index_metadata (
            model_id, 
            embedding_ids, 
            library,
            username
            ) VALUES (%s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE
            embedding_ids = VALUES(embedding_ids)
            """, (model_id, json.dumps(embedding_ids), library, username))

    # print('FAISS index stored in parts')

    # store the whole index in local SQLite DB
    # Instead of splitting and storing in parts, we'll store the whole index
    index_bytes = faiss.serialize_index(myIndex)

    # Store in local SQLite DB
    local_conn, local_cursor = init_local_db()
    local_cursor.execute('''
        INSERT OR REPLACE INTO faiss_index (model_id, library, username, index_data, embedding_ids)
        VALUES (?, ?, ?, ?, ?)
        ''', (model_id, library, username, index_bytes, json.dumps(embedding_ids)))
    local_conn.commit()
    local_conn.close()

    # print('FAISS index stored in local SQLite database')

    return myIndex


@reconnect_on_failure
def retrieve_faiss_index(model_name, language, library, username, cursor):
    # Get the model_id
    s = time.time()
    cursor.execute("SELECT id FROM embeddings_models WHERE model_name=%s AND language=%s", (model_name, language))
    model_id = cursor.fetchone()[0]
    print('time to get model_id:', time.time() - s)

    # Initialize Redis managers for different data types
    string_redis = RedisStateManager(decode_responses=True)  # For JSON/string data
    binary_redis = RedisStateManager(decode_responses=False)  # For binary data

    # Create keys for faiss index and embeddings
    index_key = f"faiss:index:{model_id}:{library}:{username}"
    embedding_key = f"faiss:embeddings:{model_id}:{library}:{username}"

    # Try user-specific index first
    index_bytes = binary_redis.redis_client.get(index_key)
    embedding_ids_json = string_redis.redis_client.get(embedding_key)

    # If not found, try all_users
    if index_bytes is None:
        all_users_index_key = f"faiss:index:{model_id}:{library}:all_users"
        all_users_embedding_key = f"faiss:embeddings:{model_id}:{library}:all_users"
        index_bytes = binary_redis.redis_client.get(all_users_index_key)
        embedding_ids_json = string_redis.redis_client.get(all_users_embedding_key)

    if index_bytes and embedding_ids_json:
        print('getting faiss index from redis')
        index_array = np.frombuffer(index_bytes, dtype=np.uint8)
        index = faiss.deserialize_index(index_array)
        embedding_ids = json.loads(embedding_ids_json)
        return index, embedding_ids

    # If not in Redis, retrieve from MySQL DB
    print('getting faiss index from online db')
    cursor.execute("""
    SELECT embedding_ids 
    FROM faiss_index_metadata 
    WHERE model_id=%s AND library=%s AND (username=%s OR username='all_users')
    """, (model_id, library, username))

    result = cursor.fetchone()
    if result is None:
        return None

    embedding_ids_json = result[0]

    # Retrieve all parts of the index
    cursor.execute("""
    SELECT part_number, index_part 
    FROM faiss_index_parts 
    WHERE model_id=%s AND library=%s AND (username=%s OR username='all_users')
    ORDER BY part_number
    """, (model_id, library, username))

    parts = cursor.fetchall()
    combined_index = b''.join(part[1] for part in parts)

    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(combined_index)
        temp_file_path = temp_file.name

    index = faiss.read_index(temp_file_path)
    os.unlink(temp_file_path)

    embedding_ids = json.loads(embedding_ids_json)

    # Store in Redis for future use
    print('storing faiss index in redis')
    index_bytes = faiss.serialize_index(index)
    binary_redis.redis_client.set(index_key, index_bytes)
    string_redis.redis_client.set(embedding_key, embedding_ids_json)

    # Set expiry time
    binary_redis.redis_client.expire(index_key, 4 * 3600)
    string_redis.redis_client.expire(embedding_key, 4 * 3600)

    return index, embedding_ids


if __name__ == '__main__':
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(current_dir)

    db_paths = [os.path.join(root_dir, 'data/LEXs/LEXs.db'),
                os.path.join(root_dir, 'data/HR/HR.db')]

    bilingual_model_names = ['mistral', 'openai']
    languages = ['fr', 'en']
    en_model_names = ['mpnet']
    fr_model_names = ['camembert']

    for db_path in db_paths:
        for model_name in bilingual_model_names:
            for language in languages:
                create_faiss_index(db_path, model_name, language)
        for model_name in en_model_names:
            create_faiss_index(db_path, model_name, 'en')
        for model_name in fr_model_names:
            create_faiss_index(db_path, model_name, 'fr')








