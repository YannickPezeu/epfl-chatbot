import os

import sys

import numpy as np

current_folder = os.path.dirname(os.path.abspath(__file__))
root_folder = os.path.abspath(os.path.abspath(os.path.join(current_folder, '..')))
utils_folder = os.path.join(root_folder, 'myUtils')
data_folder = os.path.join(root_folder, 'data')
sys.path.append(root_folder)

# print('root_folder:', root_folder)


import nltk
nltk.download('punkt')
from myUtils.connect_acad import reconnect_on_failure, initialize_all_connection

from myUtils.get_embeddings import get_embeddings



@reconnect_on_failure
def insert_embeddings_models_into_db(model_name, language, dtype, cursor):
    # Insert a row of data
    # print('inserting embeddings_models', model_name, language, dtype)
    cursor.execute(
        "INSERT IGNORE INTO embeddings_models (model_name, language, dtype) VALUES (%s, %s, %s)",
        (model_name, language, dtype)
    )


def get_id_from_model_name(model_name, language, cursor):
    # Connect to database

    cursor.execute(
    "SELECT id FROM embeddings_models WHERE model_name = %s AND language = %s",
    (model_name, language)
)

    model_id = cursor.fetchone()[0]
    # print('model_id', model_id)

    return model_id

def insert_embeddings_into_db(library, model_name, language, small_chunk_id, embedding, username, cursor):

    model_id = get_id_from_model_name(model_name, language, cursor)
    # print('model_id', model_id)

    # Check if embedding already exists
    cursor.execute(
        "SELECT id FROM embeddings WHERE model_id = %s AND small_chunk_id = %s AND library = %s AND username = %s",
        (model_id, small_chunk_id, library, username)
    )
    # print('check5')
    if cursor.fetchone():
        return

    # Insert a row of data
    cursor.execute(
        """INSERT IGNORE INTO embeddings (model_id, small_chunk_id, embedding, library, username) 
        VALUES (%s, %s, %s, %s, %s)""",
        (model_id, small_chunk_id, embedding, library, username)
    )
    # print('check6')

def check_if_small_chunk_is_embedded(model_name, language, small_chunk_id, cursor):
    model_id = get_id_from_model_name(model_name, language, cursor)

    # Check if embedding already exists
    cursor.execute(
        "SELECT id FROM embeddings WHERE model_id = %s AND small_chunk_id = %s",
        (model_id, small_chunk_id)
    )
    if cursor.fetchone():
        return True

    return False

@reconnect_on_failure
def embedd_small_chunk(model_name, small_chunk_id, cursor, mistral_key=None, openai_key=None):


    # retrieve small chunk
    cursor.execute("SELECT * FROM small_chunks WHERE id=%s", (small_chunk_id,))
    small_chunk = cursor.fetchone()

    if not small_chunk:
        # print('small chunk not found')
        return

    # if language == 'fr':
    content = small_chunk[3]  # base content
    # else:
    #     content = small_chunk[5]  # en-translated content

    if not content:
        # print('content not found')
        return

    embedding = get_embeddings(content, model_name, mistral_key, openai_key)

    if embedding is None:
        # print('embedding not found')
        return

    # print('embedding shape', embedding.shape)
    return embedding

def embedd_multiple_small_chunks(model_name, small_chunk_ids, cursor, mistral_key=None, openai_key=None):
    # create small_chunks_list
    cursor.execute("SELECT chunk_content FROM small_chunks WHERE id IN %s", (small_chunk_ids,))
    small_chunks_list = cursor.fetchall()

    #to list
    small_chunks_list = [s[0] for s in small_chunks_list]
    # for s in small_chunks_list:
        # print(s)
        # print('-'*50)
    embeddings_list = get_embeddings(
        text_list=small_chunks_list,
        model_name=model_name,
        mistral_key=mistral_key,
        openai_key=openai_key
    )
    # print('embeddings_list', embeddings_list)



    return embeddings_list

@reconnect_on_failure
def embedd_all_small_chunks(library, model_name, language, username, cursor, step_size=50, mistral_key=None, openai_key=None):

    cursor.execute("SELECT id FROM small_chunks where library=%s AND username=%s ORDER BY id", (library, username))

    small_chunk_ids = cursor.fetchall()
    small_chunk_ids = [small_chunk_id[0] for small_chunk_id in small_chunk_ids]

    # print('small_chunk_ids', small_chunk_ids)

    for i in range(0, len(small_chunk_ids), step_size):
        small_chunk_group = small_chunk_ids[i:i + step_size]
        try:
            # print('small_chunk_group', small_chunk_group)
            embeddings = embedd_multiple_small_chunks(model_name, small_chunk_group, cursor, mistral_key=mistral_key, openai_key=openai_key)
            # print('embeddings', embeddings)
            for j, small_chunk_id in enumerate(small_chunk_group):
                embedding = embeddings[j][np.newaxis, :]
                embedding_bytes = embedding.tobytes()
                insert_embeddings_into_db(library, model_name, language, small_chunk_id, embedding_bytes, username, cursor)
        except Exception as e:

            # print(f'Error embedding small chunk {small_chunk_group}', e)
            #print traceback


            #embedd small chunks one by one
            for small_chunk_id in small_chunk_group:
                try:
                    embedding = embedd_small_chunk(model_name, small_chunk_id, cursor, mistral_key=mistral_key, openai_key=openai_key)
                    embedding_bytes = embedding.tobytes()
                    insert_embeddings_into_db(library, model_name, language, small_chunk_id, embedding_bytes, username, cursor)
                except Exception as e:
                    print(f'Error embedding small chunk {small_chunk_id}')
            # print(e)


    # for i, small_chunk_id in enumerate(small_chunk_ids):
    #     if i%step_size == 0:
    #         print('small_chunk_id', small_chunk_id)
    #         print(f'processing small chunk {i}/{len(small_chunk_ids)}')
    #     try:
    #         print('check0')
    #         is_small_chunk_already_embedded_in_db = check_if_small_chunk_is_embedded(
    #             model_name, language, small_chunk_id, cursor
    #         )
    #         print('check1')
    #         if is_small_chunk_already_embedded_in_db:
    #             continue
    #         embedding = embedd_small_chunk(model_name, small_chunk_id, cursor, mistral_key=mistral_key, openai_key=openai_key)
    #         print('check2')
    #         embedding_bytes = embedding.tobytes()
    #         print('check3')
    #         insert_embeddings_into_db(library, model_name, language, small_chunk_id, embedding_bytes, username, cursor)
    #         print('check4')
    #     except Exception as e:
    #         print(f'Error embedding small chunk {small_chunk_id}')
    #         print(e)
    #
    #
    # print('check9')

dtypes_for_models = {
    'openai': 'float64',
    'mistral': 'float64',
    'camembert': 'float32',
    'mpnet': 'float32',
    'gte': 'float32',
    'embaas': 'float32',
    'fr_long_context': 'float32',
}

if __name__ == '__main__':

    conn, cursor = initialize_all_connection()
    import dotenv
    import os
    openai_key = os.getenv('OPENAI_KEY')

    embedd_multiple_small_chunks('openai', [1,2,3], cursor=cursor, mistral_key=None, openai_key=openai_key)
    test = embedd_small_chunk('openai', 1, cursor=cursor, mistral_key=None, openai_key=openai_key)
    print(test.shape)

    exit()
    db_lex_path = os.path.join(data_folder, 'LEXs', 'LEXs.db')
    db_rh_path = os.path.join(data_folder, 'HR', 'HR.db')
    model_names_bilingual = ['mistral', 'openai']
    languages = ['fr', 'en']
    model_names_fr = ['camembert']
    model_names_en = ['mpnet']


    #
    # for db_path in [db_lex_path, db_rh_path]:
    #     create_table_embeddings_models()
    #     create_table_embeddings()
    #     for model_name in model_names_bilingual:
    #         for language in languages:
    #             dtype = dtypes_for_models[model_name]
    #             print(f'Embedding {model_name} in {language}')
    #             insert_embeddings_models_into_db(db_path, model_name, language, dtype)
    #             embedd_all_small_chunks(db_path, model_name, language)
    #
    #     for model_name in model_names_fr:
    #         for language in ['fr']:
    #             dtype = dtypes_for_models[model_name]
    #             print(f'Embedding {model_name} in {language}')
    #             insert_embeddings_models_into_db(db_path, model_name, language, dtype)
    #             embedd_all_small_chunks(db_path, model_name, language)
    #
    #     for model_name in model_names_en:
    #         for language in ['en']:
    #             dtype = dtypes_for_models[model_name]
    #             print(f'Embedding {model_name} in {language}')
    #             insert_embeddings_models_into_db(db_path, model_name, language, dtype)
    #             embedd_all_small_chunks(db_path, model_name, language)




