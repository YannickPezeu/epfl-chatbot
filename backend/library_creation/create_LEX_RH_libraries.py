from fastapi import HTTPException

from library_creation._2_embedd_small_chunks import dtypes_for_models
from routes.libraries import insert_user_library, insert_big_chunks_into_db, insert_small_chunks_into_db, insert_embeddings_models_into_db, embedd_all_small_chunks, create_faiss_index
from myUtils.connect_acad2 import reconnect_on_failure, initialize_all_connection

import dotenv
import os

dotenv.load_dotenv(dotenv.find_dotenv())
mistral_key = os.getenv('MISTRAL_KEY')
openai_key = os.getenv('OPENAI_KEY')

def create_library_from_pdf_table(library_name):
    conn, cursor = initialize_all_connection()
    username = 'all_users'
    print(
        f"Creating library '{library_name}' from source_docs table. This may take a while depending on the number of source_docs in the table.")

    #create source_docs table if not exists
    try:

        # insert_big_chunks_into_db(library_name, username, cursor=cursor)
        # conn.commit()
        # print('inserting small chunks')
        # insert_small_chunks_into_db(library_name, username, cursor=cursor, connection=conn)
        # conn.commit()
        # print('creating embeddings models')


        for model_name, language in zip([ 'rcp',
                                          # 'camembert', 'mistral', 'mpnet','fr_long_context', 'embaas', 'gte'
                                          ], ['fr',
                                              # 'fr','fr','fr','fr','fr','fr',
                                              ]):
            print('model_name', model_name, 'language', language)
            print('inserting embeddings models into db')
            insert_embeddings_models_into_db(model_name, language, dtypes_for_models[model_name], cursor=cursor, )
            conn.commit()

            print('embedding all small chunks:', model_name, language, library_name, username)
            embedd_all_small_chunks(library_name, model_name, language, username, cursor=cursor, connection=conn, mistral_key=mistral_key, openai_key=openai_key)
            conn.commit()

            print('creating faiss index')
            create_faiss_index(library_name, model_name, language, username, cursor=cursor)
            conn.commit()

        insert_user_library(username, library_name, cursor=cursor)
        conn.commit()

        print('user library inserted')

        return {
            "success": True,
            "library_name": library_name,
            "message": f"Database '{library_name}' created successfully"
        }

    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == '__main__':

    conn, cursor = initialize_all_connection()

    # create_library_from_pdf_table('LEX')
    # create_library_from_pdf_table('RH')
    create_library_from_pdf_table('LEX AND RH')