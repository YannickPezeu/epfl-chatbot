from fastapi import HTTPException

from library_creation._2_embedd_small_chunks import dtypes_for_models
from routes.libraries import insert_user_library, insert_big_chunks_into_db, insert_small_chunks_into_db, insert_embeddings_models_into_db, embedd_all_small_chunks, create_faiss_index
from myUtils.connect_acad2 import reconnect_on_failure, initialize_all_connection

import dotenv
import os

dotenv.load_dotenv(dotenv.find_dotenv())
mistral_key = os.getenv('MISTRAL_KEY')
openai_key = os.getenv('OPENAI_KEY')

def create_library_from_pdf_table(library_name, username, model):
    conn, cursor = initialize_all_connection()
    print(
        f"Creating library '{library_name}' from source_docs table. for user {username} and model {model} This may take a while depending on the number of source_docs in the table.")

    #create source_docs table if not exists
    try:
        insert_big_chunks_into_db(library_name, username, cursor=cursor, conn=conn)
        conn.commit()
        print('inserting small chunks')
        insert_small_chunks_into_db(library_name, username, cursor=cursor, connection=conn)
        conn.commit()
        print('creating embeddings models')


        for model_name, language in zip([ model,
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


def update_library_with_minimal_downtime(source_library, username, model=None):
    """
    Updates a library with fresh data from source_docs for a specific username and model while minimizing downtime.

    Args:
        source_library (str): The name of the source library to update
        username (str): The username whose library data should be updated
        model (str, optional): The model name to lookup in embeddings_models. Defaults to None.

    Returns:
        dict: A dictionary containing success status and message
    """
    conn, cursor = initialize_all_connection()
    temp_library = f"{source_library}_{username}_temp"
    model_id = None

    try:
        # If model is provided, get the corresponding model_id from embeddings_models
        if model:
            print(f"Looking up model_id for model '{model}'...")
            cursor.execute("SELECT id FROM embeddings_models WHERE model_name = %s", (model,))
            result = cursor.fetchone()
            if result:
                model_id = result[0]
                print(f"Found model_id: {model_id} for model: '{model}'")
            else:
                print(f"Warning: Model '{model}' not found in embeddings_models table")

        # Step 1: Check if temp_library already exists and clean it if it does
        print(f"Checking for existing '{temp_library}' data for user '{username}'...")
        cursor.execute("SELECT COUNT(*) FROM source_docs WHERE library = %s AND username = %s",
                       (temp_library, username))
        count = cursor.fetchone()[0]
        if count > 0:
            print(f"Found {count} existing records in {temp_library} for {username}. Cleaning up...")
            cursor.execute("DELETE FROM source_docs WHERE library = %s AND username = %s",
                           (temp_library, username))
            conn.commit()

        # Step 2: Copy source_docs from source_library to temp_library for the specific username
        print(f"Copying documents from '{source_library}' to '{temp_library}' for user '{username}'...")
        cursor.execute("""
            INSERT INTO source_docs 
            (file, date_detected, date_extracted, url, title, breadCrumb, checksum, library, username, doc_type)
            SELECT file, date_detected, date_extracted, url, title, breadCrumb, checksum, %s, username, doc_type
            FROM source_docs 
            WHERE library = %s AND username = %s
        """, (temp_library, source_library, username))
        copied_count = cursor.rowcount
        conn.commit()
        print(f"Copied {copied_count} documents to '{temp_library}' for user '{username}'")

        # Step 3: Process temp_library to create all derivative data
        print(f"Processing '{temp_library}' for user '{username}' to create derivative data...")
        create_library_from_pdf_table(temp_library, username, model)  # Pass model name to this function
        print(f"Finished processing '{temp_library}' for user '{username}'")

        # Step 4: Find all tables with both 'library' and 'username' columns
        print("Identifying tables with both 'library' and 'username' columns...")
        cursor.execute("""
            SELECT t1.TABLE_NAME 
            FROM INFORMATION_SCHEMA.COLUMNS t1
            JOIN INFORMATION_SCHEMA.COLUMNS t2 
            ON t1.TABLE_NAME = t2.TABLE_NAME AND t1.TABLE_SCHEMA = t2.TABLE_SCHEMA
            WHERE t1.COLUMN_NAME = 'library' 
            AND t2.COLUMN_NAME = 'username'
            AND t1.TABLE_SCHEMA = DATABASE()
        """)
        tables_with_library_and_username = [row[0] for row in cursor.fetchall()]

        # Identify tables that also have model_id column
        cursor.execute("""
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
        """)
        tables_with_model_id = [row[0] for row in cursor.fetchall()]

        print(
            f"Found {len(tables_with_library_and_username)} tables with both 'library' and 'username' columns: {', '.join(tables_with_library_and_username)}")
        print(
            f"Found {len(tables_with_model_id)} tables with 'library', 'username', and 'model_id' columns: {', '.join(tables_with_model_id)}")

        # Step 5: Start transaction for the update
        print("Starting transaction for library update...")
        cursor.execute("START TRANSACTION")

        # Step 6: Delete all rows with source_library, username (and model_id where applicable)
        # Organize tables by dependency
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

        for table in tables_in_order:
            if table in tables_with_model_id and model_id:
                print(
                    f"Deleting '{source_library}' records for user '{username}' and model_id '{model_id}' from {table}...")
                cursor.execute(f"DELETE FROM {table} WHERE library = %s AND username = %s AND model_id = %s",
                               (source_library, username, model_id))
            else:
                print(f"Deleting '{source_library}' records for user '{username}' from {table}...")
                cursor.execute(f"DELETE FROM {table} WHERE library = %s AND username = %s",
                               (source_library, username))

            deleted_count = cursor.rowcount
            print(f"Deleted {deleted_count} records from {table}")

        # Step 7: Update all rows with temp_library to source_library
        for table in tables_with_library_and_username:
            if table in tables_with_model_id and model_id:
                print(
                    f"Updating '{temp_library}' to '{source_library}' for user '{username}' and model_id '{model_id}' in {table}...")
                cursor.execute(
                    f"UPDATE {table} SET library = %s WHERE library = %s AND username = %s AND model_id = %s",
                    (source_library, temp_library, username, model_id))
            else:
                print(f"Updating '{temp_library}' to '{source_library}' for user '{username}' in {table}...")
                cursor.execute(f"UPDATE {table} SET library = %s WHERE library = %s AND username = %s",
                               (source_library, temp_library, username))

            updated_count = cursor.rowcount
            print(f"Updated {updated_count} records in {table}")

        # Step 8: Commit the transaction
        cursor.execute("COMMIT")
        print("Transaction committed successfully!")

        return {
            "success": True,
            "source_library": source_library,
            "username": username,
            "model": model,
            "model_id": model_id,
            "message": f"Library '{source_library}' for user '{username}'" +
                       (f" with model '{model}'" if model else "") +
                       " updated successfully with minimal downtime"
        }

    except Exception as e:
        # Roll back the transaction if anything goes wrong
        cursor.execute("ROLLBACK")
        print(f"Error during library update: {e}")
        import traceback
        traceback.print_exc()
        return {
            "success": False,
            "source_library": source_library,
            "username": username,
            "model": model,
            "model_id": model_id,
            "message": f"Error updating library: {str(e)}"
        }
    finally:
        # Close the connection
        if conn:
            conn.close()
            print("Database connection closed")

if __name__ == '__main__':

    conn, cursor = initialize_all_connection()

    # create_library_from_pdf_table('LEX')
    # create_library_from_pdf_table('RH')
    # create_library_from_pdf_table('LEX AND RH')
    update_library_with_minimal_downtime('LEX', 'all_users', 'openai')