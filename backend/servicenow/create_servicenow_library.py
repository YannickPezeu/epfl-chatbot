import json
import asyncio
import hashlib
from datetime import datetime

# Import the necessary functions
from myUtils.connect_acad2 import initialize_all_connection
from myUtils.redisStateManager import RedisStateManager

# Import process_library_creation directly from the libraries module
from routes.libraries import process_library_creation

# Initialize Redis state manager
redis_state_manager = RedisStateManager()


async def process_kb_articles_and_create_library(json_file_path, library_name, username):
    """
    Process KB articles from a JSON file and then create a library using process_library_creation.

    Args:
        json_file_path (str): Path to the JSON file containing KB articles
        library_name (str): The name of the library to associate with the KB articles
        username (str): The username to associate with the KB articles
    """
    # Generate a task ID for tracking progress
    task_id = f"kb_import_{username}_{int(datetime.now().timestamp())}"

    # Connect to the database
    conn, cursor = initialize_all_connection()

    try:
        # Set initial state
        redis_state_manager.set_state(task_id, {"status": "Started", "progress": 0})

        # Update state to processing articles
        redis_state_manager.set_state(task_id, {"status": "Processing KB articles", "progress": 5})

        # Read the JSON file with KB articles
        with open(json_file_path, 'r', encoding='utf-8') as f:
            kb_articles = json.load(f)

        # Process each KB article
        for idx, article in enumerate(kb_articles):
            # Extract article data
            article_id = article.get('article_id', '')
            article_title = article.get('article_title', '')
            article_url = article.get('article_url', '')
            kb_id = article.get('kb_id', '')
            kb_title = article.get('kb_title', '')

            # Create file content (JSON representation of the article)
            file_content = json.dumps(article, ensure_ascii=False).encode('utf-8')

            # Generate a checksum for the file content
            checksum = hashlib.md5(file_content).hexdigest()

            # Create breadCrumb from KB title
            bread_crumb = f"ServiceNow KB > {kb_title}"

            # Use json as the doc_type
            doc_type = "json"

            # Current timestamp
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

            try:
                # Insert into source_docs
                cursor.execute(
                    """
                    INSERT IGNORE INTO source_docs 
                    (file, date_detected, date_extracted, url, title, breadCrumb, checksum, library, username, doc_type) 
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        file_content,
                        timestamp,  # date_detected
                        timestamp,  # date_extracted
                        article_url,
                        article_title,
                        bread_crumb,
                        checksum,
                        library_name,
                        username,
                        doc_type
                    )
                )
                print(f"Inserted article: {article_title}")

                # Update progress periodically
                if (idx + 1) % 5 == 0 or idx == len(kb_articles) - 1:
                    progress = 5 + int((idx + 1) / len(kb_articles) * 5)  # Progress from 5% to 10%
                    redis_state_manager.set_state(task_id,
                                                  {"status": f"Processing KB articles ({idx + 1}/{len(kb_articles)})",
                                                   "progress": progress})

            except Exception as e:
                print(f"Error inserting article {article_title}: {e}")

        # Commit changes after inserting all articles
        conn.commit()
        cursor.close()
        conn.close()

        print(f"Processed {len(kb_articles)} KB articles")

        # Now continue with library creation
        # For library creation, we need an empty list for temp_file_paths since we're not using files
        temp_file_paths = []

        # Call the process_library_creation function that's imported from libraries.py
        await process_library_creation(
            task_id=task_id,
            username=username,
            library_name=library_name,
            temp_file_paths=temp_file_paths,
            model_name='rcp',
            special_prompt=None,
            doc_type='json'  # Specify json as the doc_type
        )

        return task_id

    except Exception as e:
        print(f"Error in process_kb_articles_and_create_library: {e}")
        redis_state_manager.set_state(task_id, {"status": "Error", "message": str(e)})

        # Make sure to close the database connection in case of error
        if 'conn' in locals() and conn:
            cursor.close()
            conn.close()

        return task_id


async def main(json_file_paths=None, library_name='ServiceNow_KB', username='servicenow_user'):
    """Main function to run the process"""

    if json_file_paths is None:
        json_file_paths = ['kb_articles_3969aa4edbb08cd079f593c8f49619a0_20250501_164724.json',
                       'kb_articles_90d645064fd6b6009d2bdf601310c7a2_20250501_164730.json',
                       'kb_articles_7243ab6347332100158b949b6c9a7194_20250501_164725.json',
                       'kb_articles_b5e390e3dbb430d031895c88f49619ed_20250501_164731.json',
                       'kb_articles_bb4ef5394f8a72009d2bdf601310c7e9_20250501_164741.json'
                       ]


    for json_file_path in json_file_paths:
        print('processing {}'.format(json_file_path))
        task_id = await process_kb_articles_and_create_library(
            json_file_path=json_file_path,
            library_name=library_name,
            username=username
        )

        print(f"Process completed with task ID: {task_id} for {json_file_path}")


if __name__ == "__main__":
    # Run the async main function
    json_file_paths = ['kb_articles_7243ab6347332100158b949b6c9a7194_20250515_123614.json']
    asyncio.run(main(json_file_paths=json_file_paths, library_name='servicenow_finance', username='servicenow_user'))