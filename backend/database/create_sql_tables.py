from myUtils.connect_acad2 import reconnect_on_failure, initialize_all_connection

@reconnect_on_failure
def create_historic_table_if_not_exists(cursor):
    cursor.execute('''CREATE TABLE IF NOT EXISTS historic (
    id INT AUTO_INCREMENT PRIMARY KEY,
    username VARCHAR(255),
    action TEXT,
    detail TEXT,
    date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);''')

@reconnect_on_failure
def create_chat_history_table(cursor):
    # Create conversations table
    cursor.execute('''CREATE TABLE IF NOT EXISTS conversations (
        conversation_id INT AUTO_INCREMENT PRIMARY KEY,
        username VARCHAR(255),
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
    );''')

    # Create messages table
    cursor.execute('''CREATE TABLE IF NOT EXISTS messages (
        message_id INT AUTO_INCREMENT PRIMARY KEY,
        conversation_id INT,
        author_type ENUM('user', 'ai_robot'),
        content TEXT,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (conversation_id) REFERENCES conversations(conversation_id)
    );''')

    # Create indexes safely by first checking if they exist
    try:
        cursor.execute('''
            SELECT COUNT(1) IndexIsThere 
            FROM INFORMATION_SCHEMA.STATISTICS
            WHERE table_schema=DATABASE()
            AND table_name='conversations'
            AND index_name='idx_conversation_username';
        ''')
        index_exists = cursor.fetchone()[0]

        if not index_exists:
            cursor.execute('''
                CREATE INDEX idx_conversation_username 
                ON conversations(username)
            ''')
    except Exception as e:
        print(f"Warning: Could not create index idx_conversation_username: {e}")

    try:
        cursor.execute('''
            SELECT COUNT(1) IndexIsThere 
            FROM INFORMATION_SCHEMA.STATISTICS
            WHERE table_schema=DATABASE()
            AND table_name='messages'
            AND index_name='idx_message_conversation';
        ''')
        index_exists = cursor.fetchone()[0]

        if not index_exists:
            cursor.execute('''
                CREATE INDEX idx_message_conversation 
                ON messages(conversation_id)
            ''')
    except Exception as e:
        print(f"Warning: Could not create index idx_message_conversation: {e}")

def save_message_to_db(username, conversation_id, author_type, content, cursor):
    cursor.execute('''INSERT INTO messages (conversation_id, author_type, content)
VALUES (%s, %s, %s)''', (conversation_id, author_type, content))

    cursor.execute('''UPDATE conversations SET updated_at = CURRENT_TIMESTAMP WHERE conversation_id=%s''', (conversation_id,))

def start_new_conversation(username, cursor):
    cursor.execute('''INSERT INTO conversations (username) VALUES (%s)''', (username,))
    cursor.execute('''SELECT LAST_INSERT_ID()''')
    conversation_id = cursor.fetchone()[0]
    return conversation_id

@reconnect_on_failure
def create_big_chunks_table(cursor, replace_table=False):

    if replace_table:
        cursor.execute("DROP TABLE IF EXISTS big_chunks")

    # Create table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS big_chunks (
        id INT AUTO_INCREMENT PRIMARY KEY,
        source_id INT NOT NULL,
        page_number INT NOT NULL,
        page_content TEXT NOT NULL,
        three_page_content TEXT NOT NULL,
        library VARCHAR(255) NOT NULL,
        username VARCHAR(255) NOT NULL,
        UNIQUE KEY (source_id, page_number, library)
    ) ENGINE=InnoDB
    ''')

@reconnect_on_failure
def create_small_chunks_table(cursor):
    # Create table
    cursor.execute('''CREATE TABLE IF NOT EXISTS small_chunks (
    id INT PRIMARY KEY AUTO_INCREMENT,
    big_chunk_id INT NOT NULL,
    chunk_number INT NOT NULL,
    chunk_content TEXT NOT NULL,
    language_detected VARCHAR(255),
    en_chunk_content TEXT,
    library VARCHAR(255),
    username VARCHAR(255),
    UNIQUE(big_chunk_id, chunk_number, library, username)
);''')

@reconnect_on_failure
def create_source_docs_table_if_not_exists(cursor):
    cursor.execute("""CREATE TABLE IF NOT EXISTS source_docs (
    id INT AUTO_INCREMENT PRIMARY KEY,
    file LONGBLOB,
    date_detected TEXT,
    date_extracted TEXT,
    url TEXT,
    title TEXT,
    breadCrumb TEXT,
    checksum TEXT,
    library TEXT,
    username TEXT,
    doc_type TEXT,
    UNIQUE KEY unique_source (url(255), title(255), library(255), username(255))
);""")

@reconnect_on_failure
def create_user_libraries_table_if_not_exists(cursor):
    # print('creating user tables')
    cursor.execute('''CREATE TABLE IF NOT EXISTS user_libraries 
                                  (id INT AUTO_INCREMENT PRIMARY KEY, 
                                   username VARCHAR(255), 
                                   library_name VARCHAR(255), 
                                   library_summary TEXT,
                                   special_prompt TEXT,
                                   UNIQUE KEY unique_user_table (username, library_name))''')

    add_no_library_sql = '''INSERT INTO user_libraries (username, library_name, library_summary)
VALUES ('all_users', 'no_library', 'no_summary')
ON DUPLICATE KEY UPDATE
username = VALUES(username),
library_name = VALUES(library_name),
library_summary = VALUES(library_summary);'''

    cursor.execute(add_no_library_sql)

@reconnect_on_failure
def create_table_embeddings_models(cursor):

    # Create table
    cursor.execute('''CREATE TABLE IF NOT EXISTS embeddings_models (id integer primary key AUTO_INCREMENT, 
    model_name TEXT,
    language TEXT,
    dtype TEXT,
    UNIQUE(model_name, language)
    )''')

@reconnect_on_failure
def create_table_embeddings(cursor):
    # Create table
    cursor.execute('''CREATE TABLE IF NOT EXISTS embeddings (
    id INT PRIMARY KEY AUTO_INCREMENT,
    model_id INT,
    small_chunk_id INT,
    embedding LONGBLOB,
    library TEXT,
    username TEXT,
    UNIQUE(model_id, small_chunk_id, library, username)
);''')

@reconnect_on_failure
def create_users_table_if_not_exists(cursor):
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            username VARCHAR(255) PRIMARY KEY,
            password TEXT,
            session_token TEXT,
            openai_key TEXT,
            openai_key_status TEXT
        )
    ''')

@reconnect_on_failure
def create_faiss_table(cursor):
    cursor.execute("""
            CREATE TABLE IF NOT EXISTS faiss_indexes (
            id INTEGER PRIMARY KEY AUTO_INCREMENT, 
            model_id INTEGER, 
            embedding_ids TEXT, 
            faiss_index LONGBLOB,
            library VARCHAR(255),
            username VARCHAR(255),
            UNIQUE(model_id, library, username))
            """)

    create_faiss_index_parts = """
        CREATE TABLE IF NOT EXISTS faiss_index_parts (
            id INT AUTO_INCREMENT PRIMARY KEY,
            model_id INT,
            part_number INT,
            total_parts INT,
            index_part LONGBLOB,
            library VARCHAR(255),
            username VARCHAR(255),
            UNIQUE KEY unique_part (model_id, library, username, part_number)
        )
        """

    create_faiss_index_metadata = """
        CREATE TABLE IF NOT EXISTS faiss_index_metadata (
            id INT AUTO_INCREMENT PRIMARY KEY,
            model_id INT,
            embedding_ids JSON,
            library VARCHAR(255),
            username VARCHAR(255),
            UNIQUE KEY unique_metadata (model_id, library, username)
        )
        """

    cursor.execute(create_faiss_index_parts)
    cursor.execute(create_faiss_index_metadata)

def create_all_tables():
    conn, cursor = initialize_all_connection()

    print('creating tables')
    create_source_docs_table_if_not_exists(cursor=cursor)
    print('source_docs table created')
    create_user_libraries_table_if_not_exists(cursor=cursor)
    print('userTables table created')
    create_big_chunks_table(cursor=cursor)
    print('big chunks table created')
    create_table_embeddings_models(cursor=cursor)
    print('table embeddings models created')
    create_table_embeddings(cursor=cursor)
    print('table embeddings created')
    create_small_chunks_table(cursor=cursor)
    print('small chunks table created')
    create_users_table_if_not_exists(cursor=cursor)
    print('users table created')
    create_faiss_table(cursor=cursor)
    print('faiss table created')
    create_historic_table_if_not_exists(cursor=cursor)
    print('historic table created')
    create_chat_history_table(cursor=cursor)
    print('chat history table created')

    conn.commit()

