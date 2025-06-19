import os

import tiktoken
from pydantic import BaseModel
import sys
current_folder = os.path.dirname(os.path.abspath(__file__))
root_folder = os.path.abspath(os.path.abspath(os.path.join(current_folder, '..')))
utils_folder = os.path.join(root_folder, 'myUtils')
data_folder = os.path.join(root_folder, 'data')
sys.path.append(root_folder)
sys.path.append(utils_folder)


from myUtils.connect_acad2 import reconnect_on_failure
print('root_folder:', root_folder)


from myUtils.read_pdf_online import read_source_doc_from_db_online


class BigChunk(BaseModel):
    source_doc_id: int
    page_number: int
    page_content: str
    three_page_content: str

def create_big_chunks_from_source_doc(cursor, source_doc_id):
    print('create_big_chunks_from_pdf')
    pages = read_source_doc_from_db_online(source_doc_id, cursor)
    print('pages', pages)

    big_chunks = []
    for i, page in enumerate(pages):
        if len(pages) == 1:
            three_page_content = page.page_content
        elif len(pages) == 2:
            three_page_content = pages[0].page_content + page.page_content
        elif i == 0:
            three_page_content = page.page_content + pages[i+1].page_content
        elif i == len(pages) - 1:
            three_page_content = pages[i-1].page_content + page.page_content
        else:
            three_page_content = pages[i-1].page_content + page.page_content + pages[i+1].page_content
        my_big_chunk = BigChunk(
            source_doc_id=source_doc_id,
            page_number=page.metadata['page'],
            page_content=page.page_content,
            three_page_content=three_page_content
        )
        big_chunks.append(my_big_chunk)
    return big_chunks


def create_big_chunk_from_json(source_doc_id, raw_text):
    print('create_big_chunk_from_raw_text')
    # consider a page is 1000 tokens

    tiktoken_encoding = tiktoken.get_encoding("cl100k_base")
    tokens = tiktoken_encoding.encode(raw_text)

    pages_tokenized = [tokens[i:i + 1000] for i in range(0, len(tokens), 1000)]
    pages_in_text = [tiktoken_encoding.decode(pages_tokenized[i]) for i in range(len(pages_tokenized))]

    big_chunks = []
    for i, page in enumerate(pages_in_text):
        if len(pages_in_text) == 1:
            three_page_content = page
        elif len(pages_in_text) == 2:
            if i == 0:
                three_page_content = page + pages_in_text[1]
            else:
                three_page_content = pages_in_text[0] + page
        elif i == 0:
            three_page_content = page + pages_in_text[i + 1]
        elif i == len(pages_in_text) - 1:
            three_page_content = pages_in_text[i - 1] + page
        else:
            three_page_content = pages_in_text[i - 1] + page + pages_in_text[i + 1]

        my_big_chunk = BigChunk(
            source_doc_id=source_doc_id,
            page_number=i + 1,  # Start page number from 1 instead of 0
            page_content=page,
            three_page_content=three_page_content
        )
        big_chunks.append(my_big_chunk)

    return big_chunks









@reconnect_on_failure
def insert_big_chunks_into_db(library, username, cursor, conn):
    # Connect to MariaDB

    # Fetch all source_docs
    cursor.execute("SELECT id, file FROM source_docs WHERE library=%s AND username=%s", (library, username))
    source_docs = cursor.fetchall()

    print('found {} source_docs'.format(len(source_docs)))

    for source_doc in source_docs:
        source_doc_id, file = source_doc
        print(f"Processing PDF {source_doc_id}...")
        big_chunks = create_big_chunks_from_source_doc(cursor, source_doc_id)  # Assuming this function is defined elsewhere
        for big_chunk in big_chunks:
            print(f"Inserting big chunk for page {big_chunk.page_number}...")
            cursor.execute(
                """
                INSERT IGNORE INTO big_chunks 
                (source_doc_id, page_number, page_content, three_page_content, library, username) 
                VALUES (%s, %s, %s, %s, %s, %s)
                """,
                (big_chunk.source_doc_id, big_chunk.page_number, big_chunk.page_content, big_chunk.three_page_content, library, username)
            )
        conn.commit()


if __name__ == '__main__':


    print('inserting big chunks into db')
    insert_big_chunks_into_db('test')



