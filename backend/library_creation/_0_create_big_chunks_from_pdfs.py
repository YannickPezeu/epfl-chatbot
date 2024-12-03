import os
from pydantic import BaseModel
import sys
current_folder = os.path.dirname(os.path.abspath(__file__))
root_folder = os.path.abspath(os.path.abspath(os.path.join(current_folder, '..')))
utils_folder = os.path.join(root_folder, 'myUtils')
data_folder = os.path.join(root_folder, 'data')
sys.path.append(root_folder)
sys.path.append(utils_folder)


from myUtils.connect_acad import reconnect_on_failure
print('root_folder:', root_folder)


from myUtils.read_pdf_online import read_pdf_from_db_online


class BigChunk(BaseModel):
    pdf_id: int
    page_number: int
    page_content: str
    three_page_content: str

def create_big_chunks_from_pdf(cursor, pdf_id):
    print('create_big_chunks_from_pdf')
    pages = read_pdf_from_db_online(pdf_id, cursor)
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
            pdf_id=pdf_id,
            page_number=page.metadata['page'],
            page_content=page.page_content,
            three_page_content=three_page_content
        )
        big_chunks.append(my_big_chunk)
    return big_chunks




@reconnect_on_failure
def insert_big_chunks_into_db(library, username, cursor):
    # Connect to MariaDB

    # Fetch all PDFs
    cursor.execute("SELECT id, file FROM pdfs WHERE library=%s AND username=%s", (library, username))
    pdfs = cursor.fetchall()

    for pdf in pdfs:
        pdf_id, file = pdf
        print(f"Processing PDF {pdf_id}...")
        big_chunks = create_big_chunks_from_pdf(cursor, pdf_id)  # Assuming this function is defined elsewhere
        for big_chunk in big_chunks:
            print(f"Inserting big chunk for page {big_chunk.page_number}...")
            cursor.execute(
                """
                INSERT IGNORE INTO big_chunks 
                (pdf_id, page_number, page_content, three_page_content, library, username) 
                VALUES (%s, %s, %s, %s, %s, %s)
                """,
                (big_chunk.pdf_id, big_chunk.page_number, big_chunk.page_content, big_chunk.three_page_content, library, username)
            )


if __name__ == '__main__':


    print('inserting big chunks into db')
    insert_big_chunks_into_db('test')



