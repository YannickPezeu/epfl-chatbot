import json

from myUtils.connect_acad import initialize_all_connection
from myUtils.read_pdf_online import read_pdf_from_db_online
from myUtils.ask_chatGPT import ask_chatGPT
import dotenv
import os

dotenv.load_dotenv(dotenv_path='C:\Dev\ChatFinderBackEnd\.env')
openai_key = os.getenv('OPENAI_KEY')

def filter_pdfs(cursor, openai_key):
    cursor.execute("SELECT id, title FROM pdfs where library = 'LEX AND RH'")
    pdfs = cursor.fetchall()
    for pdf in pdfs:
        pdf_id, title = pdf
        if title.endswith('.pdf') or title.endswith('.docx'):
            continue
        else:
            print('reading pdf', title)
            pages = read_pdf_from_db_online(pdf_id, cursor)
            if len(pages) > 2:
                continue
            else:
                total_content = ''
                for page in pages:
                    total_content += page.page_content

                prompt = f'''I scraped a website and I am building a search engine on it. 
                I need your help to filter pages that I should keep and pages that I should discard.
                this is the content of a page: {total_content}
                \n\n 
                if you think this page is relevant, answer {{"keep": true}}
                else answer {{"keep": false}}
                it is important you answer is a json object with the key "keep" and the value either true or false.
                '''

                answer = ask_chatGPT(prompt, openai_key=openai_key)
                answer = answer.choices[0].message.content
                print('answer:', answer)
                response = json.loads(answer)
                response = response['keep']
                print('response:', response)

                if not response:
                    delete_pdf(cursor, pdf_id)


def delete_pdf(cursor, pdf_id):
    print(f"Deleting pdf with id {pdf_id}")
    cursor.execute("DELETE FROM pdfs WHERE id=%s", (pdf_id,))
    # get big_chunks ids corresponding to the pdf_id
    cursor.execute("SELECT id FROM big_chunks WHERE pdf_id=%s", (pdf_id,))
    big_chunks_ids = cursor.fetchall()
    big_chunks_ids = [big_chunk_id[0] for big_chunk_id in big_chunks_ids]

    small_chunk_ids = []
    for big_chunk_id in big_chunks_ids:
        cursor.execute("SELECT id FROM small_chunks WHERE big_chunk_id=%s", (big_chunk_id,))
        small_chunk_ids_current = cursor.fetchall()
        small_chunk_ids_current = [small_chunk_id[0] for small_chunk_id in small_chunk_ids_current]
        small_chunk_ids += small_chunk_ids_current


    cursor.execute("DELETE FROM big_chunks WHERE pdf_id=%s", (pdf_id,))
    for big_chunk_id in big_chunks_ids:
        cursor.execute("DELETE FROM small_chunks WHERE big_chunk_id=%s", (big_chunk_id,))

    for small_chunk_id in small_chunk_ids:
        cursor.execute("DELETE FROM embeddings WHERE small_chunk_id=%s", (small_chunk_id,))


if __name__ == '__main__':
    conn, cursor = initialize_all_connection()
    openai_key = os.getenv('OPENAI_KEY')
    filter_pdfs(cursor, openai_key)



