import json

import tiktoken
from langchain_community.document_loaders import PyPDFLoader
import os


# from langchain.document_loaders import PyPDFLoader
from pdf2image import convert_from_path
import pytesseract
from langchain.schema import Document
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)



def read_source_doc_from_db_online(id, cursor):
    cursor.execute("SELECT file, doc_type FROM source_docs WHERE id=%s", (id,))
    result = cursor.fetchone()
    if result is None:
        logger.info(f"Document with id {id} not found")
        return []
    content = result[0]
    doc_type = result[1]

    if doc_type == "pdf":
        # Save the pdf in temp folder
        temp_pdf_dir = "temp"
        os.makedirs(temp_pdf_dir, exist_ok=True)
        temp_pdf_path = os.path.join(temp_pdf_dir, f"{id}.pdf")
        print('temp_pdf_path:', temp_pdf_path)

        # Write the binary content to a PDF file
        with open(temp_pdf_path, 'wb') as f:
            f.write(content)

        logger.info('pdf saved')

        pages = read_pdf(temp_pdf_path)

        # Delete the pdf
        os.remove(temp_pdf_path)

        return pages

    elif doc_type == "text":
        try:
            # Handle text content (from local text files)
            if isinstance(content, bytes):
                raw_text = content.decode('utf-8')
            else:
                raw_text = content

            # Use tiktoken to split the text into chunks of approximately 1000 tokens
            tiktoken_encoding = tiktoken.get_encoding("cl100k_base")
            tokens = tiktoken_encoding.encode(raw_text)

            # Split into chunks of 1000 tokens
            pages_tokenized = [tokens[i:i + 1000] for i in range(0, len(tokens), 1000)]

            # Create Document objects for each chunk, similar to the PDF format
            pages = []
            for i, page_tokens in enumerate(pages_tokenized, start=1):
                page_content = tiktoken_encoding.decode(page_tokens)
                # Create a Document object with the same structure as those returned by read_pdf
                page_doc = Document(page_content=page_content, metadata={'page': i})
                pages.append(page_doc)

            return pages

        except Exception as e:
            logger.error(f"Error processing text document (id={id}): {e}")
            return []

    elif doc_type == "json":
        try:
            # Parse binary content as JSON
            json_content = json.loads(content)

            # Extract text from the JSON article
            article_title = json_content.get('article_title', '')
            article_content = json_content.get('content', '')
            article_url = json_content.get('article_url', '')
            kb_id = json_content.get('kb_id', '')
            kb_title = json_content.get('kb_title', '')

            # Create a formatted text with all the relevant article information
            raw_text = f"Title: {article_title}\n\n"
            raw_text += f"KB: {kb_title} (ID: {kb_id})\n"
            raw_text += f"URL: {article_url}\n\n"
            raw_text += article_content

            # If no content was found, use the entire JSON
            if not article_content:
                raw_text = json.dumps(json_content, indent=2)

            # Use tiktoken to split the text into chunks of approximately 1000 tokens
            tiktoken_encoding = tiktoken.get_encoding("cl100k_base")
            tokens = tiktoken_encoding.encode(raw_text)

            # Split into chunks of 1000 tokens
            pages_tokenized = [tokens[i:i + 1000] for i in range(0, len(tokens), 1000)]

            # Create Document objects for each chunk, similar to the PDF format
            pages = []
            for i, page_tokens in enumerate(pages_tokenized, start=1):
                page_content = tiktoken_encoding.decode(page_tokens)
                # Create a Document object with the same structure as those returned by read_pdf
                page_doc = Document(page_content=page_content, metadata={'page': i})
                pages.append(page_doc)

            return pages

        except Exception as e:
            logger.error(f"Error processing JSON document (id={id}): {e}")
            return []
    else:
        logger.error('doc_type not recognized')
        raise Exception(f"doc_type '{doc_type}' not recognized")

def read_pdf(pdf_path):
    '''returns a list of pages from the pdf file
    each page has two props: page_number and page_content
    '''
    real_pages = []

    # First attempt: Use PyPDFLoader
    try:
        loader = PyPDFLoader(pdf_path)
        pages = loader.load_and_split()

        last_page_index = -1
        for page in pages:
            print(f"Page {page.metadata['page']}: {page.page_content[:100]}...")
            print('real_pages:', len(real_pages))
            page_index = page.metadata['page']
            if page_index == last_page_index:
                real_pages[-1].page_content += page.page_content
            else:
                real_pages.append(page)
                last_page_index = page_index

    except Exception as e:
        print(f'Error reading pdf {pdf_path} with PyPDFLoader: {e}')

    # If PyPDFLoader failed or returned empty results, try OCR
    if not real_pages:
        print('PyPDFLoader failed or returned no pages. Attempting to read PDF using OCR...')
        try:
            ocr_text = ocr_scanned_pdf(pdf_path)
            # Split the OCR text into pages
            ocr_pages = ocr_text.split('--- Page')
            for i, page_content in enumerate(ocr_pages[1:], start=1):  # Skip the first split as it's empty
                page_content = page_content.strip()
                if page_content:
                    real_pages.append(Document(page_content=page_content, metadata={'page': i}))
        except Exception as ocr_error:
            print(f'Error reading pdf {pdf_path} with OCR: {ocr_error}')

    return real_pages

def ocr_scanned_pdf(pdf_path):
    text = ""
    images = convert_from_path(pdf_path)
    for i, image in enumerate(images):
        page_text = pytesseract.image_to_string(image)
        print(f"OCR result for page {i+1}: {page_text[:100]}...")
        text += f"--- Page {i+1} ---\n{page_text}\n\n"
    return text


if __name__ == '__main__':
    pdf_path = '''C:/Users/pezeu/Documents/test.pdf'''
    pages = read_pdf(pdf_path)
    print(pages)