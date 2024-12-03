from langchain_community.document_loaders import PyPDFLoader
import os


# from langchain.document_loaders import PyPDFLoader
from pdf2image import convert_from_path
import pytesseract
from langchain.schema import Document

def read_pdf_from_db_online(id, cursor):

    cursor.execute("SELECT file FROM pdfs WHERE id=%s", (id,))
    pdf = cursor.fetchone()
    if pdf is None:
        print(f"pdf with id {id} not found")
        return []
    pdf = pdf[0]


    #save the pdf in temp folder
    temp_pdf_dir = "temp"
    os.makedirs(temp_pdf_dir, exist_ok=True)
    temp_pdf_path = os.path.join(temp_pdf_dir, f"{id}.pdf")
    print('temp_pdf_path:', temp_pdf_path)

    # Write the binary content to a PDF file
    with open(temp_pdf_path, 'wb') as f:
        f.write(pdf)

    print('pdf saved')

    pages = read_pdf(temp_pdf_path)

    #delete the pdf
    os.remove(temp_pdf_path)

    return pages


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