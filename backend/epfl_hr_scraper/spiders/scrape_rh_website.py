import datetime
import hashlib
import json
import re
import sqlite3
import time

import scrapy
from urllib.parse import urlparse

from bs4 import BeautifulSoup
import sys
import os

# Current file's directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Directory two levels up from the current file
root_project_dir = os.path.abspath(os.path.join(current_dir, '../../..'))
print('parent_dir', root_project_dir)

# Add the parent directory to sys.path
sys.path.insert(0, root_project_dir)
from myUtils.textExtractor import read_docx_pdf
from myUtils.urlToPDF import UrlToPdf
import pickle as pkl
from docx2pdf import convert
from myUtils.connect_acad2 import initialize_all_connection

class HrSpider(scrapy.Spider):
    name = 'hr_spider'
    allowed_domains = ['www.epfl.ch']
    start_urls = ['https://www.epfl.ch/campus/services/human-resources/']
    database_folder = os.path.join(root_project_dir, 'data', 'HR')
    database_name = 'HR'

    # create folder
    if not os.path.exists(database_folder):
        os.makedirs(database_folder)


    # Connect to database
    conn, cursor = initialize_all_connection()
    
    def generate_checksum(self, content):
        """Generate a SHA-256 checksum for the given content."""
        return hashlib.sha256(content.encode('utf-8')).hexdigest()

    def has_page_changed(self, current_checksum, stored_checksum):
        """Check if the webpage has changed compared to a stored checksum."""
        return current_checksum != stored_checksum, current_checksum

    def has_page_changed_from_db(self, response):
        new_checksum = self.generate_checksum(response.text)
        self.cursor.execute("SELECT checksum FROM pdfs WHERE url=%s", (response.url,))
        stored_checksum = self.cursor.fetchone()
        if stored_checksum:
            has_changed, new_checksum = self.has_page_changed(new_checksum, stored_checksum[0])
            return has_changed, new_checksum
        else:
            return True, new_checksum

    def has_page_changed_from_file(self, response, details_path):
        if os.path.exists(details_path):
            with open(details_path, 'r', encoding='utf-8') as f:
                stored_details = json.load(f)
            content = response.text
            current_checksum = self.generate_checksum(content)
            has_changed, new_checksum = self.has_page_changed(current_checksum, stored_details.get('checksum'))
            return has_changed, new_checksum
        else:
            content = response.text
            current_checksum = self.generate_checksum(content)
            return True, current_checksum

    def scrape_all_documents_in_page(self, response):
        # Find links to documents and save them
        document_links = response.css(
            'a[href$=".pdf"]::attr(href), '
            'a[href$=".docx"]::attr(href)'
            ).getall()

        print('document_links', document_links)
        for link in document_links:
            absolute_url = response.urljoin(link)
            yield scrapy.Request(absolute_url, callback=self.save_document)

    def find_and_follow_links(self, response):
        # Find and follow links within the human-resources subtree
        all_links = response.css('a::attr(href)').getall()
        for link in all_links:
            absolute_url = response.urljoin(link)
            if self.is_within_hr_subtree(absolute_url):
                yield response.follow(absolute_url, self.parse)

    def parse(self, response):
        # Save the current page's content
        print('visiting url', response.url)
        if '.xls' in response.url or '.xlsx' in response.url or '.doc' in response.url or '.docx' in response.url or '.pdf' in response.url:
            return

        filename = self.get_filename(response.url)

        yield from self.scrape_all_documents_in_page(response)

        yield from self.find_and_follow_links(response)

        self.save_url_to_pdf(response)

    def save_document(self, response):
        print('calling save document on', response.url)
        filename = 'noFileNameDetected.pdf'
        urlSplit = response.url.split('/')
        for part in reversed(urlSplit):
            if part != '':
                filename = part
                break

        temp_folder = os.path.join(self.database_folder, 'temp_downloads')
        destination_path = os.path.join(temp_folder, filename)

        if not os.path.exists(temp_folder):
            os.makedirs(temp_folder)

        try:
            if filename.endswith('.docx'):
                with open(destination_path, 'wb') as f:
                    f.write(response.body)
                destination_path_pdf = destination_path.replace('.docx', '.pdf')
                convert(destination_path, destination_path_pdf)
                destination_path = destination_path_pdf
            else:
                print('destination_path', destination_path)
                with open(destination_path, 'wb') as f:
                    f.write(response.body)

            # read the pdf file just saved
            with open(destination_path, 'rb') as f:
                mypdf = f.read()

            date_detected = self.extract_date_from_url(response.url)

            breadCrumb = ''

            # insert into database with ON DUPLICATE KEY UPDATE
            self.cursor.execute("""
                INSERT INTO pdfs 
                    (file, date_detected, date_extracted, url, title, breadCrumb, checksum, library, username) 
                VALUES 
                    (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE
                    file = VALUES(file),
                    date_detected = VALUES(date_detected),
                    date_extracted = VALUES(date_extracted),
                    breadCrumb = VALUES(breadCrumb),
                    checksum = VALUES(checksum)
                """,
                                (mypdf, date_detected, datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                 response.url, filename, breadCrumb, '', 'RH', 'all_users'))

            self.conn.commit()

        except Exception as e:
            print(f"Error saving document: {e}")
            raise e

        finally:
            # Add a small delay before cleaning
            time.sleep(1)

            # clean temp folder
            try:
                for file in os.listdir(temp_folder):
                    file_path = os.path.join(temp_folder, file)
                    try:
                        if os.path.exists(file_path):
                            os.remove(file_path)
                    except Exception as e:
                        print(f"Error removing temp file {file_path}: {e}")
            except Exception as e:
                print(f"Error cleaning temp folder: {e}")

    def extract_date_from_url(self, url):
        date_pattern = r'/(\d{4})/(\d{2})/'

        # Search for the pattern in the URL
        match = re.search(date_pattern, url)
        if match:
            year, month = match.groups()
            date = f"{year}-{month}"
        else:
            date = 'unknown'

        return date
    def get_filename(self, url):
        # take the last part of the url that is not empty
        filename = 'noFileNameDetected.pdf'
        urlSplit = url.split('/')
        for part in reversed(urlSplit):
            if part != '':
                filename = part
                break

        return filename

    def get_breadcrumbs(self, soup):
        # Find the breadcrumb <ol> element
        breadcrumbs_ol = soup.find('ol', class_='breadcrumb')
        if not breadcrumbs_ol:
            return ''

        # Extract text from each <li> within the breadcrumb <ol>
        breadcrumbs = []
        for li in breadcrumbs_ol.find_all('li',
                                          recursive=False):  # `recursive=False` ensures only direct <li> children are considered
            # Extract and clean the text from each <li>
            breadcrumb_text = li.get_text(strip=True)
            breadcrumbs.append(breadcrumb_text)

        return breadcrumbs

    def is_within_hr_subtree(self, url):
        # Checks if the URL is within the human-resources subtree
        parsed_url = urlparse(url)
        return parsed_url.netloc == 'www.epfl.ch' and '/campus/services/human-resources/' in parsed_url.path

    def extract_text_with_formatting(self, soup_element, url):
        text = ''
        for element in soup_element.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li']):
            text += '\n\n' + ''.join(element.stripped_strings)
        text = text.strip()
        final_results=[{
            'page_content': text,
            'metadata': {
                'page': '0',  # Sheet number (1-indexed)
                'source': url
            }
        }]
        return final_results

    def save_url_to_pdf(self, response, lex_number=None, url=None, title=None):
        # Save the current page's content
        print('visiting url', response.url)
        if '.xls' in response.url or '.xlsx' in response.url or '.doc' in response.url or '.docx' in response.url or '.pdf' in response.url:
            return

        if not title:
            title = self.get_filename(response.url)
        else:
            if lex_number:
                title = str(lex_number) + '_' + title
            else:
                title = title

        # check if checkSum has changed
        has_changed, new_checksum = self.has_page_changed_from_db(response)
        if not has_changed:
            return

        # save page as pdf
        pdf_files = UrlToPdf([response.url]).main()

        # Get the PDF data from the BytesIO object
        pdf_data = pdf_files[0].getvalue()

        # Verify it's a PDF (optional but recommended)
        if not pdf_data.startswith(b'%PDF-'):
            print(f"Warning: Generated file does not appear to be a valid PDF")
            return

        # get date_detected
        date_detected = self.extract_date_from_url(response.url)

        # get BreadCrumbs
        soup = BeautifulSoup(response.text, 'html.parser')
        breadcrumb_list = self.get_breadcrumbs(soup)
        breadCrumb = str(breadcrumb_list)

        # Use ON DUPLICATE KEY UPDATE for MySQL
        self.cursor.execute("""
            INSERT INTO pdfs 
                (file, date_detected, date_extracted, url, title, breadCrumb, checksum, library, username)
            VALUES 
                (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE
                file = VALUES(file),
                date_detected = VALUES(date_detected),
                date_extracted = VALUES(date_extracted),
                breadCrumb = VALUES(breadCrumb),
                checksum = VALUES(checksum)
            """,
                            (pdf_data, date_detected, datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                             response.url, title, breadCrumb, new_checksum, 'RH', 'all_users'))

        self.conn.commit()