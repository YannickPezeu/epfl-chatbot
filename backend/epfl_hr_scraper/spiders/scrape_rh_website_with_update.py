import datetime
import hashlib
import json
import re
import sqlite3
import time
import traceback

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
    name = 'hr_spider_update'
    allowed_domains = ['www.epfl.ch']
    start_urls = ['https://www.epfl.ch/campus/services/human-resources/']
    database_folder = os.path.join(root_project_dir, 'data', 'HR')
    database_name = 'HR'

    # create folder
    if not os.path.exists(database_folder):
        os.makedirs(database_folder)

    def __init__(self, *args, **kwargs):
        super(HrSpider, self).__init__(*args, **kwargs)
        # Connect to database
        self.conn, self.cursor = initialize_all_connection()

        # Track URLs found during this crawl - initialize in __init__ to ensure it's per-instance
        self.found_urls = set()

        # Debug tracking
        self.pages_visited = 0
        self.docs_processed = 0

        print("Spider initialized with fresh database connection and empty found_urls set")

    def generate_checksum(self, content):
        """Generate a SHA-256 checksum for the given content."""
        return hashlib.sha256(content.encode('utf-8')).hexdigest()

    def has_page_changed(self, current_checksum, stored_checksum):
        """Check if the webpage has changed compared to a stored checksum."""
        return current_checksum != stored_checksum, current_checksum

    def has_page_changed_from_db(self, response):
        new_checksum = self.generate_checksum(response.text)
        self.cursor.execute("SELECT checksum FROM source_docs WHERE url=%s", (response.url,))
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

        print(f'Found {len(document_links)} document links on page {response.url}')
        for link in document_links:
            absolute_url = response.urljoin(link)
            # Normalize and add to found URLs - we add it here AND in save_document
            # to ensure it's tracked even if the request fails
            normalized_url = self.normalize_url(absolute_url)
            self.found_urls.add(normalized_url)
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
        self.pages_visited += 1
        print(f'Visiting URL #{self.pages_visited}: {response.url}')

        # Skip binary files
        if '.xls' in response.url or '.xlsx' in response.url or '.doc' in response.url or '.docx' in response.url or '.pdf' in response.url:
            return

        # Normalize URL to ensure consistent comparison
        normalized_url = self.normalize_url(response.url)

        # Add current URL to found URLs
        self.found_urls.add(normalized_url)
        print(f'Added to found_urls: {normalized_url}')

        filename = self.get_filename(response.url)

        # First process documents on the page
        yield from self.scrape_all_documents_in_page(response)

        # Then follow links to other pages
        yield from self.find_and_follow_links(response)

        # Save the page itself as PDF
        self.save_url_to_pdf(response)

    def normalize_url(self, url):
        """Normalize URL to ensure consistent comparison."""
        # Remove trailing slashes
        normalized = url.rstrip('/')
        # Ensure consistent protocol (using https)
        if normalized.startswith('http:'):
            normalized = 'https:' + normalized[5:]
        return normalized

    def save_document(self, response):
        self.docs_processed += 1
        print(f'Processing document #{self.docs_processed}: {response.url}')

        # Normalize URL and add to found URLs set
        normalized_url = self.normalize_url(response.url)
        self.found_urls.add(normalized_url)
        print(f'Added document to found_urls: {normalized_url}')

        # Check if document already exists in database (to determine if this is an update)
        self.cursor.execute("SELECT id FROM source_docs WHERE url=%s", (response.url,))
        existing_doc = self.cursor.fetchone()
        operation_type = "Updating" if existing_doc else "Adding new"
        print(f"{operation_type} document: {response.url}")

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
            # Download the document
            with open(destination_path, 'wb') as f:
                f.write(response.body)

            # If it's a DOCX, convert to PDF
            if filename.endswith('.docx'):
                try:
                    print(f"Converting DOCX to PDF: {destination_path}")
                    destination_path_pdf = destination_path.replace('.docx', '.pdf')
                    convert(destination_path, destination_path_pdf)
                    destination_path = destination_path_pdf
                except Exception as convert_error:
                    import traceback
                    error_traceback = traceback.format_exc()
                    print(f"======= DOCX CONVERSION ERROR =======")
                    print(f"Error converting DOCX to PDF: {convert_error}")
                    print(f"Traceback:\n{error_traceback}")
                    print(f"======================================")
                    # Skip this document if conversion fails
                    print(f"Skipping {response.url} because DOCX conversion failed")
                    return  # Exit the function early

            # Read the file we just saved
            try:
                with open(destination_path, 'rb') as f:
                    mypdf = f.read()

                # Verify we have valid PDF data
                if not mypdf.startswith(b'%PDF-') and destination_path.endswith('.pdf'):
                    raise ValueError(f"File {destination_path} is not a valid PDF")
            except Exception as file_error:
                import traceback
                error_traceback = traceback.format_exc()
                print(f"======= FILE READING ERROR =======")
                print(f"Error reading file {destination_path}: {file_error}")
                print(f"Traceback:\n{error_traceback}")
                print(f"==================================")
                # Skip this document if reading fails
                print(f"Skipping {response.url} because file reading failed")
                return  # Exit the function early

            date_detected = self.extract_date_from_url(response.url)
            breadCrumb = ''

            # Generate checksum for the document
            doc_checksum = hashlib.sha256(mypdf).hexdigest()

            # Update or insert the document in the database
            try:
                if existing_doc:
                    # Update existing document
                    self.cursor.execute("""
                        UPDATE source_docs SET
                            file = %s,
                            date_detected = %s,
                            date_extracted = %s,
                            breadCrumb = %s,
                            checksum = %s
                        WHERE url = %s
                        """,
                                        (mypdf, date_detected, datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                         breadCrumb, doc_checksum, response.url))
                else:
                    # Insert new document
                    self.cursor.execute("""
                        INSERT INTO source_docs 
                            (file, date_detected, date_extracted, url, title, breadCrumb, checksum, library, username) 
                        VALUES 
                            (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                        """,
                                        (mypdf, date_detected, datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                         response.url, filename, breadCrumb, doc_checksum, 'RH', 'all_users'))

                self.conn.commit()
                print(f"Successfully {operation_type.lower()}d document in database: {response.url}")
            except Exception as db_error:
                import traceback
                error_traceback = traceback.format_exc()
                print(f"======= DATABASE ERROR =======")
                print(f"Database error for {response.url}: {db_error}")
                print(f"Traceback:\n{error_traceback}")
                print(f"=============================")
                try:
                    self.conn.rollback()
                except:
                    pass

        except Exception as save_error:
            import traceback
            error_traceback = traceback.format_exc()
            print(f"======= DOCUMENT PROCESSING ERROR =======")
            print(f"Critical error saving document {response.url}: {save_error}")
            print(f"Traceback:\n{error_traceback}")
            print(f"=========================================")

            # Make sure the URL is still tracked as found even if saving failed
            self.found_urls.add(normalized_url)

            # Try to update just the timestamp if possible for existing documents
            if existing_doc:
                try:
                    self.cursor.execute("""
                        UPDATE source_docs SET
                            date_extracted = %s
                        WHERE url = %s
                        """,
                                        (datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), response.url))
                    self.conn.commit()
                    print(f"Updated timestamp for {response.url} despite error")
                except Exception as timestamp_error:
                    print(f"Could not update timestamp: {timestamp_error}")
                    try:
                        self.conn.rollback()
                    except:
                        pass

        except Exception as save_error:
            print(f"Critical error saving document {response.url}: {save_error}")

            # Make sure the URL is still tracked as found even if saving failed
            self.found_urls.add(normalized_url)

            # Try to update just the timestamp if possible for existing documents
            if existing_doc:
                try:
                    self.cursor.execute("""
                        UPDATE source_docs SET
                            date_extracted = %s
                        WHERE url = %s
                        """,
                                        (datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), response.url))
                    self.conn.commit()
                    print(f"Updated timestamp for {response.url} despite error")
                except Exception as timestamp_error:
                    print(f"Could not update timestamp: {timestamp_error}")
                    try:
                        self.conn.rollback()
                    except:
                        pass

        finally:
            # Add a small delay before cleaning
            time.sleep(1)

            # Clean temp folder
            try:
                for file in os.listdir(temp_folder):
                    file_path = os.path.join(temp_folder, file)
                    try:
                        if os.path.exists(file_path):
                            os.remove(file_path)
                    except Exception as clean_error:
                        print(f"Non-critical error removing temp file {file_path}: {clean_error}")
            except Exception as folder_error:
                print(f"Non-critical error cleaning temp folder: {folder_error}")

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
        final_results = [{
            'page_content': text,
            'metadata': {
                'page': '0',  # Sheet number (1-indexed)
                'source': url
            }
        }]
        return final_results

    def save_url_to_pdf(self, response, lex_number=None, url=None, title=None):
        # Save the current page's content
        print(f'Converting to PDF: {response.url}')

        # Skip binary files
        if '.xls' in response.url or '.xlsx' in response.url or '.doc' in response.url or '.docx' in response.url or '.pdf' in response.url:
            return

        # Set the title
        if not title:
            title = self.get_filename(response.url)
        else:
            if lex_number:
                title = str(lex_number) + '_' + title
            else:
                title = title

        # Normalize URL for consistent comparison
        normalized_url = self.normalize_url(response.url)

        # Make sure this URL is in our found set
        self.found_urls.add(normalized_url)

        # Check if checkSum has changed - ALWAYS update date regardless of content change
        has_changed, new_checksum = self.has_page_changed_from_db(response)

        # If page hasn't changed, still store the URL but skip PDF generation
        if not has_changed:
            print(f"Page hasn't changed, updating timestamp only: {response.url}")

            # Update the timestamp even though content hasn't changed
            try:
                self.cursor.execute("""
                    UPDATE pdfs SET
                        date_extracted = %s
                    WHERE url = %s
                    """,
                                    (datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), response.url))
                self.conn.commit()
                print(f"Updated timestamp for {response.url}")
            except Exception as e:
                print(f"Error updating timestamp: {e}")

            return

        # Try to save page as PDF, but handle WebDriver errors
        try:
            # Use a try-except block specifically for PDF generation
            try:
                print(f"Attempting to generate PDF for {response.url}")
                pdf_files = UrlToPdf([response.url]).main()
                # Get the PDF data from the BytesIO object
                pdf_data = pdf_files[0].getvalue()

                # Verify it's a PDF
                if not pdf_data.startswith(b'%PDF-'):
                    raise ValueError("Generated file does not appear to be a valid PDF")

                print(f"Successfully generated PDF for {response.url}")
            except Exception as pdf_error:
                import traceback
                error_traceback = traceback.format_exc()
                print(f"======= PDF GENERATION ERROR =======")
                print(f"Error generating PDF for {response.url}: {pdf_error}")
                print(f"Traceback:\n{error_traceback}")
                print(f"====================================")

                # Skip this URL completely if PDF generation fails
                print(f"Skipping {response.url} because PDF generation failed")
                # Make sure URL is still tracked to prevent deletion
                normalized_url = self.normalize_url(response.url)
                self.found_urls.add(normalized_url)
                return  # Exit the function early



            # Get date_detected
            date_detected = self.extract_date_from_url(response.url)

            # Get BreadCrumbs
            soup = BeautifulSoup(response.text, 'html.parser')
            breadcrumb_list = self.get_breadcrumbs(soup)
            breadCrumb = str(breadcrumb_list)

        except Exception as e:
            print(f"Error getting breadcrumbs: {e}", traceback.format_exc())

        # Check if document exists in database
        self.cursor.execute("SELECT id FROM pdfs WHERE url=%s", (response.url,))
        existing_doc = self.cursor.fetchone()

        if existing_doc:
            # Update existing document
            self.cursor.execute("""
                UPDATE pdfs SET
                    file = %s,
                    date_detected = %s,
                    date_extracted = %s,
                    breadCrumb = %s,
                    checksum = %s
                WHERE url = %s
                """,
                                (pdf_data, date_detected, datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                 breadCrumb, new_checksum, response.url))
        else:
            # Insert new document
            self.cursor.execute("""
                INSERT INTO pdfs 
                    (file, date_detected, date_extracted, url, title, breadCrumb, checksum, library, username)
                VALUES 
                    (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                                (pdf_data, date_detected, datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                 response.url, title, breadCrumb, new_checksum, 'RH', 'all_users'))

        self.conn.commit()

    def closed(self, reason):
        """Handle cleanup and document removal when spider is closed."""
        print(f"\n{'=' * 80}")
        print(f"Spider closing. Reason: {reason}")
        print(f"Pages visited: {self.pages_visited}")
        print(f"Documents processed: {self.docs_processed}")
        print(f"Total unique URLs found: {len(self.found_urls)}")
        print(f"{'=' * 80}\n")

        print("Removing documents that weren't found in this crawl...")

        try:
            # Get all HR documents from database
            self.cursor.execute("""
                SELECT id, url FROM pdfs 
                WHERE library = 'RH' AND username = 'all_users'
            """)
            all_docs = self.cursor.fetchall()

            # Debug info
            print(f"Total documents in database: {len(all_docs)}")

            # Process each document
            deleted_count = 0
            kept_count = 0
            for doc_id, doc_url in all_docs:
                # Normalize the URL from the database
                normalized_db_url = self.normalize_url(doc_url)

                if normalized_db_url not in self.found_urls:
                    print(f"Removing document: {doc_url}")
                    # Delete document that wasn't found during this crawl
                    self.cursor.execute("DELETE FROM pdfs WHERE id = %s", (doc_id,))
                    deleted_count += 1
                else:
                    kept_count += 1

        except Exception as e:
            print(f"Error removing document: {e}", traceback.format_exc())

        # Additional debug info
        if deleted_count == 0 and len(all_docs) > 0:
            print("\nWARNING: No documents were deleted. Here's what might be wrong:")
            print("1. URL format mismatch between database and found_urls set")
            print("2. Spider might not be reaching all parts of the website")
            print("\nHere are 5 sample URLs from found_urls:")
            sample_urls = list(self.found_urls)[:5]
            for i, url in enumerate(sample_urls):
                print(f"  {i + 1}. {url}")

            print("\nHere are 5 sample URLs from the database:")
            sample_db_urls = [doc_url for _, doc_url in all_docs[:5]]
            for i, url in enumerate(sample_db_urls):
                normalized = self.normalize_url(url)
                print(f"  {i + 1}. Original: {url}")
                print(f"     Normalized: {normalized}")
                print(f"     In found_urls: {normalized in self.found_urls}")

        # Commit changes and close connections
        self.conn.commit()
        print(f"\nSpider finished. Summary:")
        print(f"- Documents kept: {kept_count}")
        print(f"- Documents removed: {deleted_count}")

        # Close database connection
        self.cursor.close()
        self.conn.close()