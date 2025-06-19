import datetime
import hashlib
import json
import os
import re
import sys
import time
import sqlite3
import uuid
from urllib.parse import urlparse

import pymysql
import scrapy
from bs4 import BeautifulSoup
from docx2pdf import convert
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from webdriver_manager.chrome import ChromeDriverManager

from epfl_hr_scraper.scraper_utils.chromedriver import get_working_chromedriver_path

# Current file's directory
current_dir = os.path.dirname(os.path.abspath(__file__))
print("current_dir", current_dir)
# Directory two levels up from the current file
root_project_dir = os.path.abspath(os.path.join(current_dir, '../../..'))
print('parent_dir', root_project_dir)

# Add the parent directory to sys.path
sys.path.insert(0, root_project_dir)
from myUtils.textExtractor import read_docx_pdf
from myUtils.urlToPDF import UrlToPdf
import pickle as pkl
from myUtils.connect_acad2 import initialize_all_connection


class HrSpider(scrapy.Spider):
    name = 'lex_spider_finance'
    allowed_domains = ['www.epfl.ch', 'www.admin.ch', 'www.efv.admin.ch', 'www.fedlex.admin.ch']
    start_urls = [
        # 'https://www.epfl.ch/about/overview/regulations-and-guidelines/polylex-en/polylex-search/',
        'https://www.epfl.ch/about/overview/fr/reglements-et-directives/polylex/polylex-recherche/'
    ]
    library_name = 'servicenow_finance'
    username = 'servicenow_user'
    database_folder = os.path.join(root_project_dir, 'data', library_name)

    # Add the local folder for saving PDFs
    local_pdf_folder = r'C:\Dev\EPFL-chatbot-clean\backend\epfl_hr_scraper\data\LEX_finances'

    # Add JSON metadata file path
    json_metadata_file = os.path.join(local_pdf_folder, 'lex_finance_articles.json')

    # List to store metadata for each document
    document_metadata = []

    # create folders
    if not os.path.exists(database_folder):
        os.makedirs(database_folder)

    # Create the local PDF folder if it doesn't exist
    if not os.path.exists(local_pdf_folder):
        os.makedirs(local_pdf_folder)

    # Connect to database
    conn, cursor = initialize_all_connection()

    def generate_checksum(self, content):
        """Generate a SHA-256 checksum for the given content."""
        return hashlib.sha256(content.encode('utf-8')).hexdigest()

    def add_document_metadata(self, url, title, lex_number=None):
        """Add document metadata to the list for later JSON export"""
        # Create a unique ID based on the URL
        doc_id = str(uuid.uuid5(uuid.NAMESPACE_URL, url))[:32]

        # Create a formatted title, including lex_number if available
        formatted_title = title
        if lex_number:
            formatted_title = f"{lex_number}_{title}"

        # Get the current date and time
        current_date = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # Create the metadata entry
        metadata = {
            "id": doc_id,
            "title": formatted_title,
            "views": "• 0 Vues",  # Default value as we don't have view counts
            "lastUpdate": current_date,
            "url": url
        }

        # Add to the list of document metadata
        self.document_metadata.append(metadata)
        print(f"Added metadata for document: {formatted_title}")

    def save_metadata_to_json(self):
        """Save the document metadata to a JSON file"""
        if self.document_metadata:
            try:
                with open(self.json_metadata_file, 'w', encoding='utf-8') as f:
                    json.dump(self.document_metadata, f, ensure_ascii=False, indent=2)
                print(f"Saved metadata for {len(self.document_metadata)} documents to {self.json_metadata_file}")
            except Exception as e:
                print(f"Error saving metadata to JSON: {e}")
                import traceback
                traceback.print_exc()

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

    def visit_all_lex_pages(self, response):
        # Initialize an empty list to hold dictionaries of lex_number, URLs, and titles
        lex_entries = []

        # Iterate through each 'div.lex-row-1' (or any other number) element to extract the needed information
        for lex_row in response.xpath('//div[contains(@class, "lex-row-1")]'):

            # Extract lex_number, url, and title from each row
            lex_number = lex_row.xpath('./div[contains(@class, "lex-number")]/text()').get().strip()
            if not lex_number.startswith('5.'):
                print('lex_number', lex_number, 'does not start with 5.')
                continue
            url = lex_row.xpath('.//a[contains(@class, "lex-url")]/@href').get()
            title = "".join(
                lex_row.xpath('.//a[contains(@class, "lex-url")]/span[@class="lex-title"]//text()').getall()).strip()

            # Construct and append the dictionary for the current row
            lex_entries.append({'lex_number': lex_number, 'url': url, 'title': title})

        # For debugging: Print the list of dictionaries
        print('Lex entries:', lex_entries)
        print('number of lex entries:', len(lex_entries))

        for entry in lex_entries:
            # if entry['url'] == 'https://www.admin.ch/opc/fr/classified-compilation/19950127/index.html':
            #     print('Following URL:', entry['url'])
            #     yield response.follow(entry['url'], self.parse_lex_page, cb_kwargs=entry)

            if (entry['url'].endswith('.pdf') or
                    entry['url'] == "https://drive.google.com/file/d/1EonBcBM5HMvhBcK40Mn_itzbs23OYQnU/view" or
                    entry['url'] == "https://drive.google.com/file/d/17tjYeWgmYMLHmbrBFMRhvTdpabeBj8wf/view"):
                print('downloading direct pdf from ', entry['url'])
                # continue
                yield scrapy.Request(entry['url'], callback=self.save_document)
            elif ('https://www.admin.ch/' in entry['url'] or
                  'http://www.admin.ch/' in entry['url'] or
                  'https://www.fedlex.admin.ch/' in entry['url'] or
                  'http://www.fedlex.admin.ch/' in entry['url']):
                # continue
                print('Following URL admin or fedlex:', entry['url'])
                print('entry', entry)
                entry['redo'] = True
                yield response.follow(entry['url'], self.parse_lex_page, errback=self.errback_function, cb_kwargs=entry)

            elif entry['url'] == "https://www.epfl.ch/education/studies/en/rules-and-procedures/study_plans/":
                # continue
                print('Following URL:', entry['url'])
                entry['redo'] = True
                yield response.follow(entry['url'], self.parse_lex_page, cb_kwargs=entry)

            elif entry[
                'url'] == "http://isa.epfl.ch/imoniteur_ISAP/%21gedpublicreports.htm?ww_i_reportmodel=1715636965":
                continue

            elif entry['url'] == "https://www.epfl.ch/education/phd/regulations/":
                # continue
                print('Following URL:', entry['url'])
                yield response.follow(entry['url'], self.parse_phd_page, cb_kwargs=entry)

            elif entry['url'] == "https://www.epfl.ch/education/studies/en/rules-and-procedures/study_plans":
                # continue
                print('Following URL:', entry['url'])
                print('entry', entry)
                yield response.follow(entry['url'], self.save_url_to_pdf, cb_kwargs=entry)

            elif entry[
                'url'] == "https://www.efv.admin.ch/efv/fr/home/themen/finanzpolitik_grundlagen/risiko_versicherungspolitik.html":
                # continue
                print('Following URL:', entry['url'])
                yield response.follow(entry['url'], self.parse_efv_page, cb_kwargs=entry)
            else:
                print('url unknown:', entry['url'])
                continue

    def parse_efv_page(self, response, lex_number, url, title):
        print('Parsing PhD page:', lex_number, url, title)
        # find div having classes .tab-content and .tab-border
        table = response.xpath('//div[contains(@class, "tab-content") and contains(@class, "tab-border")]')
        links = table.xpath('.//a/@href').getall()
        print('efv links', links)
        print('efv links', len(links))
        for link in links:
            print('link', link)
        for link in links:
            if 'www.admin.ch' in link:
                continue
            elif link.endswith('.pdf'):
                yield response.follow(link, self.save_document)
            else:
                yield response.follow(link, self.save_url_to_pdf,
                                      cb_kwargs={'title': title, 'url': link, 'lex_number': lex_number})

    def parse_phd_page(self, response, lex_number, url, title):
        print('Parsing PhD page:', lex_number, url, title)
        # find div .card-title
        card_bodies = response.xpath('//div[contains(@class, "card-title")]')
        for card_body in card_bodies:
            href = card_body.xpath('.//a/@href').get()
            print('href', href)
            if href.endswith('.pdf'):
                yield response.follow(href, self.save_document)
            elif href in ["https://www.epfl.ch/education/phd/regulations/internal-regulations/",
                          "https://www.epfl.ch/education/phd/regulations/doctoral-programs-regulations/"]:
                yield response.follow(href, self.parse_pdh_page_second_level)
            elif href == "href https://www.epfl.ch/education/phd/regulations/edoc-doctoral-commission-decisions-cdoct/":
                yield response.follow(href, self.save_url_to_pdf,
                                      cb_kwargs={'title': title, 'url': href, 'lex_number': lex_number})
            else:
                continue

    def parse_pdh_page_second_level(self, response):
        print('Parsing PhD page:', response.url)
        print(response.text)
        # find all links in <main id="content role="main">
        links = response.xpath('//main[@id="content"]//a/@href').getall()
        for link in links:
            if link.endswith('.pdf'):
                yield response.follow(link, self.save_document)
            else:
                yield response.follow(link, self.save_url_to_pdf)

    def errback_function(self, failure):
        request = failure.request
        self.logger.error(f"Request for {request.url} failed: {repr(failure)}")

    def parse_lex_page(self, response, lex_number, url, title, redo=True):
        print('Parsing lex page:', lex_number, url, title)

        if not redo:
            # Check if url is already in db
            self.cursor.execute("SELECT url FROM source_docs WHERE url=%s", (url,))
            if self.cursor.fetchone():
                return

        # Get BreadCrumbs using BeautifulSoup for parsing HTML
        soup = BeautifulSoup(response.text, 'html.parser')
        breadcrumb_list = self.get_breadcrumbs(soup)
        breadCrumb = str(breadcrumb_list)

        # Set up the download directory
        download_directory = os.path.abspath(os.path.join(self.database_folder, 'temp_downloads'))
        if not os.path.exists(download_directory):
            os.makedirs(download_directory)

        # Initialize Selenium WebDriver with custom download directory
        options = webdriver.ChromeOptions()
        prefs = {
            "download.default_directory": download_directory,
            "download.prompt_for_download": False,
            "download.directory_upgrade": True,
            "plugins.always_open_pdf_externally": True
        }
        options.add_experimental_option("prefs", prefs)
        options.add_argument("--verbose")
        options.add_argument("--log-path=chromedriver.log")

        try:
            # Get a working ChromeDriver path
            chromedriver_path = get_working_chromedriver_path()

            if chromedriver_path:
                print(f"Using ChromeDriver at: {chromedriver_path}")
                service = Service(chromedriver_path)
                newDriver = webdriver.Chrome(service=service, options=options)
            else:
                print("Could not find a working ChromeDriver, trying default configuration...")
                # Last resort - try without specifying a driver path
                newDriver = webdriver.Chrome(options=options)

            # Navigate to the URL
            newDriver.get(url)

            # Your logic to interact with the page, click the PDF button, and download
            try:
                # Locate the <tr> element that contains a <td> with class 'no-padding-right is-active'
                target_row = WebDriverWait(newDriver, 10).until(
                    EC.presence_of_element_located(
                        (By.XPATH, "//tr[.//td[contains(@class, 'no-padding-right is-active')]]"))
                )

                # Within this <tr>, find the <td> that contains the date and extract its text
                date_element = target_row.find_element(By.XPATH,
                                                       ".//td[contains(@class, 'no-padding-right is-active')]")
                date_detected = date_element.text.strip()  # Expected format: 'DD.MM.YYYY'
                date_detected = datetime.datetime.strptime(date_detected, '%d.%m.%Y').strftime('%Y-%m')

                # Also find the PDF button within this <tr>
                pdf_button = target_row.find_element(By.XPATH,
                                                     ".//button[contains(@class, 'app-button') and contains(text(), 'PDF')]")
                print('Found PDF button:', pdf_button)
                pdf_button.click()

                download_link = WebDriverWait(newDriver, 10).until(
                    EC.element_to_be_clickable(
                        (By.XPATH,
                         "//a[contains(@data-title-hover, 'Download') or contains(@data-title-hover, 'Télécharger')]")
                    )
                )
                print('download_link', download_link)
                download_link.click()

                # Wait for the file to download
                time.sleep(5)  # You might need to adjust this delay

                # Process downloaded files
                files = os.listdir(download_directory)
                print('Screening files:', files)
                for file in files:
                    if file.endswith('.pdf'):
                        print('Processing file:', file)
                        file_path = os.path.join(download_directory, file)
                        # add to database
                        with open(file_path, 'rb') as f:
                            mypdf = f.read()

                        # Create a sanitized filename for the local copy
                        sanitized_filename = f"{lex_number}_{self.sanitize_filename(title)}.pdf"
                        local_file_path = os.path.join(self.local_pdf_folder, sanitized_filename)

                        # Save a copy to the local folder
                        with open(local_file_path, 'wb') as f:
                            f.write(mypdf)
                        print(f"Saved PDF to local folder: {local_file_path}")

                        # Add metadata for this document
                        self.add_document_metadata(url, title, lex_number)

                        # Try to insert with duplicate handling
                        try:
                            # Try to insert first
                            self.cursor.execute(
                                "INSERT INTO source_docs (file, date_detected, date_extracted, url, title, breadCrumb, checksum, library, username, doc_type) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
                                (mypdf, date_detected, datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), url,
                                 str(lex_number) + '_' + title, breadCrumb,
                                 '', self.library_name + '_new', self.username, 'pdf'))
                        except pymysql.err.IntegrityError as e:
                            if e.args[0] == 1062:  # MySQL error code for duplicate entry
                                # If duplicate URL, update the existing record
                                self.cursor.execute(
                                    "UPDATE source_docs SET file=%s, date_detected=%s, date_extracted=%s, title=%s, breadCrumb=%s WHERE url=%s",
                                    (mypdf, date_detected, datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                     str(lex_number) + '_' + title, breadCrumb, url))
                                print(f"Updated existing document for URL: {url}")
                            else:
                                # Re-raise if it's not a duplicate entry error
                                raise

                        self.conn.commit()

                        # delete the pdf file from temp directory
                        os.remove(file_path)
            except Exception as e:
                print(f"Error during page interaction: {e}")
                import traceback
                traceback.print_exc()

        except Exception as e:
            print(f"Error setting up WebDriver: {e}")
            import traceback
            traceback.print_exc()
            return

        finally:
            if 'newDriver' in locals():
                try:
                    newDriver.quit()
                except Exception as e:
                    print(f"Error closing WebDriver: {e}")

    def parse(self, response):
        yield from self.visit_all_lex_pages(response)

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
        pdf_file = UrlToPdf(
            [response.url]).main()

        # get date_detected
        date_detected = self.extract_date_from_url(response.url)

        # get BreadCrumbs
        soup = BeautifulSoup(response.text, 'html.parser')
        breadcrumb_list = self.get_breadcrumbs(soup)
        breadCrumb = str(breadcrumb_list)

        # Create a sanitized filename for the local copy
        sanitized_filename = f"{self.sanitize_filename(title)}.pdf"
        if lex_number:
            sanitized_filename = f"{lex_number}_{self.sanitize_filename(title)}.pdf"

        local_file_path = os.path.join(self.local_pdf_folder, sanitized_filename)

        # Save a copy to the local folder
        with open(local_file_path, 'wb') as f:
            f.write(pdf_file[0].getbuffer().tobytes())
        print(f"Saved PDF to local folder: {local_file_path}")

        # Add metadata for this document
        self.add_document_metadata(response.url, title, lex_number)

        # insert into database
        self.cursor.execute(
            "INSERT INTO source_docs (file, date_detected, date_extracted, url, title, breadCrumb, checksum, library, username, doc_type) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s )",
            (
            pdf_file[0].getbuffer(), date_detected, datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), response.url,
            title, breadCrumb,
            new_checksum, self.library_name + '_new', self.username, 'pdf'))

        self.conn.commit()

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

        if filename.endswith('.docx'):
            with open(destination_path, 'wb') as f:
                f.write(response.body)
            convert(destination_path, destination_path.replace('.docx', '.pdf'))
            destination_path = destination_path.replace('.docx', '.pdf')
            filename = filename.replace('.docx', '.pdf')
        else:
            print('destination_path', destination_path)
            with open(destination_path, 'wb') as f:
                f.write(response.body)

        # read the pdf file just saved
        with open(destination_path, 'rb') as f:
            mypdf = f.read()

        date_detected = self.extract_date_from_url(response.url)

        # Create a sanitized filename for the local copy
        sanitized_filename = self.sanitize_filename(filename)
        local_file_path = os.path.join(self.local_pdf_folder, sanitized_filename)

        # Save a copy to the local folder
        with open(local_file_path, 'wb') as f:
            f.write(mypdf)
        print(f"Saved PDF to local folder: {local_file_path}")

        # Add metadata for this document
        self.add_document_metadata(response.url, filename)

        # insert into database
        try:
            # Try to insert first
            self.cursor.execute(
                "INSERT INTO source_docs (file, date_detected, date_extracted, url, title, breadCrumb, checksum, library, username, doc_type) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
                (
                    mypdf, date_detected, datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), response.url, filename,
                    '',
                    '', self.library_name + '_new', 'all_users', 'pdf'))
        except pymysql.err.IntegrityError as e:
            if e.args[0] == 1062:  # MySQL error code for duplicate entry
                # If duplicate URL, update the existing record
                self.cursor.execute(
                    "UPDATE source_docs SET file=%s, date_detected=%s, date_extracted=%s, title=%s WHERE url=%s",
                    (mypdf, date_detected, datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), filename,
                     response.url))
                print(f"Updated existing document for URL: {response.url}")
            else:
                # Re-raise if it's not a duplicate entry error
                raise

        self.conn.commit()

        # clean temp folder
        for file in os.listdir(temp_folder):
            os.remove(os.path.join(temp_folder, file))

    def sanitize_filename(self, filename):
        """Create a valid filename by removing invalid characters."""
        # Replace invalid characters with underscore
        invalid_chars = r'[<>:"/\\|?*]'
        sanitized = re.sub(invalid_chars, '_', filename)
        # Ensure the filename is not too long
        if len(sanitized) > 200:
            name, ext = os.path.splitext(sanitized)
            sanitized = name[:195] + ext
        # Ensure it has a .pdf extension
        if not sanitized.lower().endswith('.pdf'):
            sanitized += '.pdf'
        return sanitized

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

    def closed(self, reason):
        """Method called when the spider is closed."""
        print(f"Spider closed with reason: {reason}")

        # Save the metadata to JSON regardless of the reason
        self.save_metadata_to_json()

        if reason == 'finished':  # Only do this if the spider completed successfully
            try:
                print("Starting post-crawl database cleanup...")

                # Check if we have any LEX_new entries
                library_value = self.library_name + '_new'
                self.cursor.execute(f"SELECT COUNT(*) FROM source_docs WHERE library = '{library_value}'")
                lex_new_count = self.cursor.fetchone()[0]

                if lex_new_count > 0:
                    print(f"Found {lex_new_count} new entries. Proceeding with cleanup...")

                    # Set a minimum threshold for new entries as a safety measure
                    min_expected_entries = 10  # Adjust this based on your expectations
                    if lex_new_count < min_expected_entries:
                        print(
                            f"Only {lex_new_count} new entries found, which is less than the expected minimum of {min_expected_entries}.")
                        print("Skipping cleanup as a precaution. Please review the crawl results.")
                        return

                    # Begin transaction
                    self.cursor.execute("START TRANSACTION")

                    # Delete old LEX entries from source_docs
                    self.cursor.execute(f"DELETE FROM source_docs WHERE library = '{self.library_name}' ")
                    deleted_source_docs = self.cursor.rowcount
                    print(f"Deleted {deleted_source_docs} old entries from source_docs")

                    # Rename LEX_new to LEX in source_docs
                    library_value = self.library_name + '_new'
                    self.cursor.execute(
                        f"UPDATE source_docs SET library = '{self.library_name}'  WHERE library = '{library_value}'")
                    updated_source_docs = self.cursor.rowcount
                    print(f"Renamed {updated_source_docs} entries from LEX_new to LEX in source_docs")

                    # Commit transaction
                    self.cursor.execute("COMMIT")
                    print("Database cleanup completed successfully!")
                else:
                    print("No new entries found. Skipping cleanup.")

            except Exception as e:
                print(f"Error during post-crawl cleanup: {e}")
                # Rollback transaction if something went wrong
                self.cursor.execute("ROLLBACK")
                import traceback
                traceback.print_exc()
            finally:
                # Close database connection
                if hasattr(self, 'conn') and self.conn:
                    self.conn.close()
                    print("Database connection closed.")