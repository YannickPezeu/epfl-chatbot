import base64
import json
import time
from io import BytesIO
from typing import List

from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.chrome.options import Options as ChromeOptions
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager


class UrlToPdf:
    """
    Generates PDF files from web pages using Selenium and Chrome's Headless print-to-PDF feature.

    Example use case:
        pdf_generator = PdfGenerator(['https://google.com'])
        pdf_files = pdf_generator.main()
        with open('new_pdf.pdf', "wb") as outfile:
            outfile.write(pdf_files[0].getbuffer())
    """
    driver = None

    # Chrome DevTools print-to-PDF options
    print_options = {
        'landscape': False,
        'displayHeaderFooter': False,
        'printBackground': True,
        'preferCSSPageSize': True,
        'paperWidth': 40.27,  # A4 size in inches
        'paperHeight': 25.7,  # A4 size in inches
    }

    def __init__(self, urls: List[str]):
        self.urls = urls

    def _get_pdf_from_url(self, url):
        self.driver.get(url)

        # Wait for the page to load
        WebDriverWait(self.driver, 10).until(EC.presence_of_element_located((By.TAG_NAME, "body")))

        # Find all buttons that need to be clicked to reveal content
        collapse_buttons = self.driver.find_elements(By.XPATH, '//button[contains(@class, "collapse-title")]')

        # Click each button to reveal the content
        for button in collapse_buttons:
            try:
                # Scroll the button into view and click it
                self.driver.execute_script("arguments[0].scrollIntoView(true);", button)
                button.click()
                # Wait a bit for the content to expand and stabilize
                time.sleep(1)
            except Exception as e:
                print(f"Could not click the button: {e}")

        # Additional wait to ensure all animations have completed and content is fully loaded
        time.sleep(2)

        # Script to hide specific elements via CSS
        hide_elements_script = """
            (function() {
                var style = document.createElement('style');
                style.type = 'text/css';
                style.media = 'print';
                style.innerText = '@media print {.footer, .page-header, .page-footer, .nav-aside-wrapper { display: none !important; } }';
                style.innerText += `@media print {` +
            `#menu-item-1, #menu-item-2, #menu-item-3, #menu-item-4, #menu-item-5,` +
            `#menu-item-6, #menu-item-7, #menu-item-8, #menu-item-9,#menu-item-10, ` +
            `header .icon, #nav-toggle {` +
            ` display: none !important; }` +
            `}`;
                document.head.appendChild(style);
            })();
        """
        self.driver.execute_script(hide_elements_script)

        # Inject custom script to add a header with the URL
        # Note: Correctly handle curly braces and string literals
        add_header_script = f"""
            (function() {{
                var header = document.createElement('div');
                header.style.position = 'absolute';
                header.style.top = '0';
                header.style.left = '0';
                header.style.width = '100%';
                header.style.textAlign = 'center';
                header.style.marginTop = '0px'; // Adjust this value as needed
                header.innerText = 'Source: {url}';
                document.body.insertBefore(header, document.body.firstChild);
            }})();
        """
        self.driver.execute_script(add_header_script)

        # Adjust print options to ensure the header is visible
        print_options = self.print_options.copy()
        print_options.update({
            'marginTop': 0.4,  # Ensure there's enough space at the top for the header
            'marginBottom': 0.4,  # Ensure there's enough space at the top for the header
        })

        result = self._send_devtools_command("Page.printToPDF", print_options)
        return base64.b64decode(result['data'])

    def _send_devtools_command(self, cmd, params):
        resource = f"/session/{self.driver.session_id}/chromium/send_command_and_get_result"
        url = self.driver.command_executor._url + resource
        body = json.dumps({'cmd': cmd, 'params': params})
        response = self.driver.command_executor._request('POST', url, body)
        return response.get('value')

    def _generate_source_docs(self):
        pdf_files = []
        for url in self.urls:
            result = self._get_pdf_from_url(url)
            file = BytesIO(result)
            pdf_files.append(file)
        return pdf_files

    def main(self) -> List[BytesIO]:
        options = ChromeOptions()
        options.add_argument('--headless')
        options.add_argument('--disable-gpu')

        self.driver = webdriver.Chrome(
            service=ChromeService(
                r"C:\Users\pezeu\.wdm\drivers\chromedriver\win64\134.0.6998.90\chromedriver-win32\chromedriver.exe"),
            options=options
        )

        try:
            result = self._generate_source_docs()
        finally:
            self.driver.quit()  # Ensure the driver is closed properly

        return result

if __name__ == '__main__':

    pdf_file = UrlToPdf(['https://www.epfl.ch/campus/services/human-resources/']).main()
    print(pdf_file)
    # save pdf to file
    with open('epfl3.pdf', "wb") as outfile:
        outfile.write(pdf_file[0].getbuffer())