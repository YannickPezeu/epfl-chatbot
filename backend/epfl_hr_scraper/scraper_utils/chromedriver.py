def get_chrome_version():
    """Get the version of Chrome installed on the system."""
    import subprocess
    import re
    import platform

    system = platform.system()

    try:
        if system == "Windows":
            # Method 1: Using registry
            import winreg
            try:
                key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Software\Google\Chrome\BLBeacon")
                version, _ = winreg.QueryValueEx(key, "version")
                return version
            except:
                # Method 2: Try with wmic command
                process = subprocess.Popen(
                    ['wmic', 'datafile', 'where',
                     'name="C:\\\\Program Files\\\\Google\\\\Chrome\\\\Application\\\\chrome.exe"', 'get', 'Version',
                     '/value'],
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True
                )
                output, _ = process.communicate()
                match = re.search(r'Version=(.+)', output.decode('utf-8'))
                if match:
                    return match.group(1)

                # Method 3: Try with alternative Chrome paths
                alt_paths = [
                    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
                    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
                    os.path.expanduser("~") + r"\AppData\Local\Google\Chrome\Application\chrome.exe"
                ]

                for path in alt_paths:
                    if os.path.exists(path):
                        # Use PowerShell to get file version info
                        process = subprocess.Popen(
                            ['powershell', '-command', f'(Get-Item "{path}").VersionInfo.ProductVersion'],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE
                        )
                        output, _ = process.communicate()
                        version = output.decode('utf-8').strip()
                        if version:
                            return version

        elif system == "Darwin":  # macOS
            process = subprocess.Popen(
                ['/Applications/Google Chrome.app/Contents/MacOS/Google Chrome', '--version'],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            output, _ = process.communicate()
            match = re.search(r'Google Chrome\s+(\d+\.\d+\.\d+\.\d+)', output.decode('utf-8'))
            if match:
                return match.group(1)

        elif system == "Linux":
            process = subprocess.Popen(
                ['google-chrome', '--version'],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            output, _ = process.communicate()
            match = re.search(r'Google Chrome\s+(\d+\.\d+\.\d+\.\d+)', output.decode('utf-8'))
            if match:
                return match.group(1)

        # If we get here, we couldn't detect the version
        return None

    except Exception as e:
        print(f"Error detecting Chrome version: {e}")
        return None


def get_working_chromedriver_path():
    """Get a working ChromeDriver path compatible with the installed Chrome version"""
    import os
    import subprocess
    import platform
    import re
    import requests
    import zipfile
    import io
    import shutil
    from selenium import webdriver
    from selenium.webdriver.chrome.service import Service

    # First, let's try to identify if ChromeDriverManager already downloaded a compatible driver
    try:
        from webdriver_manager.chrome import ChromeDriverManager
        driver_path = ChromeDriverManager().install()

        # The driver_path might be pointing to the THIRD_PARTY_NOTICES file
        # Let's look for the actual chromedriver.exe in the same directory
        driver_dir = os.path.dirname(driver_path)

        # Look for chromedriver.exe in this directory or any subdirectories
        chromedriver_path = None
        for root, dirs, files in os.walk(driver_dir):
            for file in files:
                if file == "chromedriver.exe" or (platform.system() != "Windows" and file == "chromedriver"):
                    chromedriver_path = os.path.join(root, file)
                    print(f"Found ChromeDriver at: {chromedriver_path}")

                    # Verify this driver works by trying to create a Chrome instance
                    try:
                        service = Service(chromedriver_path)
                        options = webdriver.ChromeOptions()
                        options.add_argument("--headless")  # Run in headless mode for quick testing
                        driver = webdriver.Chrome(service=service, options=options)
                        driver.quit()
                        print("Successfully verified ChromeDriver works!")
                        return chromedriver_path
                    except Exception as e:
                        print(f"Found ChromeDriver at {chromedriver_path} but it doesn't work: {e}")
                        # Continue searching for other instances

        print("No working ChromeDriver found in ChromeDriverManager directory.")
    except Exception as e:
        print(f"Error with ChromeDriverManager: {e}")

    # If we reach here, we need to manually download a compatible driver
    print("Manually downloading ChromeDriver...")

    # For Chrome 136, we know we need to use the Chrome for Testing framework
    # The version number might be slightly different (e.g., 136.0.7103.92)
    download_url = "https://edgedl.me.gvt1.com/edgedl/chrome/chrome-for-testing/136.0.7103.92/win64/chromedriver-win64.zip"

    # Create a temporary directory for download
    temp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "temp_chromedriver")
    os.makedirs(temp_dir, exist_ok=True)

    # Download the zip file
    try:
        print(f"Downloading from: {download_url}")
        response = requests.get(download_url)

        if response.status_code != 200:
            print(f"Download failed with status code: {response.status_code}")
            return None

        zip_path = os.path.join(temp_dir, "chromedriver.zip")
        with open(zip_path, 'wb') as f:
            f.write(response.content)

        # Extract the zip file
        with zipfile.ZipFile(zip_path) as zip_ref:
            zip_ref.extractall(temp_dir)

        # Find chromedriver.exe in the extracted files
        chromedriver_path = None
        for root, dirs, files in os.walk(temp_dir):
            for file in files:
                if file == "chromedriver.exe" or (platform.system() != "Windows" and file == "chromedriver"):
                    chromedriver_path = os.path.join(root, file)
                    print(f"Found downloaded ChromeDriver at: {chromedriver_path}")

                    # Verify this driver works
                    try:
                        service = Service(chromedriver_path)
                        options = webdriver.ChromeOptions()
                        options.add_argument("--headless")
                        driver = webdriver.Chrome(service=service, options=options)
                        driver.quit()
                        print("Successfully verified downloaded ChromeDriver works!")

                        # Copy to a more permanent location
                        permanent_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chromedriver")
                        os.makedirs(permanent_dir, exist_ok=True)
                        permanent_path = os.path.join(permanent_dir,
                                                      "chromedriver.exe" if platform.system() == "Windows" else "chromedriver")
                        shutil.copy2(chromedriver_path, permanent_path)

                        # Make executable on Linux/Mac
                        if platform.system() != "Windows":
                            os.chmod(permanent_path, 0o755)

                        return permanent_path
                    except Exception as e:
                        print(f"Found ChromeDriver at {chromedriver_path} but it doesn't work: {e}")

        print("No working ChromeDriver found in downloaded files.")
        return None

    except Exception as e:
        print(f"Error manually downloading ChromeDriver: {e}")
        return None
if __name__ == '__main__':
    version = get_chrome_version()
    print('version', version)
    chrome_driver_path = get_working_chromedriver_path()
    print('chrome_driver_path', chrome_driver_path)