from bs4 import BeautifulSoup
import re
import requests
from requests.auth import HTTPBasicAuth
import json
import dotenv
import os
from datetime import datetime

# Load environment variables
dotenv.load_dotenv('C:/Dev/EPFL-chatbot-compose/backend/.env')
password = os.getenv("SERVICENOW_KEY")

# Authentication credentials
username = "WS_AI"

# Base URL and knowledge base ID
base_url = "https://epfl.service-now.com/api/now/table/kb_knowledge"
kb_id = "7243ab6347332100158b949b6c9a7194"  # Specified KB ID


# Function to fetch articles with pagination
def fetch_kb_articles():
    all_articles = []
    total_fetched = 0

    # Query parameters with pagination
    params = {
        "sysparm_display_value": "true",
        "kb_knowledge_base": kb_id,  # Using the specific KB ID
        "active": "true",  # Only active articles
        "sysparm_limit": 100,  # Fetch 100 at a time
        "sysparm_offset": 0
    }

    print(f"Fetching KB articles from knowledge base ID: {kb_id}")

    while True:
        # Make API request
        response = requests.get(
            base_url,
            params=params,
            auth=HTTPBasicAuth(username, password),
            headers={"Accept": "application/json"}
        )

        # Check if successful
        if response.status_code == 200:
            data = response.json()
            articles = data.get('result', [])

            # If no more articles, break
            if not articles:
                break

            # Add to our collection
            all_articles.extend(articles)
            total_fetched += len(articles)

            # Print progress
            print(f"Retrieved {len(articles)} articles, total so far: {total_fetched}")

            # Update offset for next page
            params["sysparm_offset"] += params["sysparm_limit"]
        else:
            print(f"Error: {response.status_code} - {response.text}")
            break

    return all_articles


# Try fetching KB base information to verify it exists
def get_kb_info():
    kb_base_url = f"https://epfl.service-now.com/api/now/table/kb_knowledge_base/{kb_id}"

    print(f"Fetching information about KB base: {kb_id}")

    response = requests.get(
        kb_base_url,
        auth=HTTPBasicAuth(username, password),
        headers={"Accept": "application/json"}
    )

    if response.status_code == 200:
        data = response.json()
        result = data.get('result', {})
        print(f"Knowledge Base Title: {result.get('title')}")
        print(f"Description: {result.get('description')}")
        return result
    else:
        print(f"Error retrieving KB base info: {response.status_code} - {response.text}")
        return None


# Try querying KB articles without specifying knowledge base
def fetch_all_kb_articles():
    print("\nAttempting to fetch articles without KB filter...")

    response = requests.get(
        base_url,
        params={"sysparm_limit": 10},
        auth=HTTPBasicAuth(username, password),
        headers={"Accept": "application/json"}
    )

    if response.status_code == 200:
        data = response.json()
        articles = data.get('result', [])
        print(f"Retrieved {len(articles)} articles without KB filter")

        if articles:
            kb_bases = set()
            for article in articles:
                if isinstance(article.get('kb_knowledge_base'), dict):
                    kb_id = article['kb_knowledge_base'].get('value')
                    kb_name = article['kb_knowledge_base'].get('display_value')
                    kb_bases.add((kb_id, kb_name))

            print("Knowledge bases found in articles:")
            for kb_id, kb_name in kb_bases:
                print(f"- {kb_name} (ID: {kb_id})")
    else:
        print(f"Error: {response.status_code} - {response.text}")




# Load environment variables
dotenv.load_dotenv('C:/Dev/EPFL-chatbot-compose/backend/.env')


# Base URLs
base_url2 = "https://epfl.service-now.com/api/now/table"
kb_base_url = f"{base_url2}/kb_knowledge_base"
kb_article_url = f"{base_url2}/kb_knowledge"

def list_knowledge_bases_with_article_counts():
    """List all knowledge bases and count articles in each"""
    print("Listing all knowledge bases accessible to WS_AI user...")

    # Parameters for knowledge bases
    params = {
        "sysparm_display_value": "true",
        "sysparm_limit": 100
    }

    response = requests.get(
        kb_base_url,
        params=params,
        auth=HTTPBasicAuth(username, password),
        headers={"Accept": "application/json"}
    )

    kb_results = []

    if response.status_code == 200:
        data = response.json()
        kb_bases = data.get('result', [])
        print(f"Found {len(kb_bases)} knowledge bases")

        if kb_bases:
            print("\nKnowledge Bases and Article Counts:")

            for i, kb in enumerate(kb_bases, 1):
                kb_id = kb.get('sys_id')
                kb_title = kb.get('title')

                print(f"\n{i}. {kb_title} (ID: {kb_id})")
                print(f"   Description: {kb.get('description', 'No description')}")
                print(f"   Active: {kb.get('active')}")

                # Count articles in this knowledge base
                article_info = count_articles_in_kb(kb_id, kb_title)
                kb_results.append({
                    "knowledge_base": kb,
                    "article_info": article_info
                })

            # Save the results to a file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"kb_article_counts_{timestamp}.json"

            with open(filename, "w", encoding="utf-8") as f:
                json.dump(kb_results, f, ensure_ascii=False, indent=2)

            print(f"\nResults saved to {filename}")

            # Print summary
            print("\nSummary of Knowledge Bases and Article Counts:")
            for result in kb_results:
                kb = result["knowledge_base"]
                article_info = result["article_info"]
                print(f"• {kb.get('title')}: {article_info['max_count']} articles")

            return kb_results
        else:
            print("No knowledge bases found")
    else:
        print(f"Error retrieving knowledge bases: {response.status_code} - {response.text}")

    return []


def count_articles_in_kb(kb_id, kb_title):
    """Count articles in a specific knowledge base"""
    print(f"\nChecking articles in '{kb_title}' (ID: {kb_id})...")

    # Try different combinations of parameters to get the most comprehensive results
    parameter_sets = [
        {"kb_knowledge_base": kb_id},
        {"kb_knowledge_base": kb_id, "workflow_state": "published"},
        {"kb_knowledge_base": kb_id, "active": "true"},
        {"kb_knowledge_base": kb_id, "workflow_state": "published", "active": "true"}
    ]

    # Store results for each parameter set
    results = []

    for params in parameter_sets:
        # Add common parameters
        full_params = {
            **params,
            "sysparm_display_value": "true",
            "sysparm_limit": 1,  # We only need the count
            "sysparm_offset": 0
        }

        # Create a description of the current parameter set
        param_desc = []
        if "workflow_state" in params:
            param_desc.append(f"workflow_state={params['workflow_state']}")
        if "active" in params:
            param_desc.append(f"active={params['active']}")

        param_str = " and ".join(param_desc) if param_desc else "all articles"

        response = requests.get(
            kb_article_url,
            params=full_params,
            auth=HTTPBasicAuth(username, password),
            headers={"Accept": "application/json"}
        )

        if response.status_code == 200:
            # Check the X-Total-Count header for the total count
            total_count = response.headers.get('X-Total-Count')

            if total_count:
                count = int(total_count)
                results.append({
                    "parameters": param_str,
                    "count": count
                })
                print(f"  - Found {count} articles ({param_str})")
            else:
                # If X-Total-Count header is not available, try to get it from the response
                data = response.json()

                # Get sample article if available
                sample_article = None
                if data.get('result') and len(data['result']) > 0:
                    sample_article = data['result'][0]

                # Try to fetch with larger limit to estimate total
                est_params = {**full_params, "sysparm_limit": 100}
                est_response = requests.get(
                    kb_article_url,
                    params=est_params,
                    auth=HTTPBasicAuth(username, password),
                    headers={"Accept": "application/json"}
                )

                if est_response.status_code == 200:
                    est_data = est_response.json()
                    est_count = len(est_data.get('result', []))

                    results.append({
                        "parameters": param_str,
                        "count": f"at least {est_count}",
                        "note": "Estimated from batch"
                    })
                    print(f"  - Found at least {est_count} articles ({param_str})")

                    # If we got articles, save a sample
                    if est_count > 0 and not sample_article:
                        sample_article = est_data['result'][0]

                # Save sample article info if available
                if sample_article:
                    print(f"  - Sample article: '{sample_article.get('short_description')}'")
                    print(f"    Last updated: {sample_article.get('sys_updated_on')}")
        else:
            print(f"  - Error ({param_str}): {response.status_code}")

    # Return the highest count found
    max_count = 0
    for result in results:
        if isinstance(result['count'], int) and result['count'] > max_count:
            max_count = result['count']

    return {
        "kb_id": kb_id,
        "kb_title": kb_title,
        "article_counts": results,
        "max_count": max_count
    }


def article_retrieval(kb_id, kb_title):
    """Test retrieving articles from a specific knowledge base and save to JSON file"""
    print(f"\n=== Testing article retrieval from '{kb_title}' (ID: {kb_id}) ===")

    success = False
    articles_data = []

    # Try JSON v2 endpoint
    jsonv2_url = f"https://epfl.service-now.com/kb_knowledge.do?JSONv2"
    jsonv2_params = {
        "sysparm_query": f"kb_knowledge_base={kb_id}^workflow_state=published^active=true",
        "sysparm_limit": 100  # Increased limit to get more articles
    }

    print(f"\nTrying JSONv2 endpoint: {jsonv2_url}")

    response = requests.get(
        jsonv2_url,
        params=jsonv2_params,
        auth=HTTPBasicAuth(username, password),
        headers={"Accept": "application/json"}
    )

    print(f"Status code: {response.status_code}")

    if response.status_code == 200:
        try:
            data = response.json()
            if 'records' in data and isinstance(data['records'], list):
                articles = data['records']
                print(f"Retrieved {len(articles)} articles via JSONv2 endpoint")

                if articles:
                    success = True
                    for j, article in enumerate(articles, 1):
                        # Process article content with extract_clean_text
                        article_text = extract_clean_text(article.get('text', ''))

                        # Create article data dictionary
                        # Generate article URL
                        article_id = article.get('sys_id', '')
                        article_url = f"https://epfl.service-now.com/kb_view.do?sys_kb_id={article_id}"

                        article_data = {
                            "kb_id": kb_id,
                            "kb_title": kb_title,
                            "article_id": article_id,
                            "article_title": article.get('short_description', ''),
                            "article_url": article_url,
                            "content": article_text
                        }

                        # Add to articles data list
                        articles_data.append(article_data)

                        # Print summary
                        print(f"Article {j}:")
                        print(f"  - Title: {article.get('short_description')}")
                        print(f"  - ID: {article.get('sys_id')}")
                        print(f"  - Content length: {len(article_text)} characters")
        except json.JSONDecodeError:
            print("Response is not valid JSON")

    # Check if we need to look at permissions
    if not success:
        print("\nCouldn't retrieve any articles. Checking permissions...")

        # Try to get the knowledge base details to check permissions
        kb_details_url = f"{base_url}/kb_knowledge_base/{kb_id}"

        response = requests.get(
            kb_details_url,
            auth=HTTPBasicAuth(username, password),
            headers={"Accept": "application/json"}
        )

        if response.status_code == 200:
            data = response.json()
            kb_details = data.get('result', {})

            print("\nKnowledge Base Details:")
            print(f"  - Title: {kb_details.get('title')}")
            print(f"  - Active: {kb_details.get('active')}")

            # Check for access control fields
            if 'roles' in kb_details:
                print(f"  - Roles: {kb_details.get('roles')}")
            if 'kb_managers' in kb_details:
                print(f"  - Managers: {kb_details.get('kb_managers')}")
            if 'owner' in kb_details:
                print(f"  - Owner: {kb_details.get('owner')}")

    # Write articles to JSON file if we retrieved any
    if articles_data:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"kb_articles_{kb_id}_{timestamp}.json"

        with open(filename, "w", encoding="utf-8") as f:
            json.dump(articles_data, f, ensure_ascii=False, indent=2)

        print(f"\nSaved {len(articles_data)} articles to {filename}")

    return success, articles_data


def extract_clean_text(html_content, include_headers=True):
    """
    Extract clean text from HTML content for RAG system

    Parameters:
    - html_content: The HTML content to clean
    - include_headers: Whether to include headers in the output

    Returns:
    - Clean text content
    """
    if not html_content:
        return ""

    # Use BeautifulSoup to parse the HTML
    soup = BeautifulSoup(html_content, 'html.parser')

    # Remove script and style tags
    for tag in soup.select('script, style'):
        tag.decompose()

    # Handle images - replace with their alt text or [IMAGE] placeholder
    for img in soup.find_all('img'):
        alt_text = img.get('alt', '').strip()
        if alt_text:
            img.replace_with(f"[IMAGE: {alt_text}]")
        else:
            img.replace_with("[IMAGE]")

    # Process the document
    if include_headers:
        # For headers, add spacing and formatting
        for header in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
            header_text = header.get_text().strip()
            # Determine header level (1-6)
            level = int(header.name[1])
            # Create formatted header with appropriate indentation
            prefix = '#' * level + ' '
            header.replace_with(f"\n\n{prefix}{header_text}\n\n")

    # Handle lists
    for ul in soup.find_all('ul'):
        for li in ul.find_all('li'):
            li_text = li.get_text().strip()
            li.replace_with(f"• {li_text}\n")

    for ol in soup.find_all('ol'):
        for i, li in enumerate(ol.find_all('li'), 1):
            li_text = li.get_text().strip()
            li.replace_with(f"{i}. {li_text}\n")

    # Handle paragraphs
    for p in soup.find_all('p'):
        p_text = p.get_text().strip()
        p.replace_with(f"{p_text}\n\n")

    # Handle tables (convert to text representation)
    for table in soup.find_all('table'):
        rows_text = []
        for row in table.find_all('tr'):
            cells = []
            for cell in row.find_all(['td', 'th']):
                cells.append(cell.get_text().strip())
            rows_text.append(" | ".join(cells))
        table.replace_with("\n" + "\n".join(rows_text) + "\n\n")

    # Get the text content
    text = soup.get_text()

    # Clean up the text
    # Replace multiple newlines with just two
    text = re.sub(r'\n{3,}', '\n\n', text)
    # Replace multiple spaces with a single space
    text = re.sub(r' {2,}', ' ', text)
    # Trim leading/trailing whitespace
    text = text.strip()

    return text




if __name__ == '__main__':

    # Execute the function to list knowledge bases
    try:
        kb_results = list_knowledge_bases_with_article_counts()
        print('kb_results', kb_results)
        for result in kb_results:
            print(result)
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback

        traceback.print_exc()

    # Define the knowledge bases to test
    test_kb_bases = [{"id": kb['knowledge_base']['sys_id'], 'title': kb['knowledge_base']['title']} for kb in
                     kb_results]
    print(test_kb_bases)

    # Knowledge bases we want to test
    # test_kb_bases = [
    #     {"id": "3969aa4edbb08cd079f593c8f49619a0", "title": "Publique (hors EPFL)"},
    #     {"id": "7243ab6347332100158b949b6c9a7194", "title": "Base de connaissances financière"},
    #     {"id": "90d645064fd6b6009d2bdf601310c7a2", "title": "EPFL"},
    #     {"id": "b5e390e3dbb430d031895c88f49619ed", "title": "Recherche"},
    #     {"id": "bb4ef5394f8a72009d2bdf601310c7e9", "title": "Service Desk"}
    # ]

    # Create a dictionary to store all articles from all knowledge bases
    all_kb_articles = {}

    # Process each knowledge base
    for kb in test_kb_bases:
        # Generate knowledge base URL
        kb_url = f"https://epfl.service-now.com/kb_view2.do?sys_kb_id={kb['id']}"

        success, articles_data = article_retrieval(kb['id'], kb['title'])
        if success:
            all_kb_articles[kb['id']] = {
                "kb_title": kb['title'],
                "kb_url": kb_url,
                "articles": articles_data
            }

    # Optionally, save all articles from all knowledge bases to a single file
    if all_kb_articles:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"all_kb_articles_{timestamp}.json"

        with open(filename, "w", encoding="utf-8") as f:
            json.dump(all_kb_articles, f, ensure_ascii=False, indent=2)

        print(f"\nSaved all articles from all knowledge bases to {filename}")
        print(f"Total knowledge bases processed: {len(all_kb_articles)}")

        # Count total articles
        total_articles = sum(len(kb_data["articles"]) for kb_data in all_kb_articles.values())
        print(f"Total articles saved: {total_articles}")

