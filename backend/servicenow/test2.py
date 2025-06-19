from bs4 import BeautifulSoup
import re
import requests
from requests.auth import HTTPBasicAuth
import json
import dotenv
import os
from datetime import datetime
import time

# Load environment variables
dotenv.load_dotenv('C:/Dev/EPFL-chatbot-compose/backend/.env')
password = os.getenv("SERVICENOW_KEY")

# Authentication credentials
username = "WS_AI"

# Base URL
base_url = "https://epfl.service-now.com/api/now/table"
kb_id = "7243ab6347332100158b949b6c9a7194"  # Base de connaissances financière

from servicenow_get_kb import extract_clean_text


def retrieve_articles_with_direct_id_lookup(kb_id, kb_title, only_published=True):
    """
    Get a list of article IDs first, then retrieve each article individually

    Parameters:
    - kb_id: Knowledge base ID
    - kb_title: Knowledge base title
    - only_published: If True, only retrieve published articles
    """
    print(f"\n=== Retrieving articles with direct ID lookup for '{kb_title}' ===")
    print(f"Filter settings: only_published={only_published}")

    # Step 1: Get a list of all article IDs in this knowledge base
    article_ids = get_article_ids(kb_id, only_published)

    if not article_ids:
        print("No article IDs found")
        return False, []

    print(f"Found {len(article_ids)} article IDs")

    # Step 2: Retrieve each article directly by ID
    all_articles = []
    success_count = 0
    failed_count = 0
    not_published_count = 0

    for i, article_id in enumerate(article_ids, 1):
        print(f"Retrieving article {i}/{len(article_ids)} (ID: {article_id})...")

        article_data = get_article_by_id(article_id, kb_id, kb_title)

        if article_data:
            # Check if we need to filter by published state
            if only_published and article_data.get('workflow_state') != 'published':
                print(f"  - Skipping: article is not published (state: {article_data.get('workflow_state')})")
                not_published_count += 1
                continue

            all_articles.append(article_data)
            success_count += 1
        else:
            failed_count += 1

        # Add a short delay to avoid overwhelming the server
        if i % 10 == 0:
            print(f"Progress: {i}/{len(article_ids)} articles processed")
            print(f"Success: {success_count}, Failed: {failed_count}, Not Published: {not_published_count}")

        # Small delay between requests
        time.sleep(0.5)

    # Final progress update
    print(f"\nFinal results:")
    print(f"Total processed: {len(article_ids)}")
    print(f"Successfully retrieved: {success_count}")
    print(f"Failed to retrieve: {failed_count}")
    print(f"Skipped (not published): {not_published_count}")

    # Write articles to JSON file if we retrieved any
    if all_articles:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"kb_articles_direct_lookup_{kb_id}_{timestamp}.json"

        with open(filename, "w", encoding="utf-8") as f:
            json.dump(all_articles, f, ensure_ascii=False, indent=2)

        print(f"\nSaved {len(all_articles)} articles to {filename}")
        print(f"Success rate: {success_count}/{len(article_ids) - not_published_count} " +
              f"({success_count / (len(article_ids) - not_published_count) * 100:.1f}%)")

    return len(all_articles) > 0, all_articles


def get_article_ids(kb_id, only_published=True):
    """
    Get all article IDs for a knowledge base

    Parameters:
    - kb_id: Knowledge base ID
    - only_published: If True, only get published article IDs
    """
    print(f"Getting article IDs for KB: {kb_id}")
    print(f"Filter settings: only_published={only_published}")

    # We'll try multiple ways to get the article IDs
    article_ids = set()  # Use a set to avoid duplicates

    # Method 1: Use the REST API to get just the sys_ids
    rest_url = f"{base_url}/kb_knowledge"

    # Adjust queries based on filter settings
    if only_published:
        queries = [
            f"kb_knowledge_base={kb_id}^workflow_state=published^active=true",
            f"kb_knowledge_base.sys_id={kb_id}^workflow_state=published^active=true"
        ]
    else:
        queries = [
            f"kb_knowledge_base={kb_id}^active=true",
            f"kb_knowledge_base.sys_id={kb_id}^active=true"
        ]

    for query in queries:
        params = {
            "sysparm_query": query,
            "sysparm_fields": "sys_id,workflow_state",  # Get IDs and workflow state
            "sysparm_limit": 1000  # Try to get as many as possible at once
        }

        try:
            response = requests.get(
                rest_url,
                params=params,
                auth=HTTPBasicAuth(username, password),
                headers={"Accept": "application/json"},
                timeout=30
            )

            if response.status_code == 200:
                data = response.json()
                results = data.get('result', [])

                for article in results:
                    if 'sys_id' in article:
                        article_ids.add(article['sys_id'])

                print(f"Found {len(article_ids)} article IDs using query: {query}")

                # Check X-Total-Count header for verification
                total_count = response.headers.get('X-Total-Count')
                if total_count:
                    print(f"X-Total-Count header indicates {total_count} total articles")
            else:
                print(f"Error with query '{query}': {response.status_code}")
        except Exception as e:
            print(f"Exception with query '{query}': {str(e)}")

    # Method 2: Use the JSONv2 endpoint
    jsonv2_url = "https://epfl.service-now.com/kb_knowledge.do?JSONv2"

    if only_published:
        jsonv2_query = f"kb_knowledge_base={kb_id}^workflow_state=published^active=true"
    else:
        jsonv2_query = f"kb_knowledge_base={kb_id}^active=true"

    jsonv2_params = {
        "sysparm_query": jsonv2_query,
        "sysparm_fields": "sys_id,workflow_state",
        "sysparm_limit": 1000
    }

    try:
        response = requests.get(
            jsonv2_url,
            params=jsonv2_params,
            auth=HTTPBasicAuth(username, password),
            headers={"Accept": "application/json"},
            timeout=30
        )

        if response.status_code == 200:
            try:
                data = response.json()
                if 'records' in data and isinstance(data['records'], list):
                    for article in data['records']:
                        if 'sys_id' in article:
                            article_ids.add(article['sys_id'])

                    print(f"Found {len(article_ids)} article IDs using JSONv2 endpoint")
            except json.JSONDecodeError:
                print("JSONv2 response is not valid JSON")
        else:
            print(f"Error with JSONv2 endpoint: {response.status_code}")
    except Exception as e:
        print(f"Exception with JSONv2 endpoint: {str(e)}")

    return list(article_ids)


def get_article_by_id(article_id, kb_id, kb_title):
    """Get a single article by its ID"""

    # Try the REST API first
    rest_url = f"{base_url}/kb_knowledge/{article_id}"

    try:
        response = requests.get(
            rest_url,
            auth=HTTPBasicAuth(username, password),
            headers={"Accept": "application/json"},
            timeout=30
        )

        if response.status_code == 200:
            data = response.json()
            article = data.get('result', {})

            # Check if article belongs to the correct KB
            article_kb = None
            if "kb_knowledge_base" in article:
                kb_field = article["kb_knowledge_base"]
                if isinstance(kb_field, dict) and "value" in kb_field:
                    article_kb = kb_field["value"]
                elif isinstance(kb_field, str):
                    article_kb = kb_field

            if article_kb and article_kb != kb_id:
                print(f"Article {article_id} belongs to a different KB: {article_kb}")
                return None

            # Process article content
            try:
                article_text = extract_clean_text(article.get('text', ''))
            except Exception as e:
                print(f"Error processing article text: {str(e)}")
                article_text = "Error processing article text."

            # Generate article URL
            article_url = f"https://epfl.service-now.com/kb_view.do?sys_kb_id={article_id}"

            # Extract workflow state
            workflow_state = None
            if "workflow_state" in article:
                workflow_field = article["workflow_state"]
                if isinstance(workflow_field, dict) and "value" in workflow_field:
                    workflow_state = workflow_field["value"]
                elif isinstance(workflow_field, str):
                    workflow_state = workflow_field

            article_data = {
                "kb_id": kb_id,
                "kb_title": kb_title,
                "article_id": article_id,
                "article_number": article.get('number', ''),
                "article_title": article.get('short_description', ''),
                "article_url": article_url,
                "created_on": article.get('sys_created_on', ''),
                "updated_on": article.get('sys_updated_on', ''),
                "workflow_state": workflow_state,
                "content": article_text
            }

            return article_data
        else:
            print(f"Error fetching article {article_id}: {response.status_code}")

            # If the REST API fails, try the JSONv2 endpoint
            jsonv2_url = "https://epfl.service-now.com/kb_knowledge.do?JSONv2"

            jsonv2_params = {
                "sysparm_query": f"sys_id={article_id}",
                "sysparm_limit": 1
            }

            try:
                jsonv2_response = requests.get(
                    jsonv2_url,
                    params=jsonv2_params,
                    auth=HTTPBasicAuth(username, password),
                    headers={"Accept": "application/json"},
                    timeout=30
                )

                if jsonv2_response.status_code == 200:
                    try:
                        jsonv2_data = jsonv2_response.json()
                        if 'records' in jsonv2_data and len(jsonv2_data['records']) > 0:
                            article = jsonv2_data['records'][0]

                            # Check if article belongs to the correct KB
                            if "kb_knowledge_base" in article and article["kb_knowledge_base"] != kb_id:
                                print(f"Article {article_id} belongs to a different KB: {article['kb_knowledge_base']}")
                                return None

                            # Process article content
                            try:
                                article_text = extract_clean_text(article.get('text', ''))
                            except Exception as e:
                                print(f"Error processing article text: {str(e)}")
                                article_text = "Error processing article text."

                            # Generate article URL
                            article_url = f"https://epfl.service-now.com/kb_view.do?sys_kb_id={article_id}"

                            article_data = {
                                "kb_id": kb_id,
                                "kb_title": kb_title,
                                "article_id": article_id,
                                "article_title": article.get('short_description', ''),
                                "article_url": article_url,
                                "workflow_state": article.get('workflow_state', ''),
                                "content": article_text
                            }

                            return article_data
                        else:
                            print(f"Article {article_id} not found in JSONv2 response")
                    except json.JSONDecodeError:
                        print("JSONv2 response is not valid JSON")
                else:
                    print(f"Error fetching article {article_id} via JSONv2: {jsonv2_response.status_code}")
            except Exception as e:
                print(f"Exception fetching article {article_id} via JSONv2: {str(e)}")
    except Exception as e:
        print(f"Exception fetching article {article_id}: {str(e)}")

    return None


def merge_with_existing_articles(new_articles, existing_filename=None):
    """Merge the new articles with existing ones"""
    existing_articles = []

    if existing_filename:
        try:
            with open(existing_filename, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
                existing_articles = existing_data
                print(f"Loaded {len(existing_articles)} articles from {existing_filename}")
        except Exception as e:
            print(f"Error loading existing articles: {str(e)}")
    else:
        # Look for the most recent combined file
        import glob

        combined_files = glob.glob("kb_articles_combined_all_*.json")
        combined_files.sort(reverse=True)  # Sort by filename (which has timestamp)

        if combined_files:
            latest_file = combined_files[0]
            print(f"Found latest combined file: {latest_file}")

            try:
                with open(latest_file, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
                    existing_articles = existing_data
                    print(f"Loaded {len(existing_articles)} articles from {latest_file}")
            except Exception as e:
                print(f"Error loading existing articles: {str(e)}")

    # Merge the articles
    all_article_ids = {article['article_id']: article for article in existing_articles}
    new_count = 0

    for article in new_articles:
        article_id = article['article_id']

        if article_id not in all_article_ids:
            all_article_ids[article_id] = article
            new_count += 1
        elif 'workflow_state' in article and article['workflow_state'] == 'published':
            # If the article is published in the new data, prioritize it
            all_article_ids[article_id] = article

    merged_articles = list(all_article_ids.values())
    print(f"Merged {len(existing_articles)} existing articles with {len(new_articles)} new articles")
    print(f"Added {new_count} new unique articles, total {len(merged_articles)} articles")

    # Filter only published articles for the final count
    published_articles = [a for a in merged_articles if a.get('workflow_state') == 'published']
    print(f"Published articles in merged set: {len(published_articles)}")

    # Save the merged articles
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"kb_articles_combined_all_{timestamp}.json"

    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(merged_articles, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(merged_articles)} merged articles to {filename}")

    # Also save only the published articles
    published_filename = f"kb_articles_published_only_{timestamp}.json"
    with open(published_filename, 'w', encoding='utf-8') as f:
        json.dump(published_articles, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(published_articles)} published articles to {published_filename}")

    return merged_articles, published_articles


# Main execution
if __name__ == "__main__":
    kb_id = "7243ab6347332100158b949b6c9a7194"
    kb_title = "Base de connaissances financière"

    print(f"Attempting to retrieve all articles for '{kb_title}' (ID: {kb_id})")

    # Ask user whether to retrieve all articles or only published ones
    only_published = input("Retrieve only published articles? (y/n): ").lower() == 'y'

    # Try the direct ID lookup approach
    success, articles = retrieve_articles_with_direct_id_lookup(kb_id, kb_title, only_published)

    if success:
        print(f"Successfully retrieved {len(articles)} articles")

        # Merge with existing articles
        merged_articles, published_articles = merge_with_existing_articles(articles)

        # Print final statistics
        print(f"\nFinal Statistics:")
        print(f"Total merged articles: {len(merged_articles)}")
        print(f"Published articles: {len(published_articles)}")

        if len(published_articles) > 0:
            expected_published = 112  # Based on the count you reported
            print(f"Coverage of published articles: {len(published_articles)}/{expected_published} " +
                  f"({len(published_articles) / expected_published * 100:.1f}%)")
    else:
        print("Failed to retrieve articles")