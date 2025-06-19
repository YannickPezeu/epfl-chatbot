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

# Base URL and knowledge base ID
kb_id = "7243ab6347332100158b949b6c9a7194"  # Specified KB ID

from servicenow_get_kb import extract_clean_text


def retrieve_by_chunks(kb_id, kb_title, chunk_size=10):
    """
    Try to retrieve articles by breaking up the retrieval into smaller chunks using
    different sorting and filtering approaches
    """
    print(f"\n=== Retrieving articles in chunks from '{kb_title}' (ID: {kb_id}) ===")

    # Track processed article IDs to avoid duplicates
    processed_article_ids = set()
    all_articles = []

    # Different field sorting options to try
    sort_options = [
        "sys_created_on",
        "sys_updated_on",
        "number",
        "short_description"
    ]

    # Different directions to sort
    directions = ["desc", "asc"]

    total_retrieved = 0

    # REST API URL
    rest_url = "https://epfl.service-now.com/api/now/table/kb_knowledge"

    # Try different sorting options to get different batches of articles
    for sort_field in sort_options:
        for direction in directions:
            print(f"\nTrying sort by {sort_field} {direction}")

            params = {
                "sysparm_query": f"kb_knowledge_base={kb_id}^workflow_state=published^active=true^ORDERBYsys_updated_on",
                "sysparm_limit": chunk_size,
                "sysparm_display_value": "true",
                "sysparm_fields": "sys_id,short_description,text,number,sys_created_on,sys_updated_on,workflow_state",
                "sysparm_order_by": sort_field,
                "sysparm_order_by_direction": direction
            }

            try:
                response = requests.get(
                    rest_url,
                    params=params,
                    auth=HTTPBasicAuth(username, password),
                    headers={"Accept": "application/json"},
                    timeout=30
                )

                print(f"Status code: {response.status_code}")

                if response.status_code == 200:
                    data = response.json()
                    batch_articles = data.get('result', [])
                    batch_size = len(batch_articles)

                    print(f"Retrieved {batch_size} articles in this batch")

                    # Check if we got any new articles
                    new_articles_count = 0

                    for article in batch_articles:
                        article_id = article.get('sys_id', '')

                        # Skip if we've already processed this article
                        if article_id in processed_article_ids:
                            continue

                        # Mark this article as processed
                        processed_article_ids.add(article_id)
                        new_articles_count += 1

                        # Debug info
                        print(
                            f"Processing new article: {article.get('short_description', '')[:30]}... (ID: {article_id})")

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
                            "article_number": article.get('number', ''),
                            "article_title": article.get('short_description', ''),
                            "article_url": article_url,
                            "created_on": article.get('sys_created_on', ''),
                            "updated_on": article.get('sys_updated_on', ''),
                            "workflow_state": article.get('workflow_state', ''),
                            "content": article_text
                        }

                        # Add to articles data list
                        all_articles.append(article_data)

                    total_retrieved += new_articles_count
                    print(f"Added {new_articles_count} new articles, total unique articles so far: {total_retrieved}")
                else:
                    print(f"Error fetching articles: {response.status_code}")
            except Exception as e:
                print(f"Exception during request: {str(e)}")

            # Add a short delay between requests
            time.sleep(1)

    # Write articles to JSON file if we retrieved any
    if all_articles:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"kb_articles_chunks_{kb_id}_{timestamp}.json"

        with open(filename, "w", encoding="utf-8") as f:
            json.dump(all_articles, f, ensure_ascii=False, indent=2)

        print(f"\nSaved {len(all_articles)} unique articles to {filename}")

    return len(all_articles) > 0, all_articles


def retrieve_with_date_ranges(kb_id, kb_title):
    """
    Try to retrieve articles by breaking the query into date ranges
    """
    print(f"\n=== Retrieving articles with date ranges from '{kb_title}' (ID: {kb_id}) ===")

    # Track processed article IDs to avoid duplicates
    processed_article_ids = set()
    all_articles = []

    # Define date ranges to try (year by year)
    date_ranges = [
        ("2015-01-01", "2015-12-31"),
        ("2016-01-01", "2016-12-31"),
        ("2017-01-01", "2017-12-31"),
        ("2018-01-01", "2018-12-31"),
        ("2019-01-01", "2019-12-31"),
        ("2020-01-01", "2020-12-31"),
        ("2021-01-01", "2021-12-31"),
        ("2022-01-01", "2022-12-31"),
        ("2023-01-01", "2023-12-31"),
        ("2024-01-01", "2024-12-31"),
        ("2025-01-01", "2025-12-31")
    ]

    total_retrieved = 0

    # REST API URL
    rest_url = "https://epfl.service-now.com/api/now/table/kb_knowledge"

    # Try different date ranges
    for start_date, end_date in date_ranges:
        print(f"\nTrying date range {start_date} to {end_date}")

        params = {
            "sysparm_query": f"kb_knowledge_base={kb_id}^workflow_state=published^active=true^sys_created_onBETWEEN{start_date}@{end_date}",
            "sysparm_limit": 100,
            "sysparm_display_value": "true",
            "sysparm_fields": "sys_id,short_description,text,number,sys_created_on,sys_updated_on,workflow_state"
        }

        try:
            response = requests.get(
                rest_url,
                params=params,
                auth=HTTPBasicAuth(username, password),
                headers={"Accept": "application/json"},
                timeout=30
            )

            print(f"Status code: {response.status_code}")

            if response.status_code == 200:
                data = response.json()
                batch_articles = data.get('result', [])
                batch_size = len(batch_articles)

                print(f"Retrieved {batch_size} articles in this batch")

                # Check if we got any new articles
                new_articles_count = 0

                for article in batch_articles:
                    article_id = article.get('sys_id', '')

                    # Skip if we've already processed this article
                    if article_id in processed_article_ids:
                        continue

                    # Mark this article as processed
                    processed_article_ids.add(article_id)
                    new_articles_count += 1

                    # Debug info
                    print(f"Processing new article: {article.get('short_description', '')[:30]}... (ID: {article_id})")

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
                        "article_number": article.get('number', ''),
                        "article_title": article.get('short_description', ''),
                        "article_url": article_url,
                        "created_on": article.get('sys_created_on', ''),
                        "updated_on": article.get('sys_updated_on', ''),
                        "workflow_state": article.get('workflow_state', ''),
                        "content": article_text
                    }

                    # Add to articles data list
                    all_articles.append(article_data)

                total_retrieved += new_articles_count
                print(f"Added {new_articles_count} new articles, total unique articles so far: {total_retrieved}")
            else:
                print(f"Error fetching articles: {response.status_code}")
        except Exception as e:
            print(f"Exception during request: {str(e)}")

        # Add a short delay between requests
        time.sleep(1)

    # Write articles to JSON file if we retrieved any
    if all_articles:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"kb_articles_dates_{kb_id}_{timestamp}.json"

        with open(filename, "w", encoding="utf-8") as f:
            json.dump(all_articles, f, ensure_ascii=False, indent=2)

        print(f"\nSaved {len(all_articles)} unique articles to {filename}")

    return len(all_articles) > 0, all_articles


def retrieve_with_advanced_queries(kb_id, kb_title):
    """
    Try to retrieve articles using alternative query approaches
    """
    print(f"\n=== Retrieving articles with advanced queries from '{kb_title}' (ID: {kb_id}) ===")

    # Track processed article IDs to avoid duplicates
    processed_article_ids = set()
    all_articles = []

    # Different query approaches to try
    query_approaches = [
        # Try with different knowledge base reference formats
        {
            "name": "Using sys_id reference",
            "query": f"kb_knowledge_base.sys_id={kb_id}^workflow_state=published^active=true"
        },
        # Try without specifying active flag
        {
            "name": "Published only",
            "query": f"kb_knowledge_base={kb_id}^workflow_state=published"
        },
        # Try with advanced query operators
        {
            "name": "Using IN operator",
            "query": f"kb_knowledge_base={kb_id}^workflow_stateINpublished^active=true"
        },
        # Try with different workflow states
        {
            "name": "All workflow states",
            "query": f"kb_knowledge_base={kb_id}^workflow_stateIN published,draft,retired^active=true"
        },
        # Try using STARTSWITH for title
        {
            "name": "Titles starting with A-E",
            "query": f"kb_knowledge_base={kb_id}^workflow_state=published^active=true^short_descriptionSTARTSWITHA^ORshort_descriptionSTARTSWITHB^ORshort_descriptionSTARTSWITHC^ORshort_descriptionSTARTSWITHD^ORshort_descriptionSTARTSWITHE"
        },
        {
            "name": "Titles starting with F-J",
            "query": f"kb_knowledge_base={kb_id}^workflow_state=published^active=true^short_descriptionSTARTSWITHF^ORshort_descriptionSTARTSWITHG^ORshort_descriptionSTARTSWITHH^ORshort_descriptionSTARTSWITHI^ORshort_descriptionSTARTSWITHJ"
        },
        {
            "name": "Titles starting with K-O",
            "query": f"kb_knowledge_base={kb_id}^workflow_state=published^active=true^short_descriptionSTARTSWITHK^ORshort_descriptionSTARTSWITHL^ORshort_descriptionSTARTSWITHM^ORshort_descriptionSTARTSWITHN^ORshort_descriptionSTARTSWITHO"
        },
        {
            "name": "Titles starting with P-T",
            "query": f"kb_knowledge_base={kb_id}^workflow_state=published^active=true^short_descriptionSTARTSWITHP^ORshort_descriptionSTARTSWITHQ^ORshort_descriptionSTARTSWITHR^ORshort_descriptionSTARTSWITHS^ORshort_descriptionSTARTSWITHT"
        },
        {
            "name": "Titles starting with U-Z",
            "query": f"kb_knowledge_base={kb_id}^workflow_state=published^active=true^short_descriptionSTARTSWITHU^ORshort_descriptionSTARTSWITHV^ORshort_descriptionSTARTSWITHW^ORshort_descriptionSTARTSWITHX^ORshort_descriptionSTARTSWITHY^ORshort_descriptionSTARTSWITHZ"
        },
        # Try with encoded query from browser
        {
            "name": "Encoded query",
            "query": f"kb_knowledge_base%3D{kb_id}%5Eworkflow_state%3Dpublished%5Eactive%3Dtrue"
        }
    ]

    total_retrieved = 0

    # REST API URL
    rest_url = "https://epfl.service-now.com/api/now/table/kb_knowledge"

    # Try different query approaches
    for approach in query_approaches:
        print(f"\nTrying approach: {approach['name']}")

        params = {
            "sysparm_query": approach["query"],
            "sysparm_limit": 100,
            "sysparm_display_value": "true",
            "sysparm_fields": "sys_id,short_description,text,number,sys_created_on,sys_updated_on,workflow_state"
        }

        try:
            response = requests.get(
                rest_url,
                params=params,
                auth=HTTPBasicAuth(username, password),
                headers={"Accept": "application/json"},
                timeout=30
            )

            print(f"Status code: {response.status_code}")

            if response.status_code == 200:
                data = response.json()
                batch_articles = data.get('result', [])
                batch_size = len(batch_articles)

                print(f"Retrieved {batch_size} articles in this batch")

                # Check if we got any new articles
                new_articles_count = 0

                for article in batch_articles:
                    article_id = article.get('sys_id', '')

                    # Skip if we've already processed this article
                    if article_id in processed_article_ids:
                        continue

                    # Mark this article as processed
                    processed_article_ids.add(article_id)
                    new_articles_count += 1

                    # Debug info
                    print(f"Processing new article: {article.get('short_description', '')[:30]}... (ID: {article_id})")

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
                        "article_number": article.get('number', ''),
                        "article_title": article.get('short_description', ''),
                        "article_url": article_url,
                        "created_on": article.get('sys_created_on', ''),
                        "updated_on": article.get('sys_updated_on', ''),
                        "workflow_state": article.get('workflow_state', ''),
                        "content": article_text
                    }

                    # Add to articles data list
                    all_articles.append(article_data)

                total_retrieved += new_articles_count
                print(f"Added {new_articles_count} new articles, total unique articles so far: {total_retrieved}")
            else:
                print(f"Error fetching articles: {response.status_code}")
        except Exception as e:
            print(f"Exception during request: {str(e)}")

        # Add a short delay between requests
        time.sleep(1)

    # Write articles to JSON file if we retrieved any
    if all_articles:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"kb_articles_advanced_{kb_id}_{timestamp}.json"

        with open(filename, "w", encoding="utf-8") as f:
            json.dump(all_articles, f, ensure_ascii=False, indent=2)

        print(f"\nSaved {len(all_articles)} unique articles to {filename}")

    return len(all_articles) > 0, all_articles


def combine_results(results_list):
    """
    Combine results from multiple retrieval approaches
    """
    print("\n=== Combining results from all approaches ===")

    # Track processed article IDs to avoid duplicates
    processed_article_ids = set()
    combined_articles = []

    # Process all sets of articles
    for success, articles in results_list:
        if success:
            for article in articles:
                article_id = article["article_id"]

                # Skip if we've already processed this article
                if article_id in processed_article_ids:
                    continue

                # Mark this article as processed
                processed_article_ids.add(article_id)

                # Add to combined articles list
                combined_articles.append(article)

    print(f"Combined {len(combined_articles)} unique articles from all approaches")

    # Write combined articles to JSON file
    if combined_articles:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"kb_articles_combined_all_{timestamp}.json"

        with open(filename, "w", encoding="utf-8") as f:
            json.dump(combined_articles, f, ensure_ascii=False, indent=2)

        print(f"Saved combined results to {filename}")

    return combined_articles


# Main execution
if __name__ == "__main__":
    kb_id = "7243ab6347332100158b949b6c9a7194"
    kb_title = "Base de connaissances financière"

    # List to store results from each approach
    results_list = []

    # 1. Try retrieving by chunks with different sorting options
    success1, articles1 = retrieve_by_chunks(kb_id, kb_title)
    results_list.append((success1, articles1))

    # 2. Try retrieving with date ranges
    success2, articles2 = retrieve_with_date_ranges(kb_id, kb_title)
    results_list.append((success2, articles2))

    # 3. Try retrieving with advanced queries
    success3, articles3 = retrieve_with_advanced_queries(kb_id, kb_title)
    results_list.append((success3, articles3))

    # Combine all the results
    combined_articles = combine_results(results_list)

    # Print final summary
    print("\n=== Final Summary ===")
    print(f"Total unique articles retrieved: {len(combined_articles)}")
    print(f"Percentage of published articles retrieved: {(len(combined_articles) / 112) * 100:.1f}%")