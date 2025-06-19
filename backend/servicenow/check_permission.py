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

def check_kb_permissions(kb_id):
    """
    Check knowledge base permissions and diagnose access issues
    """
    print(f"\n=== Checking permissions for knowledge base (ID: {kb_id}) ===")

    # Check the knowledge base record
    kb_url = f"https://epfl.service-now.com/api/now/table/kb_knowledge_base/{kb_id}"

    response = requests.get(
        kb_url,
        auth=HTTPBasicAuth(username, password),
        headers={"Accept": "application/json"}
    )

    if response.status_code == 200:
        data = response.json()
        kb_details = data.get('result', {})

        print("\nKnowledge Base Details:")
        print(f"  - Title: {kb_details.get('title')}")
        print(f"  - Active: {kb_details.get('active')}")

        # Check ownership and permissions
        print("\nPermission Information:")

        # Check roles field
        if 'roles' in kb_details:
            roles_value = kb_details.get('roles')
            if isinstance(roles_value, dict):
                print(f"  - Roles: {roles_value.get('display_value', '')}")
            else:
                print(f"  - Roles: {roles_value}")
        else:
            print("  - Roles: Not specified")

        # Check managers field
        if 'kb_managers' in kb_details:
            managers_value = kb_details.get('kb_managers')
            if isinstance(managers_value, dict):
                print(f"  - Managers: {managers_value.get('display_value', '')}")
            else:
                print(f"  - Managers: {managers_value}")
        else:
            print("  - Managers: Not specified")

        # Check owner field
        if 'owner' in kb_details:
            owner_value = kb_details.get('owner')
            if isinstance(owner_value, dict):
                print(f"  - Owner: {owner_value.get('display_value', '')}")
            else:
                print(f"  - Owner: {owner_value}")
        else:
            print("  - Owner: Not specified")

        # Check visibility settings
        for field in ['use_view_default', 'public', 'disable_suggesting']:
            if field in kb_details:
                print(f"  - {field}: {kb_details.get(field)}")

        # Check current user permissions
        print("\nChecking current API user (WS_AI) permissions...")

        # Try to get user information
        user_url = "https://epfl.service-now.com/api/now/table/sys_user?sysparm_query=user_name=WS_AI"

        user_response = requests.get(
            user_url,
            auth=HTTPBasicAuth(username, password),
            headers={"Accept": "application/json"}
        )

        if user_response.status_code == 200:
            user_data = user_response.json()
            user_results = user_data.get('result', [])

            if user_results:
                user = user_results[0]
                print(f"  - User found: {user.get('name')}")
                print(f"  - Active: {user.get('active')}")

                # Check roles
                if 'roles' in user:
                    roles_value = user.get('roles')
                    if isinstance(roles_value, str):
                        print(f"  - Roles: {roles_value}")
                    elif isinstance(roles_value, dict):
                        print(f"  - Roles: {roles_value.get('display_value', '')}")
                else:
                    # Get roles through a different API
                    roles_url = f"https://epfl.service-now.com/api/now/table/sys_user_has_role?sysparm_query=user={user.get('sys_id')}"

                    roles_response = requests.get(
                        roles_url,
                        auth=HTTPBasicAuth(username, password),
                        headers={"Accept": "application/json"}
                    )

                    if roles_response.status_code == 200:
                        roles_data = roles_response.json()
                        roles_results = roles_data.get('result', [])

                        print(f"  - User has {len(roles_results)} roles")

                        for i, role in enumerate(roles_results, 1):
                            role_name = "Unknown"
                            if 'role' in role and isinstance(role['role'], dict):
                                role_name = role['role'].get('display_value', 'Unknown')
                            print(f"    {i}. {role_name}")
            else:
                print("  - User information not found")
        else:
            print(f"  - Error retrieving user information: {user_response.status_code}")

        # Try to determine access scope
        print("\nTesting article visibility...")

        # Try different query parameters to determine what we can see
        test_params = [
            {"query": f"kb_knowledge_base={kb_id}", "description": "All articles"},
            {"query": f"kb_knowledge_base={kb_id}^workflow_state=published", "description": "Published articles"},
            {"query": f"kb_knowledge_base={kb_id}^workflow_state=published^active=true",
             "description": "Published and active"}
        ]

        for test in test_params:
            params = {
                "sysparm_query": test["query"],
                "sysparm_limit": 1,
                "sysparm_fields": "sys_id,short_description,workflow_state,active"
            }

            test_response = requests.get(
                "https://epfl.service-now.com/api/now/table/kb_knowledge",
                params=params,
                auth=HTTPBasicAuth(username, password),
                headers={"Accept": "application/json"}
            )

            if test_response.status_code == 200:
                test_data = test_response.json()
                test_results = test_data.get('result', [])

                # Check headers for total count
                total_count = test_response.headers.get('X-Total-Count')

                print(f"  - {test['description']}: ", end="")
                if total_count:
                    print(f"Found {total_count} articles")
                else:
                    print(f"Found {len(test_results)} articles (sample)")
            else:
                print(f"  - {test['description']}: Error {test_response.status_code}")

        # Suggest solutions based on findings
        print("\nSuggestions:")
        print(
            "1. If you can see articles in the count but can't retrieve them, you might need additional roles or permissions")
        print("2. Check if the knowledge base has specific visibility settings that might be restricting access")
        print("3. Verify if the articles are in a specific workflow state that your user can't access")
        print("4. Contact the ServiceNow administrator to request additional permissions if needed")
    else:
        print(f"Error retrieving knowledge base details: {response.status_code} - {response.text}")


# To test this function
if __name__ == "__main__":
    kb_id = "7243ab6347332100158b949b6c9a7194"
    check_kb_permissions(kb_id)