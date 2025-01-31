from redis import Redis
from typing import Optional, Any
import json
from datetime import timedelta
import os

class AgentSession:
    '''AgenSession object is used to access the details of the chat agent and the chat history'''
    def __init__(self, username, agent, model_name, embedding_model, library, n_documents_searched_no_llm, n_documents_searched, mistral_key=None, openai_key=None, groq_key=None, interaction_type='chat', rerank=False, special_prompt=None, conversation_id=None):
        self.username = username
        self.agent = agent
        self.model_name = model_name
        self.chat_history = []
        self.embedding_model = embedding_model
        self.library = library
        self.n_documents_searched_no_llm = n_documents_searched_no_llm
        self.n_documents_searched = n_documents_searched
        self.mistral_key = mistral_key
        self.openai_key = openai_key
        self.groq_key = groq_key
        self.interaction_type = interaction_type
        self.rerank = rerank
        self.special_prompt = special_prompt
        self.conversation_id = conversation_id

class RedisStateManager:
    def __init__(self, host: str = 'localhost', port: int = 6379, db: int = 0, decode_responses: bool = True):
        """Initialize Redis connection"""
        self.redis_client = Redis(
            host=os.getenv('REDIS_HOST', 'localhost'),
            port=int(os.getenv('REDIS_PORT', 6379)),
            db=int(os.getenv('REDIS_DB', 0)),
            decode_responses=decode_responses
        )

    def set_state(self, key: str, value: Any, expiry: Optional[int] = None) -> bool:
        """
        Store state in Redis
        Args:
            key: Unique identifier for the state
            value: Data to store (will be JSON serialized)
            expiry: Optional TTL in seconds
        """
        try:
            serialized_value = json.dumps(value)
            if expiry:
                return self.redis_client.setex(key, expiry, serialized_value)
            return self.redis_client.set(key, serialized_value)
        except Exception as e:
            print(f"Error setting state: {e}")
            return False

    def get_state(self, key: str) -> Optional[Any]:
        """
        Retrieve state from Redis
        Args:
            key: Unique identifier for the state
        """
        try:
            value = self.redis_client.get(key)
            return json.loads(value) if value else None
        except Exception as e:
            print(f"Error getting state: {e}")
            return None

    def delete_state(self, key: str) -> bool:
        """
        Delete state from Redis
        Args:
            key: Unique identifier for the state
        """
        try:
            return bool(self.redis_client.delete(key))
        except Exception as e:
            print(f"Error deleting state: {e}")
            return False

    def update_state(self, key: str, value: Any) -> Optional[Any]:
        """
        Update existing state in Redis
        Args:
            key: Unique identifier for the state
            value: New value to store
        """
        try:
            # Get existing state
            existing_state = self.get_state(key)
            if existing_state is None:
                return None

            # If existing state is a dict, update it
            if isinstance(existing_state, dict) and isinstance(value, dict):
                existing_state.update(value)
                self.set_state(key, existing_state)
                return existing_state

            # Otherwise just set the new value
            self.set_state(key, value)
            return value
        except Exception as e:
            print(f"Error updating state: {e}")
            return None

    def handle_uploaded_files(self, ws_connection_id: str, action: str, file_data: dict = None,
                              filename_to_remove: str = None) -> dict:
        """
        Handle uploaded files in Redis
        actions: 'get', 'add', 'remove', 'clear'
        """
        key = f"uploaded_files:{ws_connection_id}"

        try:
            if action == "get":
                files = self.get_state(key) or []
                return files

            elif action == "add":
                files = self.get_state(key) or []
                files.append(file_data)
                self.set_state(key, files)
                return files

            elif action == "remove":
                files = self.get_state(key) or []
                files = [f for f in files if f['filename'] != filename_to_remove]
                self.set_state(key, files)
                return files

            elif action == "clear":
                self.delete_state(key)
                return []

        except Exception as e:
            print(f"Error handling uploaded files: {e}")
            return []

    # def set_agent_session(self, ws_connection_id: str, agent_session, expiry: int = 4*3600) -> bool:
    #     """Store agent session in Redis"""
    #     try:
    #         # Convert object attributes to dictionary
    #         session_data = {
    #             'username': agent_session.username,
    #             'agent': agent_session.agent,
    #             'model_name': agent_session.model_name,
    #             'chat_history': agent_session.chat_history,
    #             'embedding_model': agent_session.embedding_model,
    #             'library': agent_session.library,
    #             'n_documents_searched_no_llm': agent_session.n_documents_searched_no_llm,
    #             'n_documents_searched': agent_session.n_documents_searched,
    #             'mistral_key': agent_session.mistral_key,
    #             'openai_key': agent_session.openai_key,
    #             'groq_key': agent_session.groq_key,
    #             'interaction_type': agent_session.interaction_type,
    #             'rerank': agent_session.rerank,
    #             'special_prompt': agent_session.special_prompt,
    #             'conversation_id': agent_session.conversation_id
    #         }
    #         return self.set_state(f"agent_session:{ws_connection_id}", session_data, expiry=expiry)
    #     except Exception as e:
    #         print(f"Error setting agent session: {e}")
    #         return False
    #
    # def get_agent_session(self, ws_connection_id: str) -> Optional[AgentSession]:
    #     """Retrieve agent session from Redis"""
    #     try:
    #         session_data = self.get_state(f"agent_session:{ws_connection_id}")
    #         if session_data:
    #             return AgentSession(**session_data)
    #         return None
    #     except Exception as e:
    #         print(f"Error getting agent session: {e}")
    #         return None
    #
    # def delete_agent_session(self, ws_connection_id: str) -> bool:
    #     """Delete agent session from Redis"""
    #     try:
    #         return self.delete_state(f"agent_session:{ws_connection_id}")
    #     except Exception as e:
    #         print(f"Error deleting agent session: {e}")
    #         return False

