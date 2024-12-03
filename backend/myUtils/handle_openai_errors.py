
from openai import AuthenticationError, APIError
from routes.auth import get_username_from_session_token, set_api_key_status
from functools import wraps
from fastapi import  HTTPException

def handle_openai_errors(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            # print('func', func)
            return await func(*args, **kwargs)
        except (AuthenticationError, APIError) as e:
            error_message = str(e)
            if hasattr(e, 'response'):
                status_code = e.response.status_code
                try:
                    error_data = e.response.json()
                    error_code = error_data.get('error', {}).get('code')
                    error_type = error_data.get('error', {}).get('type')
                except ValueError:
                    error_code = None
                    error_type = None
            else:
                status_code = 500
                error_code = None
                error_type = None
            if error_code == 'invalid_api_key':
            # Extract the session token from kwargs
                session_token = kwargs.get('session_token')
                # print('session_token', session_token)
                if not session_token:
                    # If session_token is not in kwargs, check if it's a Cookie in the request
                    request = kwargs.get('request')
                    if request:
                        session_token = request.cookies.get('session_token')

                if session_token:
                    username = get_username_from_session_token(session_token)
                    set_api_key_status(username, 'invalid')
                else:
                    username = kwargs.get('username')
                    # print('username test', username)
                    # print('kwargs', kwargs)
                    if username:
                        set_api_key_status(username, 'invalid')
                raise HTTPException(status_code=401, detail="Invalid OpenAI API key")
            elif error_code:
                raise HTTPException(status_code=status_code, detail=f"OpenAI API error: {error_code}")
            else:
                raise HTTPException(status_code=status_code, detail=f"OpenAI API error: {error_message}")

        except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

    return wrapper

