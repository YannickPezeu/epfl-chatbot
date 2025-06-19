import os
import secrets

import bcrypt
from fastapi import HTTPException, APIRouter, Cookie
from pydantic import BaseModel

from myUtils.connect_acad2 import initialize_all_connection
from myUtils.ask_chatGPT import ask_chatGPT
import base64
from myUtils.connect_acad2 import reconnect_on_failure
from fastapi import Response


router = APIRouter(
    prefix="/auth",
    tags=["auth"]  # This will group your library endpoints in the FastAPI docs
)

class OpenAIKeyUpload(BaseModel):
    openai_key: str

from typing import Optional

CIPHER_KEY = os.environ.get('CIPHER_KEY')

@router.get('/get-username')
def get_username(session_token: str = Cookie(None)):
    conn, cursor = initialize_all_connection()
    username = get_username_from_session_token(session_token, conn, cursor)
    conn.close()
    return {"username": username}

def set_api_key_status(username, status, conn=None, cursor=None):
    if cursor is None:
        conn, cursor = initialize_all_connection()
    cursor.execute("UPDATE users SET openai_key_status=%s WHERE username=%s", (status, username))
    conn.commit()
    conn.close()

def check_session_helper(session_token, conn=None, cursor=None):
    if session_token is None:
        return {"is_logged_in": False}
    if cursor is None:
        conn, cursor = initialize_all_connection()
    cursor.execute('''SELECT username, session_token from users''')
    myUsers = cursor.fetchall()
    dico_users_session_tokens = {u[0]: u[1] for u in myUsers}
    conn.close()

    for username in dico_users_session_tokens:
        # print('username', username, 'token', dico_users_session_tokens[username])
        if session_token == dico_users_session_tokens[username]:
            return {"is_logged_in": True, "username": username}

    return {"is_logged_in": False}



def get_username_from_session_token(session_token, conn=None, cursor=None):
    # use the check_session function to check if the user is authenticated
    if session_token is None:
        raise HTTPException(status_code=401, detail="Not authenticated")

    if not session_token:
        raise HTTPException(status_code=401, detail="No session token provided")

    checked_session = check_session_helper(session_token, conn, cursor)

    if 'username' not in checked_session:
        raise HTTPException(status_code=401, detail="Invalid session token")

    username = checked_session['username']

    # print(f"Authenticated username: {username}")

    return username


@router.get("/check_session")
async def check_session(session_token: str = Cookie(None)):
    response = check_session_helper(session_token)
    return response


def xor_encrypt(data: str, key: str) -> str:
    xored = ''.join(chr(ord(c) ^ ord(key[i % len(key)])) for i, c in enumerate(data))
    return base64.b64encode(xored.encode()).decode()

def xor_decrypt(encrypted: str, key: str) -> str:
    xored = base64.b64decode(encrypted).decode()
    return ''.join(chr(ord(c) ^ ord(key[i % len(key)])) for i, c in enumerate(xored))

@router.post('/upload-openai-key')
async def upload_openai_key(key_data: OpenAIKeyUpload, session_token: str = Cookie(None)):

    encrypted_key = xor_encrypt(key_data.openai_key, CIPHER_KEY)

    if session_token is None:
        raise HTTPException(status_code=401, detail="Not authenticated")

    username = get_username_from_session_token(session_token)

    # check validity of openai key
    keyStatus = await check_openai_key_helper(openai_key=key_data.openai_key, username=username)

    # print('key',key_data.openai_key, 'keyStatus', keyStatus)
    if keyStatus['openaiKeyStatus'] == 'invalid':
        raise HTTPException(status_code=400, detail="Invalid OpenAI key")

    elif keyStatus['openaiKeyStatus'] == 'missing':
        raise HTTPException(status_code=400, detail="OpenAI key missing")

    else:
        username = get_username_from_session_token(session_token)
        conn, cursor = initialize_all_connection()
        cursor.execute(
            "UPDATE users SET openai_key=%s, openai_key_status='valid' WHERE username=%s",
            (encrypted_key, username)
        )
        # print('username', username)
        conn.commit()
        conn.close()
        return {"success": True}


def get_openai_key(username: str, conn=None, cursor=None):
    if cursor is None:
        conn, cursor = initialize_all_connection()
    cursor.execute("SELECT openai_key, openai_key_status FROM users WHERE username=%s", (username,))
    encrypted_key ,openai_key_status = cursor.fetchone()
    conn.close()
    return xor_decrypt(encrypted_key, CIPHER_KEY), openai_key_status


@router.get("/check-openai-key")
async def check_openai_key(session_token: str = Cookie(None)):
    conn, cursor = initialize_all_connection()
    username = get_username_from_session_token(session_token, conn, cursor)
    openai_key, openai_key_status = get_openai_key(username, conn=conn, cursor=cursor)
    # print('openai_key', openai_key, 'openai_key_status', openai_key_status)
    if openai_key_status is None:
        openai_key_status = await check_openai_key_helper(openai_key=openai_key, username=username)
        status = openai_key_status['openaiKeyStatus']
        cursor.execute("UPDATE users SET openai_key_status=%s WHERE username=%s", (status, username))
        conn.commit()
    else:
        openai_key_status = {"openaiKeyStatus": openai_key_status}
    return openai_key_status


async def check_openai_key_helper(openai_key, username=None):
    # print('openai_key', openai_key)
    if openai_key is None:
        print('openai_key is None')
        return {"openaiKeyStatus": 'missing'}

    else:
        try:
            ask_chatGPT(
                prompt="This is a test",
                openai_key=openai_key,
                max_tokens=5,
                max_retries=1
            )
            return {"openaiKeyStatus": 'valid'}
        except Exception as e:
            return {"openaiKeyStatus": 'invalid'}




@router.post("/signup")
@reconnect_on_failure
async def signup(username: str, password: str, response: Response):
    print('username', username, 'password', password)

    conn, cursor = initialize_all_connection()
    # check if user exists in table users
    cursor.execute("SELECT * FROM users WHERE username=%s", (username,))
    user = cursor.fetchone()
    if user:
        raise HTTPException(status_code=400, detail="Username already exists")


    # Hash the password
    hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
    print('hashed_password', hashed_password)

    # Create a session for the new user
    session_token = secrets.token_urlsafe(16)
    # sessions[session_token] = username

    key = 'no_key_yet_no_key_yet_no_key_yet'
    encrypted_key = xor_encrypt(key, CIPHER_KEY)

    cursor.execute("INSERT INTO users (username, password, session_token, openai_key, openai_key_status) VALUES (%s, %s, %s, %s, %s)",
                   (username, hashed_password.decode('utf-8'), session_token, encrypted_key, 'invalid'))
    conn.commit()
    conn.close()

    # Set the session cookie
    response.set_cookie(
        key="session_token",
        value=session_token,
        httponly=False,
        secure=False,
        samesite="strict",
        max_age=3600*24*365  # 1 year
    )

    return {"success": True, "message": "User created successfully"}

from fastapi import Request

@router.post("/logout")
async def logout(request: Request, response: Response, session_token: str = Cookie(None)):
    # print('session_token', session_token)
    # conn, cursor = initialize_all_connection()
    # cursor.execute("UPDATE users SET session_token=%s WHERE session_token=%s", (None, session_token))
    # conn.commit()
    # conn.close()
    # print('logged out')
    host = request.headers.get("host", "").split(":")[0]

    response.delete_cookie(key="session_token", httponly=False, secure=False, samesite="lax",
                           domain=host if host != "127.0.0.1" else None,)
    return {"success": True}

@router.post("/login")
@reconnect_on_failure
def login(request: Request, username: str, password: str, response: Response):
    conn, cursor = initialize_all_connection()
    try:
        cursor.execute("SELECT * FROM users WHERE username=%s", (username,))
        user = cursor.fetchone()
        print('user', user)

        if not user:
            raise HTTPException(status_code=401, detail="Invalid username")

        #check if password is correct first
        if not bcrypt.checkpw(password.encode('utf-8'), user[1].encode('utf-8')):
            raise HTTPException(status_code=401, detail="Invalid password")

        #check if session_token exists
        if user[2] is not None:
            session_token = user[2]
        else:
            # create session
            session_token = secrets.token_urlsafe(16)
            cursor.execute("UPDATE users SET session_token=%s WHERE username=%s",
                           (session_token, username))
            conn.commit()

        response.set_cookie(
            key="session_token",
            value=session_token,
            httponly=False,
            secure=False,
            samesite="lax",
            max_age=3600*24*365,
            path = "/"
        )
        return {"success": True, "session_token": session_token}

    except HTTPException:
        raise
    except Exception as e:
        print('error', e)
        raise HTTPException(status_code=401, detail="Invalid username")
    finally:
        print('closing connection')
        if cursor:
            print('closing cursor')
            cursor.close()
        if conn:
            print('closing connection')
            conn.close()