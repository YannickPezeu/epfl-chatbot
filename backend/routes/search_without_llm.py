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
import dotenv
dotenv.load_dotenv(dotenv.find_dotenv())

router = APIRouter(
    prefix="/search",
    tags=["search"]  # This will group your library endpoints in the FastAPI docs
)


from searchEngine.search_engines import search_engine

@router.get("/search/{query}")
async def search(query: str, session_token: str = Cookie(None)):
    conn, cursor = initialize_all_connection()
    username = 'all_users'
    password = os.environ.get('RCP_API_KEY')
    print('search, query:', query)

    results = search_engine(
        username=username,
        library="LEX AND RH",
        text=query,
        n_results=5,
        openai_key=password,
        model_name='rcp'
    )

    conn.close()

    if not results:
        raise HTTPException(status_code=404, detail="No results found")
    # print('results', results)


    return {"results": results}


import asyncio

if __name__ == '__main__':
    async def searchin(text):
        results = await search(text)
        print(results)


    # Créer une boucle d'événements asyncio et exécuter la coroutine
    asyncio.run(searchin('parking prices'))