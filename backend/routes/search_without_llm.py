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
import logging
from typing import List

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/search",
    tags=["search"]  # This will group your library endpoints in the FastAPI docs
)


from searchEngine.search_engines import search_engine

@router.get("/LEX_AND_RH/{query}")
async def LEX_AND_RH(query: str, n_results: int = 5, session_token: str = Cookie(None)):
    conn, cursor = initialize_all_connection()
    username = 'all_users'
    password = os.environ.get('RCP_API_KEY')
    print('search, query:', query)

    results = search_engine(
        username=username,
        library="LEX AND RH",
        text=query,
        n_results=n_results,
        openai_key=password,
        model_name='rcp',
        rerank=True
    )

    conn.close()

    if not results:
        raise HTTPException(status_code=404, detail="No results found")
    # print('results', results)

    return {"results": results}

@router.get("/servicenow_KB/{query}")
async def servicenow_KB(query: str, n_results: int = 5, session_token: str = Cookie(None)):
    conn, cursor = initialize_all_connection()
    username = 'test'
    password = os.environ.get('RCP_API_KEY')
    print('search, query:', query)

    results = search_engine(
        username=username,
        library="servicenow_KB",
        text=query,
        n_results=n_results,
        openai_key=password,
        model_name='rcp',
        rerank=True
    )

    conn.close()

    if not results:
        raise HTTPException(status_code=404, detail="No results found")
    # print('results', results)

    return {"results": results}

@router.get("/servicenow_finance/{query}")
async def servicenow_finance(query: str, n_results: int = 5, session_token: str = Cookie(None)):
    conn, cursor = initialize_all_connection()
    username = 'servicenow_user'
    password = os.environ.get('RCP_API_KEY')
    print('search, query:', query)

    results = search_engine(
        username=username,
        library="servicenow_finance",
        text=query,
        n_results=n_results,
        openai_key=password,
        model_name='rcp'
    )

    conn.close()

    if not results:
        raise HTTPException(status_code=404, detail="No results found")
    # print('results', results)

    return {"results": results}
@router.get("/servicenow_KB_error_test/{query}")
async def servicenow_KB(query: str, session_token: str = Cookie(None)):
    conn, cursor = initialize_all_connection()
    username = 'test'
    password = os.environ.get('RCP_API_KEY')
    logger.info('search, query error_test:', query)

    results = search_engine(
        username=username,
        library="servicenow_KB",
        text=query,
        n_results=5,
        openai_key=password,
        model_name='rcp_error_test'
    )

    conn.close()

    if not results:
        raise HTTPException(status_code=404, detail="No results found")
    # print('results', results)


    return {"results": results}




if __name__ == '__main__':
    import asyncio
    async def searchin(text):
        results = await servicenow_KB(text)
        print(results)


    # Créer une boucle d'événements asyncio et exécuter la coroutine
    asyncio.run(searchin('Quels facteurs influencent les performances de transfert avec ENACSHARE ?'))