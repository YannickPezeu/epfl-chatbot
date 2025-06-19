import json
import os
import time

# import sqlite3
# import time
#
# import faiss

from myUtils.get_embeddings import get_embeddings
# from sentence_transformers import CrossEncoder
from myUtils.connect_acad2 import reconnect_on_failure, initialize_all_connection, get_db_connection
from library_creation._3_create_faiss_index import retrieve_faiss_index

import logging
from pymysql.err import InterfaceError, OperationalError
import requests

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import tiktoken

tiktoken_encoding = tiktoken.get_encoding("cl100k_base")
from langchain.agents import tool, create_tool_calling_agent

# Add reranking configuration
RCP_API_KEY = os.getenv("RCP_API_KEY")
RCP_API_ENDPOINT = 'https://inference-dev.rcp.epfl.ch/v1'


def rerank_documents(query, documents, model="BAAI/bge-reranker-v2-m3", api_key=None, api_endpoint=None):
    """
    Rerank documents using the BGE reranker model via RCP API.

    Args:
        query (str): The search query
        documents (list): List of document texts to rerank
        model (str): The reranker model to use
        api_key (str): API key for RCP service
        api_endpoint (str): API endpoint for RCP service

    Returns:
        list: Reranked document indices in order of relevance
    """
    if not api_key:
        api_key = RCP_API_KEY
    if not api_endpoint:
        api_endpoint = RCP_API_ENDPOINT

    if not api_key:
        logger.warning("No RCP API key provided, skipping reranking")
        return list(range(len(documents)))

    if not documents:
        return []

    try:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        data = {
            "model": model,
            "query": query,
            "documents": documents,
            "top_k": len(documents)
        }

        response = requests.post(
            f"{api_endpoint}/rerank",
            headers=headers,
            json=data,
            timeout=30
        )

        response.raise_for_status()
        result = response.json()

        if "results" in result:
            # Extract the reranked indices
            reranked_indices = [item.get("index", i) for i, item in enumerate(result["results"])]
            logger.info(f"Successfully reranked {len(documents)} documents")
            return reranked_indices
        else:
            logger.warning("Unexpected reranking response format, using original order")
            return list(range(len(documents)))

    except Exception as e:
        logger.error(f"Reranking failed: {str(e)}, using original order")
        return list(range(len(documents)))


def search_engine(
        username,
        library,
        text,
        model_name,
        n_results=5,
        mistral_key=None,
        openai_key=None,
        rerank=False,
        rerank_model="BAAI/bge-reranker-v2-m3"
):
    start = time.time()
    language = 'fr'

    # Connect to database
    conn, cursor = initialize_all_connection()
    try:
        with get_db_connection() as (conn, cursor):
            # Get model ID
            cursor.execute("SELECT id FROM embeddings_models WHERE model_name=%s AND language=%s",
                           (model_name, language))
            model_id = cursor.fetchone()[0]

            # Get embedding
            text_embedding = get_embeddings(
                text_list=[text],
                model_name=model_name,
                mistral_key=mistral_key,
                openai_key=openai_key)

            print('model_name', model_name)
            print('language', language)
            print('library', library)

            print('time to get embeddings:', time.time() - start)

            index, embedding_ids = retrieve_faiss_index(model_name, language, library, username, cursor=cursor)

            print('time to retrieve faiss index:', time.time() - start)
            # Search in faiss index
            D, I = index.search(text_embedding, 4 * n_results)

            try:
                if not embedding_ids:
                    logger.warning("No embedding IDs provided for query")
                    return []

                cursor.execute(
                    "SELECT embedding_ids FROM faiss_index_metadata WHERE model_id=%s AND library=%s AND (username=%s OR username='all_users')",
                    (model_id, library, username))
                embedding_ids = json.loads(cursor.fetchone()[0])
                # print('embedding_ids:', embedding_ids)
                print("len(embedding_ids)", len(embedding_ids))
                embedding_ids = [embedding_ids[i] for i in I[0]]

                query = "SELECT id, small_chunk_id FROM embeddings WHERE id IN ({})".format(
                    ','.join(['%s'] * len(embedding_ids)))

                cursor.execute(query, embedding_ids)

            except (InterfaceError, OperationalError) as e:
                logger.error(f"Database error occurred: {str(e)}")
                # Handle the error appropriately, maybe retry the connection or raise a custom exception
            except Exception as e:
                logger.exception(f"Unexpected error in search_engine_online: {str(e)}")
                raise

            fetched_data = cursor.fetchall()
            print('time to fetch data:', time.time() - start)
            id_to_small_chunk_id = {id: small_chunk_id for id, small_chunk_id in fetched_data}
            small_chunk_ids = [id_to_small_chunk_id[id] for id in embedding_ids if id in id_to_small_chunk_id]

            print('time to get small chunk ids:', time.time() - start)

            try:
                if not small_chunk_ids:
                    logger.warning("No small chunk IDs found for query")
                    return []
                query = "SELECT * FROM small_chunks WHERE id IN ({})".format(','.join(['%s'] * len(small_chunk_ids)))
                cursor.execute(query, small_chunk_ids)

            except (InterfaceError, OperationalError) as e:
                logger.error(f"Database error occurred: {str(e)}")

            except Exception as e:
                logger.exception(f"Unexpected error in search_engine_online: {str(e)}")
                raise

            # c.execute("SELECT * FROM small_chunks WHERE id IN ({})".format(','.join(['%s']*len(small_chunk_ids))), small_chunk_ids)
            small_chunks = cursor.fetchall()

            print('time to fetch small chunks:', time.time() - start)
            # Create a dictionary to map small_chunk_ids to small_chunks and reorder
            id_to_small_chunk = {sc[0]: sc for sc in small_chunks}
            small_chunks = [id_to_small_chunk[id] for id in small_chunk_ids if id in id_to_small_chunk]

            # Filter out duplicate big chunks and prepare for retrieval
            seen_big_chunk_ids = set()
            filtered_small_chunks = []
            for sc in small_chunks:
                if sc[1] not in seen_big_chunk_ids:
                    filtered_small_chunks.append(sc)
                    seen_big_chunk_ids.add(sc[1])
                    if len(filtered_small_chunks) >= 50:
                        break

            print('time to filter small chunks:', time.time() - start)

            big_chunk_ids = list(seen_big_chunk_ids)
            cursor.execute(
                "SELECT id, source_doc_id, page_number, three_page_content, page_content  FROM big_chunks WHERE id IN ({})".format(
                    ','.join(['%s'] * len(big_chunk_ids))), big_chunk_ids)
            big_chunks = cursor.fetchall()
            big_chunk_map = {bc[0]: bc for bc in big_chunks}

            print('time to fetch big chunks:', time.time() - start)

            source_doc_ids = [bc[1] for bc in big_chunks if bc[0] in big_chunk_map]
            # print('source_doc_ids:', source_doc_ids)
            cursor.execute("SELECT id, url, title FROM source_docs WHERE id IN ({})".format(
                ','.join(['%s'] * len(source_doc_ids))), source_doc_ids)
            source_docs = cursor.fetchall()
            source_doc_map = {source_doc[0]: source_doc for source_doc in source_docs}

            print('time to fetch source_docs:', time.time() - start)

        # Construct results BEFORE reranking
        results = []
        for i, small_chunk in enumerate(filtered_small_chunks):
            big_chunk = big_chunk_map.get(small_chunk[1])
            if big_chunk is None:
                continue  # Skip if no corresponding big_chunk found
            pdf = source_doc_map.get(big_chunk[1])
            if pdf is None:
                continue  # Skip if no corresponding PDF found

            results.append({
                'url': pdf[1],
                'title': pdf[2],
                'page_number': big_chunk[2],
                'small_chunk_content': small_chunk[3],
                'three_page_content': big_chunk[3],
                'page_content': big_chunk[4],
                'source_doc_id': pdf[0],
                'document_index': i
            })

        print('time to construct results:', time.time() - start)

        # RERANKING STEP: Apply reranking if requested and we have results
        if rerank and results and len(results) > 1:
            print(f"Applying reranking with model {rerank_model}...")
            rerank_start = time.time()

            # Extract text content for reranking (using three_page_content as it's more comprehensive)
            documents_for_reranking = [result['three_page_content'] for result in results]

            # Get reranked indices
            reranked_indices = rerank_documents(
                query=text,
                documents=documents_for_reranking,
                model=rerank_model
            )

            # Reorder results based on reranking
            if reranked_indices and len(reranked_indices) == len(results):
                reranked_results = [results[i] for i in reranked_indices if i < len(results)]
                results = reranked_results
                print(f'time for reranking: {time.time() - rerank_start:.3f}s')
            else:
                logger.warning("Reranking indices don't match results length, keeping original order")

        # Truncate to final n_results AFTER reranking
        results = results[:n_results]

        print('total time:', time.time() - start)
        return results

    except Exception as e:
        logger.error(f"Database error occurred: {str(e)}")
        return []


def search_engine_for_llm(username, library, question, n_results=1,
                          model_name='camembert', mistral_key=None,
                          openai_key=None, rerank=False, rerank_model="BAAI/bge-reranker-v2-m3"):
    '''Moteur de recherche en français pour les documents législatifs de l'EPFL
            args: question (str): Une question comme vous le demanderiez à un collègue
            n_results (int): nombre de résultats à renvoyer
            rerank (bool): whether to apply reranking
            rerank_model (str): reranking model to use
            returns: result (str): les noms des big_chunks contenant du contenu pertinent.
    '''
    # print('search_engine_general', username, library, question, n_results, model_name, mistral_key, openai_key)
    question = question.encode().decode('unicode-escape')
    results = search_engine(
        username=username,
        library=library,
        text=question,
        model_name=model_name,
        n_results=n_results,
        mistral_key=mistral_key,
        openai_key=openai_key,
        rerank=rerank,
        rerank_model=rerank_model
    )

    context = ''
    n_big_chunks_selected = len(results)
    for i, result_element in enumerate(results):
        three_page_content = result_element['three_page_content']
        context += f"\n\n--- Document {i} ---\n{three_page_content}"

    prompt_question = f'''
        A partir des {n_big_chunks_selected} documents sélectionnés, réponds à la question suivante: {question}
        Tu dois citer tes sources après chaque affirmation en écrivant [0] pour désigner le document 0, [1] pour désigner le document 1,
        [2] pour le document 2...
        C'est très important de respecter ce format car une regex viendra chercher les sources et les afficher joliment.
        Tu dois seulement indiquer le numéro du document entre crochet sans ajouter le titre du document\n\n
        N'ajoute pas le détail des sources à la fin de ta réponse. Elles seront automatiquement ajoutées. 

                '''

    full_prompt = prompt_question + context + prompt_question
    total_n_tokens_input = len(tiktoken_encoding.encode(full_prompt))

    sources_formated = []
    for i, result in enumerate(results):
        title = result['title']
        page_number = result['page_number']
        sources_formated.append({'title': title,
                                 'page_number': page_number,
                                 'document_index': i,
                                 'source_doc_id': result['source_doc_id'],
                                 'url': result['url']
                                 })

    return json.dumps({"data": full_prompt, "sources": sources_formated, "n_tokens_input": total_n_tokens_input},
                      ensure_ascii=False)


def create_search_engine_tool(username, library, model_name, n_results=1, mistral_key=None, openai_key=None,
                              rerank=False, rerank_model="BAAI/bge-reranker-v2-m3"):
    @tool
    def search_engine_tool(question, n_results=n_results,
                           # mistral_key=mistral_key,
                           # openai_key=openai_key,
                           # rerank=rerank
                           ):
        '''Moteur de recherche en français pour tout type de documents
                        args: question (str): Une question comme vous le demanderiez à un collègue
                        n_results (int): nombre de résultats à renvoyer
                        returns: result (str): les elements les plus pertinents pour répondre à la question ainsi que les sources.
                '''
        return search_engine_for_llm(
            username=username,
            library=library,
            question=question,
            n_results=n_results,
            model_name=model_name,
            mistral_key=mistral_key,
            openai_key=openai_key,
            rerank=rerank,
            rerank_model=rerank_model
        )

    return search_engine_tool


if __name__ == '__main__':
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Test the reranking functionality
    test_results = search_engine(
        username="servicenow_user",
        library="LEX AND RH",
        text="parking prices",
        model_name="rcp",
        n_results=3,
        rerank=True
    )
    print("Test results with reranking:", len(test_results))
    print('test_results:', test_results)