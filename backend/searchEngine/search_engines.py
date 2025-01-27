import json
import os
# import sqlite3
# import time
#
# import faiss

from myUtils.get_embeddings import get_embeddings
# from sentence_transformers import CrossEncoder
from myUtils.connect_acad import reconnect_on_failure, initialize_all_connection
from library_creation._3_create_faiss_index import retrieve_faiss_index

import logging
from pymysql.err import InterfaceError, OperationalError

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# import torch
# from transformers import AutoModelForSequenceClassification, AutoTokenizer
import tiktoken
tiktoken_encoding = tiktoken.get_encoding("cl100k_base")
from langchain.agents import tool, create_tool_calling_agent

rerankers = {
    # 'gte_bu': {
    #     'tokenizer': AutoTokenizer.from_pretrained('Alibaba-NLP/gte-multilingual-reranker-base'),
    #     'model': AutoModelForSequenceClassification.from_pretrained('Alibaba-NLP/gte-multilingual-reranker-base',
    #                                                                 trust_remote_code=True,
    #                                                                 torch_dtype=torch.float32)
    #     }
    'gte': {}
}

# def reranker(model_name_or_path, query, paragraphs):
#     if model_name_or_path in rerankers:
#         tokenizer = rerankers[model_name_or_path]['tokenizer']
#         model = rerankers[model_name_or_path]['model']
#
#     else:
#         raise ValueError('Model name not supported')
#
#     pairs = [[query, p] for p in paragraphs]
#     with torch.no_grad():
#         inputs = tokenizer(pairs, padding=True, truncation=True, return_tensors='pt')
#         scores = model(**inputs, return_dict=True).logits.view(-1, ).float()
#
#     ordered_paragraphs = [p for _, p in sorted(zip(scores, paragraphs), reverse=True)]
#     print('query:', query)
#     print('ordered_paragraphs:')
#     for i, p in enumerate(ordered_paragraphs):
#         print('---'*100)
#         print('paragraph', i)
#         print(p)
#         print('---'*100)
#
#     return scores


def search_engine(
        username,
        library,
        text,
        model_name,
        n_results=5,
        mistral_key=None,
        openai_key=None,
        rerank=False,
):

    language = 'fr'

    # Connect to database
    conn, cursor = initialize_all_connection()

    # Get model ID
    cursor.execute("SELECT id FROM embeddings_models WHERE model_name=%s AND language=%s", (model_name, language))
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

    index, embedding_ids = retrieve_faiss_index(model_name, language, library, username, cursor=cursor)

    # Search in faiss index
    D, I = index.search(text_embedding, 30)

    try:
        if not embedding_ids:
            logger.warning("No embedding IDs provided for query")
            return []



        cursor.execute("SELECT embedding_ids FROM faiss_index_metadata WHERE model_id=%s AND library=%s AND (username=%s OR username='all_users')",
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
    id_to_small_chunk_id = {id: small_chunk_id for id, small_chunk_id in fetched_data}
    small_chunk_ids = [id_to_small_chunk_id[id] for id in embedding_ids if id in id_to_small_chunk_id]

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

    big_chunk_ids = list(seen_big_chunk_ids)
    cursor.execute("SELECT id, pdf_id, page_number, three_page_content, page_content  FROM big_chunks WHERE id IN ({})".format(','.join(['%s']*len(big_chunk_ids))), big_chunk_ids)
    big_chunks = cursor.fetchall()
    big_chunk_map = {bc[0]: bc for bc in big_chunks}

    pdf_ids = [bc[1] for bc in big_chunks if bc[0] in big_chunk_map]
    # print('pdf_ids:', pdf_ids)
    cursor.execute("SELECT id, url, title FROM pdfs WHERE id IN ({})".format(','.join(['%s']*len(pdf_ids))), pdf_ids)
    pdfs = cursor.fetchall()
    pdf_map = {pdf[0]: pdf for pdf in pdfs}

    conn.close()

    # Construct results
    results = []
    for i, small_chunk in enumerate(filtered_small_chunks):
        big_chunk = big_chunk_map.get(small_chunk[1])
        if big_chunk is None:
            continue  # Skip if no corresponding big_chunk found
        pdf = pdf_map.get(big_chunk[1])
        if pdf is None:
            continue  # Skip if no corresponding PDF found

        results.append({
            'url': pdf[1],
            'title': pdf[2],
            'page_number': big_chunk[2],
            'small_chunk_content': small_chunk[3],
            'three_page_content': big_chunk[3],
            'page_content': big_chunk[4],
            'pdf_id': pdf[0],
            'document_index': i
        })

    # # Rerank results
    # if rerank:
    #     print('reranking')
    #     scores = reranker('gte', text, [r['page_content'] for r in results])
    #     scores = scores.tolist()
    #     print('scores:', scores)
    #     print('reranked')
    #     results = [r for _, r in sorted(zip(scores, results), key=lambda x: x[0], reverse=True)]
    results = results[:n_results]




    return results

def search_engine_for_llm(username, library, question, n_results=1,
                          model_name='camembert', mistral_key=None,
                          openai_key=None, rerank=False):
    '''Moteur de recherche en français pour les documents législatifs de l'EPFL
            args: question (str): Une question comme vous le demanderiez à un collègue
            n_results (int): nombre de résultats à renvoyer
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
        rerank=rerank
    )

    context = ''
    n_big_chunks_selected=0
    for i, result_element in enumerate(results):
        three_page_content = result_element['three_page_content']
        context += three_page_content


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
                                 'pdf_id': result['pdf_id'],
                                 'url': result['url']
                                 })



    return json.dumps({"data":full_prompt, "sources":sources_formated, "n_tokens_input":total_n_tokens_input}, ensure_ascii=False)


def create_search_engine_tool(username, library, model_name, n_results=1, mistral_key=None, openai_key=None, rerank=False):
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
            rerank=rerank
        )
    return search_engine_tool


if __name__ == '__main__':
    current_dir = os.path.dirname(os.path.abspath(__file__))



