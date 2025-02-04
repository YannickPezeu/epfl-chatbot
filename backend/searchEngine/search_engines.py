import json
import logging
import time
from typing import List, Dict, Any, Optional
import tiktoken
from langchain.agents import tool
from myUtils.get_embeddings import get_embeddings
from myUtils.connect_acad2 import get_db_connection, check_database_health
from library_creation._3_create_faiss_index import retrieve_faiss_index

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(message)s - %(filename)s:%(lineno)d'
)
logger = logging.getLogger(__name__)

# Initialize tiktoken
tiktoken_encoding = tiktoken.get_encoding("cl100k_base")


class SearchEngine:
    def __init__(self):
        self.language = 'fr'

    def execute_search(
            self,
            username: str,
            library: str,
            text: str,
            model_name: str,
            n_results: int = 5,
            mistral_key: Optional[str] = None,
            openai_key: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Execute search with proper connection and error handling"""
        start_time = time.time()

        if not check_database_health():
            logger.error("Database health check failed")
            raise Exception("Database connection unhealthy")

        try:
            with get_db_connection() as (conn, cursor):
                # Get model ID and embedding
                cursor.execute(
                    "SELECT id FROM embeddings_models WHERE model_name=%s AND language=%s",
                    (model_name, self.language)
                )
                model_id = cursor.fetchone()[0]
                logger.debug(f"Got model ID in {time.time() - start_time:.2f}s")

                # Get embedding
                text_embedding = get_embeddings(
                    text_list=[text],
                    model_name=model_name,
                    mistral_key=mistral_key,
                    openai_key=openai_key
                )
                logger.debug(f"Got embeddings in {time.time() - start_time:.2f}s")

                # Get FAISS index and search
                index, embedding_ids = retrieve_faiss_index(
                    model_name, self.language, library, username, cursor=cursor
                )
                if not embedding_ids:
                    logger.warning("No embedding IDs found")
                    return []

                D, I = index.search(text_embedding, 30)
                logger.debug(f"Completed FAISS search in {time.time() - start_time:.2f}s")

                # Process search results
                cursor.execute(
                    """SELECT embedding_ids 
                       FROM faiss_index_metadata 
                       WHERE model_id=%s AND library=%s 
                       AND (username=%s OR username='all_users')""",
                    (model_id, library, username)
                )
                embedding_ids = json.loads(cursor.fetchone()[0])
                embedding_ids = [embedding_ids[i] for i in I[0]]

                # Get and process chunks
                results = self._process_chunks(cursor, embedding_ids, n_results)
                logger.info(f"Search completed in {time.time() - start_time:.2f}s")
                return results

        except Exception as e:
            logger.error(f"Search error: {e}", exc_info=True)
            raise

    def _process_chunks(self, cursor, embedding_ids: List[int], n_results: int) -> List[Dict[str, Any]]:
        """Process document chunks and build results"""
        try:
            # Get small chunk IDs
            query = """SELECT id, small_chunk_id FROM embeddings WHERE id IN ({})""".format(
                ','.join(['%s'] * len(embedding_ids))
            )
            cursor.execute(query, embedding_ids)
            fetched_data = cursor.fetchall()

            # Map IDs
            id_to_small_chunk_id = {id: sc_id for id, sc_id in fetched_data}
            small_chunk_ids = [id_to_small_chunk_id[id] for id in embedding_ids if id in id_to_small_chunk_id]

            if not small_chunk_ids:
                return []

            # Get and process small chunks
            small_chunks = self._get_small_chunks(cursor, small_chunk_ids)
            filtered_chunks = self._filter_chunks(small_chunks)

            # Get and process big chunks
            big_chunk_data = self._get_big_chunks(cursor, filtered_chunks)

            # Build final results
            return self._build_results(cursor, filtered_chunks, big_chunk_data, n_results)

        except Exception as e:
            logger.error(f"Error processing chunks: {e}", exc_info=True)
            raise

    def _get_small_chunks(self, cursor, chunk_ids: List[int]) -> List[Any]:
        """Retrieve small chunks from database"""
        query = """SELECT * FROM small_chunks WHERE id IN ({})""".format(
            ','.join(['%s'] * len(chunk_ids))
        )
        cursor.execute(query, chunk_ids)
        return cursor.fetchall()

    def _filter_chunks(self, chunks: List[Any]) -> List[Any]:
        """Filter out duplicate chunks"""
        seen_big_chunk_ids = set()
        filtered = []
        for chunk in chunks:
            if chunk[1] not in seen_big_chunk_ids:
                filtered.append(chunk)
                seen_big_chunk_ids.add(chunk[1])
                if len(filtered) >= 50:
                    break
        return filtered

    def _get_big_chunks(self, cursor, filtered_chunks: List[Any]) -> Dict[int, Any]:
        """Retrieve big chunks from database"""
        big_chunk_ids = list({chunk[1] for chunk in filtered_chunks})
        cursor.execute(
            """SELECT id, pdf_id, page_number, three_page_content, page_content
               FROM big_chunks WHERE id IN ({})""".format(
                ','.join(['%s'] * len(big_chunk_ids))
            ),
            big_chunk_ids
        )
        return {chunk[0]: chunk for chunk in cursor.fetchall()}

    def _build_results(
            self,
            cursor,
            filtered_chunks: List[Any],
            big_chunks: Dict[int, Any],
            n_results: int
    ) -> List[Dict[str, Any]]:
        """Build final search results"""
        results = []
        pdf_ids = list({big_chunks[chunk[1]][1] for chunk in filtered_chunks
                        if chunk[1] in big_chunks})

        cursor.execute(
            """SELECT id, url, title FROM pdfs WHERE id IN ({})""".format(
                ','.join(['%s'] * len(pdf_ids))
            ),
            pdf_ids
        )
        pdf_map = {pdf[0]: pdf for pdf in cursor.fetchall()}

        for i, small_chunk in enumerate(filtered_chunks):
            if small_chunk[1] not in big_chunks:
                continue

            big_chunk = big_chunks[small_chunk[1]]
            if big_chunk[1] not in pdf_map:
                continue

            pdf = pdf_map[big_chunk[1]]
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

        return results[:n_results]


# Create singleton instance
search_engine = SearchEngine()


def search_engine_for_llm(
        username: str,
        library: str,
        question: str,
        n_results: int = 1,
        model_name: str = 'camembert',
        mistral_key: Optional[str] = None,
        openai_key: Optional[str] = None,
        rerank: bool = False
) -> str:
    """Process search results for LLM consumption"""
    try:
        # Decode question to handle unicode
        question = question.encode().decode('unicode-escape')

        # Get search results
        results = search_engine.execute_search(
            username=username,
            library=library,
            text=question,
            model_name=model_name,
            n_results=n_results,
            mistral_key=mistral_key,
            openai_key=openai_key
        )

        # Process results and create context
        context = ''.join(r['three_page_content'] for r in results)
        n_big_chunks_selected = len(results)

        # Create prompt
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

        # Format sources for return
        sources_formatted = [
            {
                'title': r['title'],
                'page_number': r['page_number'],
                'document_index': i,
                'pdf_id': r['pdf_id'],
                'url': r['url']
            }
            for i, r in enumerate(results)
        ]

        return json.dumps({
            "data": full_prompt,
            "sources": sources_formatted,
            "n_tokens_input": total_n_tokens_input
        }, ensure_ascii=False)

    except Exception as e:
        logger.error(f"Error in search_engine_for_llm: {str(e)}", exc_info=True)
        raise


def create_search_engine_tool(
        username: str,
        library: str,
        model_name: str,
        n_results: int = 1,
        mistral_key: Optional[str] = None,
        openai_key: Optional[str] = None,
        rerank: bool = False
):
    """Create a LangChain tool for the search engine"""

    @tool
    def search_engine_tool(
            question: str,
            n_results: int = n_results
    ) -> str:
        """French search engine for documents

        Args:
            question (str): Question in natural language
            n_results (int): Number of results to return

        Returns:
            str: JSON string containing relevant results and sources
        """
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
    # Example usage
    logging.basicConfig(level=logging.INFO)
    logger.info("Search engine module loaded successfully")