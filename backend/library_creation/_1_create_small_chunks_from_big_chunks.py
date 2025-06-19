import os

from pydantic import BaseModel
import sys
current_folder = os.path.dirname(os.path.abspath(__file__))
root_folder = os.path.abspath(os.path.abspath(os.path.join(current_folder, '..')))
utils_folder = os.path.join(root_folder, 'myUtils')
data_folder = os.path.join(root_folder, 'data')
sys.path.append(root_folder)

# print('root_folder:', root_folder)

#from transformers import CamembertModel, CamembertTokenizer, pipeline

from myUtils.connect_acad2 import reconnect_on_failure


# import nltk
# nltk.download('punkt')
# from nltk.tokenize import sent_tokenize
# from langdetect import detect

import tiktoken
openai_tokenizer = tiktoken.get_encoding("cl100k_base")


class SmallChunk(BaseModel):
    big_chunk_id: int
    chunk_number: int
    chunk_content: str
    language_detected: str
    en_chunk_content: str
    n_token: int = None


dico_language = {
    'en': 'english',
    'fr': 'french',
    'de': 'german',
}

# translators = {
#     'de': pipeline("translation_de_to_en", model="Helsinki-NLP/opus-mt-de-en"),
#     'fr': pipeline("translation_fr_to_en", model="Helsinki-NLP/opus-mt-fr-en")
# }

# tokenizer_camembert = CamembertTokenizer.from_pretrained("camembert/camembert-large")



# def cut_text_into_small_chunks_bu(text, tokenizer, big_chunk_id, chunk_max_length):
#     # print('text to cut:', text)
#     try:
#         language = detect(str(text))
#     except:
#         language = 'unknown'
#     if language not in ['en', 'fr', 'de']:
#         sentences = sent_tokenize(str(text))
#     else:
#         sentences = sent_tokenize(str(text), language=dico_language.get(language))
#
#
#     #preprocess sentences to cut them if too long
#     sentences_shortened = []
#     for i, sentence in enumerate(sentences):
#         sentence_token_length = len(tokenizer.encode(sentence))
#         if sentence_token_length > chunk_max_length:
#             sentence_chunks = [sentence[i:i+chunk_max_length] for i in range(0, len(sentence), chunk_max_length)]
#             sentences_shortened += sentence_chunks
#         else:
#             sentences_shortened.append(sentence)
#
#
#     # create chunks by taking sentences up to the max length
#     small_chunks = []
#     current_chunk = ''
#     current_token_length = 0
#     for i, sentence in enumerate(sentences_shortened):
#         sentence_token_length = len(tokenizer.encode(sentence))
#         current_chunk_plus_sentence_token_length = len(tokenizer.encode(current_chunk + sentence))
#         # print('current_token_length:', current_token_length)
#         # print('sentence_token_length:', sentence_token_length)
#         if current_chunk_plus_sentence_token_length <= chunk_max_length:
#             current_chunk += sentence
#             current_token_length += sentence_token_length
#         else:
#             small_chunk = SmallChunk(
#                 big_chunk_id=big_chunk_id,
#                 chunk_number=i,
#                 chunk_content=current_chunk,
#                 language_detected=language,
#                 en_chunk_content=current_chunk,
#                 n_token=current_token_length
#             )
#             small_chunks.append(small_chunk)
#             current_chunk = sentence
#             current_token_length = sentence_token_length
#
#
#     small_chunk = SmallChunk(
#         big_chunk_id=big_chunk_id,
#         chunk_number=len(small_chunks),
#         chunk_content=current_chunk,
#         language_detected=language,
#         en_chunk_content=current_chunk,
#         n_token=current_token_length
#     )
#     small_chunks.append(small_chunk)
#     return small_chunks


def cut_text_into_small_chunks(text, tokenizer, big_chunk_id, chunk_max_length):
    tokens = tokenizer.encode(str(text))

    # Split tokens into chunks of max_length
    token_chunks = [tokens[i:i + chunk_max_length] for i in range(0, len(tokens), chunk_max_length)]

    small_chunks = []
    for i, token_chunk in enumerate(token_chunks):
        chunk_text = tokenizer.decode(token_chunk)
        small_chunk = SmallChunk(
            big_chunk_id=big_chunk_id,
            chunk_number=i,
            chunk_content=chunk_text,
            language_detected='unknown',  # Simplified language handling
            en_chunk_content=chunk_text,
            n_token=len(token_chunk)
        )
        small_chunks.append(small_chunk)

    return small_chunks

@reconnect_on_failure
def insert_small_chunks_into_db(library, username, cursor, connection=None):

    cursor.execute("SELECT id, source_doc_id, page_content FROM big_chunks where library=%s AND username=%s", (library, username))
    big_chunks = cursor.fetchall()
    for i, big_chunk in enumerate(big_chunks):
        print(f'processing big chunk {i}/{len(big_chunks)}')
        # print(big_chunk)
        big_chunk_id, source_doc_id, page_content = big_chunk

        # check if small chunks for this big chunk already exist
        cursor.execute("SELECT id FROM small_chunks WHERE big_chunk_id=%s AND library=%s AND username=%s",
                  (big_chunk_id, library, username))
        if cursor.fetchone():
            print(f'small chunks for big chunk {big_chunk_id} already exist')
            continue

        small_chunks = cut_text_into_small_chunks(page_content, openai_tokenizer, big_chunk_id, 500)
        for small_chunk in small_chunks:
            cursor.execute(
                """INSERT IGNORE INTO small_chunks 
                (big_chunk_id, chunk_number, chunk_content, language_detected, en_chunk_content, library, username, n_token) 
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)""",
                (small_chunk.big_chunk_id, small_chunk.chunk_number, small_chunk.chunk_content,
                 small_chunk.language_detected, small_chunk.en_chunk_content, library, username, small_chunk.n_token)
            )
            if connection:
                connection.commit()






if __name__ == '__main__':
    pass
