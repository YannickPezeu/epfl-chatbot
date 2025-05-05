import json

import numpy as np
# from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import os

from openai import AzureOpenAI

# from mistralai.client import MistralClient
# from transformers import AutoModel, AutoTokenizer
# import torch.nn.functional as F

load_dotenv()
rcp_api_key = os.getenv('RCP_API_KEY')
azure_api_base = os.getenv('AZURE_OPENAI_ENDPOINT_EMBEDDING')

# MISTRAL_KEY = os.getenv("MISTRAL_KEY")
# OPENAI_KEY = os.getenv("OPENAI_KEY")

from openai import OpenAI
import requests

print('initializing models...')
models = {
    # 'camembert': SentenceTransformer('dangvantuan/sentence-camembert-large'),
    # 'mpnet': SentenceTransformer('all-mpnet-base-v2'),
    # 'gte': AutoModel.from_pretrained('Alibaba-NLP/gte-multilingual-base', trust_remote_code=True),
    # 'embaas': SentenceTransformer('embaas/sentence-transformers-multilingual-e5-large'),
    # 'fr_long_context': SentenceTransformer('dangvantuan/french-embedding-LongContext', trust_remote_code=True)
}
print('models initialized')

def get_embeddings(text_list, model_name, mistral_key=None, openai_key=None):
    '''
    Get embeddings for a list of texts using a specific model
    :param text_list:
    :param model_name:
    :return: embeddings_nd_array of shape (len(text_list), embedding_size)
    '''
    # print('mistral_key:', mistral_key)
    # print('openai_key:', openai_key)
    # if model_name == 'camembert':
    #     model = models['camembert']
    #     embeddings_nd_array = model.encode(text_list)
    #
    # elif model_name == 'mpnet':
    #     model = models['mpnet']
    #     embeddings_nd_array = model.encode(text_list)
    #
    # elif model_name == 'gte':
    #     tokenizer = AutoTokenizer.from_pretrained('Alibaba-NLP/gte-multilingual-base')
    #     model = models['gte']
    #     batch_dict = tokenizer(text_list, max_length=8192, padding=True, truncation=True, return_tensors='pt')
    #     outputs = model(**batch_dict)
    #     dimension = 768
    #     embeddings = outputs.last_hidden_state[:, 0][:dimension]
    #     embeddings = F.normalize(embeddings, p=2, dim=1)
    #     embeddings_nd_array = embeddings.detach().numpy()
    #
    # elif model_name == 'embaas':
    #     model = models['embaas']
    #     embeddings_nd_array = model.encode(text_list)
    #
    # elif model_name == 'fr_long_context':
    #     model = models['fr_long_context']
    #     embeddings_nd_array = model.encode(text_list)

    # if model_name == 'mistral':
    #     client = MistralClient(api_key=mistral_key)
    #     embeddings_batch_response = client.embeddings(
    #         model="mistral-embed",
    #         input=text_list,
    #     )
    #     embeddings_list = [el.embedding for el in embeddings_batch_response.data]
    #     embeddings_nd_array = np.array(embeddings_list)

    if model_name == 'openai':
        try:
            client =OpenAI(api_key=openai_key)
            embeddings_batch_response = client.embeddings.create(input=text_list, model='text-embedding-3-large')
            embeddings_list = [el.embedding for el in embeddings_batch_response.data]
            embeddings_nd_array = np.array(embeddings_list)
        except Exception as e:
            raise(e)

    elif model_name == 'rcp':
        base_url = "https://inference-dev.rcp.epfl.ch/v1"
        endpoint = f"{base_url}/embeddings"

        # Prepare headers
        headers = {
            "Authorization": f"Bearer {rcp_api_key}",
            "Content-Type": "application/json"
        }

        # Ensure text is a list if it's a single string
        if isinstance(text_list, str):
            text_list = [text_list]

        # Prepare request payload
        payload = {
            "model": "Linq-AI-Research/Linq-Embed-Mistral",
            "input": text_list
        }

        # Make the request
        try:
            response = requests.post(
                endpoint,
                headers=headers,
                data=json.dumps(payload),
                timeout=180
            )

            # Check if the request was successful
            response.raise_for_status()

            # Return the response data
            result = response.json()

            embedding_list = [result["data"][i]["embedding"] for i in range(len(result["data"]))]
            embeddings_nd_array = np.array(embedding_list)

            return embeddings_nd_array

        except requests.exceptions.RequestException as e:
            print(f"Error: {e}")
            if hasattr(e, 'response') and e.response is not None:
                print(f"Response content: {e.response.text}")
            return None


    # if model_name == 'openai':
    #     try:
    #         client = AzureOpenAI(
    #             api_key='',
    #             api_version="2024-02-15-preview",
    #             azure_endpoint=azure_api_base,
    #             azure_deployment="text-embedding-3-large"  # Use azure_deployment instead of deployment_name
    #         )
    #
    #         embeddings_batch_response = client.embeddings.create(
    #             input=text_list,
    #             model='text-embedding-3-large'
    #         )
    #
    #         embeddings_list = [el.embedding for el in embeddings_batch_response.data]
    #         embeddings_nd_array = np.array(embeddings_list)
    #     except Exception as e:
    #         raise (e)

    else:
        raise ValueError("Model must be either 'camembert', 'mpnet', 'mistral' or 'openai', not {}".format(model_name))

    return embeddings_nd_array


def check_embeddings():
    text_list = ['Hello, how are you?', 'I am fine, thank you.']
    model_name = 'camembert'
    embeddings = get_embeddings(text_list, model_name)
    print(embeddings)
    print(len(embeddings))
    print(embeddings.shape)
    print(type(embeddings))
    print('-------------------')

    model_name = 'mpnet'
    embeddings = get_embeddings(text_list, model_name)
    print(embeddings)
    print(len(embeddings))
    print(embeddings.shape)
    print(type(embeddings))
    print('-------------------')

    model_name = 'mistral'
    embeddings = get_embeddings(text_list, model_name)
    print(embeddings)
    print(len(embeddings))
    print(embeddings.shape)
    print(type(embeddings))
    print('-------------------')

    model_name = 'openai'
    embeddings = get_embeddings(text_list, model_name)
    print(embeddings)
    print(len(embeddings))
    print(embeddings.shape)
    print(type(embeddings))
    print('-------------------')

def test_similarity(text1, text2, model_name):
    embeddings = get_embeddings([text1, text2], model_name)
    distance = euclidean_distances([embeddings[0]], [embeddings[1]])[0][0]
    return distance

if __name__ == '__main__':

    import dotenv
    from sklearn.metrics.pairwise import euclidean_distances

    openai_key = os.getenv('OPENAI_KEY2')

    e = get_embeddings(['Hello, how are you?'], 'openai', openai_key=openai_key)

    print(e.shape)

    exit()

    text_list_test = ['Hello, how are you?', 'I am fine, thank you.']
    embeddings_openai = get_embeddings(text_list_test, 'openai', openai_key=openai_key)
    print(type(embeddings_openai))
    print(embeddings_openai.shape)
    print(embeddings_openai[0].shape)

    exit()

    test_sentence = 'Hello world'
    for model_name in ['camembert',
                       # 'mpnet', 'mistral', 'openai',
                       'gte']:
        embeddings = get_embeddings([test_sentence], model_name)
        print(embeddings.shape)
        print(len(embeddings[0]))

    # exit()
    sentences_similar = [
        ['A tree is in the valley.', 'A tree is in the bottom of the valley.'],
        ['The man is looking at me happily', 'The man is looking at me with a smile.'],
        ['The cat is on the mat.', 'The cat is on the carpet.'],
        ['The dog is on the mat.', 'The cat is on the mat.'],
        ['The dog is on the mat.', 'The dog is on the carpet.'],
        ['The dog is on the mat.', 'The dog is on the carpet.'],
        ['Machine learning is a fascinating topic.', 'Machine learning is an exciting topic.'],
    ]

    sentences_dissimilar = [
        ['A tree is in the valley.', 'The doctor is in the hospital.'],
        ['The man is looking at me happily', 'The cat is on the carpet.'],
        ['The cat is on the mat.', 'Machine learning is an exciting topic.'],
        ['The dog is on the mat.', 'The doctor is in the hospital.'],
        ['The dog is on the mat.', 'Machine learning is an exciting topic.'],
        ['The dog is on the mat.', 'The doctor is in the hospital.'],
        ['Machine learning is a fascinating topic.', 'The doctor is in the hospital.']
    ]

    sentences_translated = [
        ['A tree is in the valley.', 'Un arbre est dans la vallée.'],
        ['The man is looking at me happily', 'L\'homme me regarde avec bonheur.'],
        ['The cat is on the mat.', 'Le chat est sur le tapis.'],
        ['The dog is on the mat.', 'Le chien est sur le tapis.'],
        ['Machine learning is a fascinating topic.', 'L\'apprentissage automatique est un sujet fascinant.'],
    ]

    sentences_french_similar = [
        ['Un arbre est dans la vallée.', 'Un arbre est au fond de la vallée.'],
        ['L\'homme me regarde avec bonheur.', 'L\'homme me regarde avec un sourire.'],
        ['Le chat est sur le tapis.', 'Le chat est sous le tapis.'],
        ['Le chien est sur le tapis.', 'Le chat est sur le tapis.'],
        ['Le chien est sur le tapis.', 'Le chien est sous le tapis.'],
        ['L\'apprentissage automatique est un sujet fascinant.', 'L\'apprentissage automatique est un sujet passionnant.'],
    ]

    sentences_french_dissimilar = [
        ['Un arbre est dans la vallée.', 'Le docteur est à l\'hôpital.'],
        ['L\'homme me regarde avec bonheur', 'Le chat est sur le tapis.'],
        ['Le chat est sur le tapis.', 'L\'apprentissage automatique est un sujet passionnant.'],
        ['Le chien est sur le tapis.', 'Le docteur est à l\'hôpital.'],
        ['Le chien est sur le tapis.', 'L\'apprentissage automatique est un sujet passionnant.'],
        ['Le chien est sur le tapis.', 'Le docteur est à l\'hôpital.'],
        ['L\'apprentissage automatique est un sujet fascinant.', 'Le docteur est à l\'hôpital.']
    ]

    distances = {
        'similar': {
            'camembert': [],
            'mpnet': [],
            'mistral': [],
            'openai': [],
            'gte': []
        },
        'dissimilar': {
            'camembert': [],
            'mpnet': [],
            'mistral': [],
            'openai': [],
            'gte': []
        },
        'translated': {
            'camembert': [],
            'mpnet': [],
            'mistral': [],
            'openai': [],
            'gte': []
        },
        'french_similar': {
            'camembert': [],
            'mpnet': [],
            'mistral': [],
            'openai': [],
            'gte': []
        },
        'french_dissimilar': {
            'camembert': [],
            'mpnet': [],
            'mistral': [],
            'openai': [],
            'gte': []
        }
    }

    def evaluate_distances(sentences, type_of_sentences):
        for sentence_pair in sentences:
            text1, text2 = sentence_pair
            print('text1:', text1)
            print('text2:', text2)
            distance_camembert = test_similarity(text1, text2, 'camembert')
            distance_gte = test_similarity(text1, text2, 'gte')
            # distance_mpnet = test_similarity(text1, text2, 'mpnet')
            # distance_mistral = test_similarity(text1, text2, 'mistral')
            # distance_openai = test_similarity(text1, text2, 'openai')

            distances[type_of_sentences]['camembert'].append(distance_camembert.item())  # Convert to Python scalar
            distances[type_of_sentences]['gte'].append(distance_gte.item())
            # distances[type_of_sentences]['mpnet'].append(distance_mpnet.item())
            # distances[type_of_sentences]['mistral'].append(distance_mistral.item())
            # distances[type_of_sentences]['openai'].append(distance_openai.item())

            print('camembert:', distance_camembert)
            print('gte:', distance_gte)
            # print('mpnet:', distance_mpnet)
            # print('mistral:', distance_mistral)
            # print('openai:', distance_openai)
            print('-------------------')

        print('-------------------' * 10)
        print('Average distances for {} sentences:'.format(type_of_sentences))
        for model_name in distances[type_of_sentences]:
            print(model_name, np.mean(distances[type_of_sentences][model_name]))
            distances[type_of_sentences][model_name] = np.mean(distances[type_of_sentences][model_name]).item()

        #save as json with indent
        with open('distances.json', 'w', encoding='utf-8') as f:
            json.dump(distances, f, indent=4, ensure_ascii=False)


    evaluate_distances(sentences_similar, 'similar')
    evaluate_distances(sentences_dissimilar, 'dissimilar')
    evaluate_distances(sentences_translated, 'translated')
    evaluate_distances(sentences_french_similar, 'french_similar')
    evaluate_distances(sentences_french_dissimilar, 'french_dissimilar')

