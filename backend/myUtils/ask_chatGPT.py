from multiprocessing.context import AuthenticationError
from time import sleep

from openai import OpenAI

import dotenv
import os

dotenv.load_dotenv(dotenv.find_dotenv())
OPENAI_KEY = os.getenv('OPENAI_KEY')



def ask_chatGPT(prompt, system_instruction=None, temperature=0.1, max_tokens=2300, top_p=1, frequency_penalty=0,
                model="gpt-4o-mini", verbose=False, openai_key=None, max_retries=3):
    '''
      Ask GPT on a prompt
    '''
    client = OpenAI(
        api_key=openai_key
    )
    print('asking GPT on prompt: ', prompt, '...')

    if verbose:
        print('asking GPT on prompt: ', prompt, '...')

    for i in range(max_retries):
        sleep(i)
        try:
            if system_instruction is not None:
                chat_completion = client.chat.completions.create(
                    messages=[
                        {
                            "role": "system",
                            "content": system_instruction,
                        },
                        {
                            "role": "user",
                            "content": prompt,
                        }
                    ],
                    model=model,
                    temperature=temperature,
                    # max_tokens=max_tokens,
                    top_p=top_p,
                    frequency_penalty=frequency_penalty,
                )
            else:
                chat_completion = client.chat.completions.create(
                    messages=[
                        {
                            "role": "user",
                            "content": prompt,
                        }
                    ],
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p,
                    frequency_penalty=frequency_penalty,
                )
            break
        except AuthenticationError as e:
            print('authenticationError handled:', e)
            raise e
        except Exception as e:
            print('error handled:', e)
            raise e
            print(f'GPT failed to respond, retrying... {i}th time, error: {e}')
            if i == max_retries:

                print('GPT failed to respond, returning empty string')
                raise e

    return chat_completion


if __name__ == '__main__':

    prompt = '''"System: You are the EPFL assistant. \n            You receive a user query and you have access to one tool: \n            a legal_document_search_engine\n            Your goal is to use the tool to answer the user query.\n            You need to find the most relevant result to the query and then reformulate the result to the user.\n            You should absolutely always quote the reference you are using: LEX_number if exists and name of the document and article number if exist. \n            You also quote the entire paragraph you used to answer the question.\n            You use the following formulation: \"comme indiqué dans l'article <numéro d'article> du <titre du document>:\".\n            When You use the legal_document_search_engine you should enter a query in french that ressembles the output you want to obtain. The query must be a fake legal document answering the question\n            \nHuman: {\"user_input\":\"a combien de vacances ai je le droit ?\"}"
'''
    answer = ask_chatGPT(prompt)
    print(answer)
    print(answer.choices[0].message.content)
    # print(answer['choices'][0]['message']['content'])