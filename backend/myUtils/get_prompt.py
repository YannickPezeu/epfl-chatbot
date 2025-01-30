
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

MEMORY_KEY = 'chat_history'

prompt_no_library = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            '''
            Tu es un assistant ravi d'aider les utilisateurs qui font des demandes. 
            ''',
        ),
        MessagesPlaceholder(variable_name=MEMORY_KEY),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",

'''Tu as accès à un moteur de recherche qui te permet d'obtenir des documents séparés par
\n\n-------------------\n\n DOCUMENT NUMEROi: nom_du_document\n\n contenu_du_document\n\n-------------------\n\n
Si tu penses que la réponse peut être retournée par le moteur de recherche, tu dois utiliser ce moteur de recherche. 
Tu dois citer tes sources en écrivant [0] pour désigner le document 0, [1] pour désigner le document 1... 
Tu dois seulement indiquer le numéro du document entre crochet sans ajouter le titre du document
Ce format de source est obligatoire pour que la regex fonctionne. Fais y bien attention.
N'ajoute pas le détail des sources à la fin de ta réponse. Elles seront automatiquement ajoutées. 
Attention, certains utilisateurs peuvent parler en anglais ou en d'autres langues. Tu dois leur répondre dans leur langue. 
            ''',
        ),
        MessagesPlaceholder(variable_name=MEMORY_KEY),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)

prompt_email = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            '''
            Tu es un assistant email, tu as accès à un moteur de recherche qui te permet d'obtenir des réponses à des questions
            tu reçois un email et tu dois retourner en réponse un email formatté poliment avec la réponse à la question.
            Tu finis ton email avec une formule de politesse.
            ''',
        ),
        MessagesPlaceholder(variable_name=MEMORY_KEY),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)

def get_prompt(interaction_type, special_prompt=None):
    if interaction_type == 'chat':
        base_prompt = '''Tu as accès à un moteur de recherche qui te permet d'obtenir des documents séparés par
\n\n-------------------\n\n DOCUMENT NUMEROi: nom_du_document\n\n contenu_du_document\n\n-------------------\n\n
Si tu penses que la réponse peut être retournée par le moteur de recherche, tu dois utiliser ce moteur de recherche. 
Tu dois citer tes sources en écrivant [0] pour désigner le document 0, [1] pour désigner le document 1... 
Tu dois seulement indiquer le numéro du document entre crochet sans ajouter le titre du document
Ce format de source est obligatoire pour que la regex fonctionne. Fais y bien attention.
N'ajoute pas le détail des sources à la fin de ta réponse. Elles seront automatiquement ajoutées. 
Attention, certains utilisateurs peuvent parler en anglais ou en d'autres langues. Tu dois leur répondre dans leur langue. 
            '''
    elif interaction_type == 'email':
        base_prompt = '''
            Tu es un assistant email, tu as accès à un moteur de recherche qui te permet d'obtenir des réponses à des questions
            tu reçois un email et tu dois retourner en réponse un email formatté poliment avec la réponse à la question.
            Tu finis ton email avec une formule de politesse.
            '''
    elif interaction_type == 'no_library':
        base_prompt = '''Tu es un assistant ravi d'aider les utilisateurs qui font des demandes. '''
    else:
        raise ValueError('interaction_type must be chat or email')

    if special_prompt is not None:
        full_prompt = base_prompt + "L'utilisateur a ajouté une demande particulière concernant le style de réponse: " + special_prompt
    else:
        full_prompt = base_prompt

    chatPromptTemplate = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                full_prompt,
            ),
            MessagesPlaceholder(variable_name=MEMORY_KEY),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )

    return chatPromptTemplate
