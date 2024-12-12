import { ChatMessageProps, Source } from "./types";
import { v4 as uuidv4 } from "uuid";

export const connectToWebSocket = async (
  baseUrl: string,
  wsBaseUrl: string,
  selectedLibrary: string | undefined,
  selectedModel: string,
  nDocumentsSearchedNoLLM: number,
  nDocumentsSearched: number,
  sources: { [key: string]: Source[] },
  nTokensInput: { [key: string]: number },
  setMessages: (
    value:
      | ChatMessageProps[]
      | ((prevMessages: ChatMessageProps[]) => ChatMessageProps[])
  ) => void,
  setSocket: (value: WebSocket) => void,
  base64Credentials: string,
  embeddingModel: string,
  setConversationID: (value: string) => void,
  openaiKey?: string,
  mistralKey?: string,
  groqKey?: string,
  interaction_type?: string,
  rerank?: string,
  setOpenaiKeyStatus?: (status: "missing" | "invalid" | "valid") => void,
  conversationID?: string,
) => {
  const modelTranslation: { [key: string]: string } = {
    "GPT3.5": "gpt-3.5-turbo-1106",
    GPT4: "gpt-4-0125-preview",
    "gpt-4o": "gpt-4o",
    "gpt-4o-mini": "gpt-4o-mini",
    "llama3-8b": "llama3-8b-8192",
    "llama3-70b": "llama3-70b-8192",
    "mixtral-8x7b": "mixtral-8x7b-32768",
    No_Model: "No_Model",
  };
  if (selectedLibrary === null) {
    console.error("No library selected");
    return false;
  }
  const modelUsed = selectedModel;
  return new Promise<boolean>(async (resolve, reject) => {
    try {
      var response;
      const modelName = modelTranslation[selectedModel];
      var urlWithParameter = `${baseUrl}/ws/new_ws_connection?model_name=${modelName}&n_documents_searched=${nDocumentsSearched}&n_documents_searched_no_llm=${nDocumentsSearchedNoLLM}&library=${selectedLibrary}&embedding_model=${embeddingModel}&interaction_type=${interaction_type}&rerank=${rerank}`;
      if (openaiKey) {
        urlWithParameter += `&openai_key=${openaiKey}`;
      }
      if (mistralKey) {
        urlWithParameter += `&mistral_key=${mistralKey}`;
      }
      if (groqKey) {
        urlWithParameter += `&groq_key=${groqKey}`;
      }
      if (conversationID) {
        urlWithParameter += `&conversation_id=${conversationID}`;
      }
      response = await fetch(urlWithParameter, {
        method: "POST",
        credentials: "include",
      });

      console.log('response', response)

      if (!response.ok) {
        // Read the error message from the response body
        console.log('mytest')
        const errorBody = await response.json();
        // throw new Error(errorBody.detail || `HTTP error! status: ${response.status}`);
        setMessages((prevMessages) => {
          // Création d'une copie du tableau pour la manipulation
          const transformed_message = String(errorBody.detail)
          const newMessages = [...prevMessages];

          newMessages.push({
            sender: "Error",
            message: { message_content: transformed_message },
            messageId: uuidv4(),
            sources: [],
            nTokensInput: 0,
            messageType: 'error',
            modelUsed: modelUsed,
          });
          return newMessages;
        });
      }      
      const data = await response.json();
      const conversation_id = data.conversation_id;

      console.log('conversation_id', conversation_id)
      setConversationID(conversation_id);
      const wsUrl = '/api/ws';
      const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
      const wsFullUrl = `${wsProtocol}//`+window.location.host+wsUrl;
      const ws = new WebSocket(`${wsFullUrl}/${data.ws_connection_id}`);
      ws.onopen = () => {
        console.log("WebSocket is open now.");
        resolve(true);
      };

      ws.onmessage = (event) => {
        console.log('event', event)
        console.log('testdata', event.data)
        const messageData = JSON.parse(event.data);

        const messageType = messageData.type;

        if (messageType === "error") {
          setMessages((prevMessages) => {
            // Création d'une copie du tableau pour la manipulation
            console.log('messageData.message: ', messageData.message)
            const transformed_message = String(messageData.message)
              .replace(/_/g, " ")
              .replace(/,/g, "\n\n");
            const newMessages = [...prevMessages];
            if(messageData.message === 'Invalid API key'){
              if(setOpenaiKeyStatus){
                setOpenaiKeyStatus('invalid');
              }
            }

            newMessages.push({
              sender: "Error",
              message: { message_content: transformed_message },
              messageId: uuidv4(),
              sources: [],
              nTokensInput: 0,
              messageType: messageType,
              modelUsed: modelUsed,
            });
            return newMessages;
          });
        }

        if (messageType === "tool_answer") {
          const run_id: string = messageData.run_id;
          sources[run_id] = messageData.sources;
          nTokensInput[run_id] = messageData.n_tokens_input;
        } else if (messageType === "No_Model") {
          console.log("No_Model");
          setMessages((prevMessages) => {
            // Création d'une copie du tableau pour la manipulation
            const transformed_message = String(messageData.data)
              .replace(/_/g, " ")
              .replace(/,/g, "\n\n");
            const sources: Source[] = messageData.sources;
            const newMessages = [...prevMessages];

            newMessages.push({
              sender: "Bot",
              message: { message_content: transformed_message },
              messageId: uuidv4(),
              sources: sources,
              nTokensInput: messageData.n_tokens_input,
              messageType: messageType,
              modelUsed: modelUsed,
            });
            return newMessages;
          });
        } else if (messageType === "tool_input") {
          setMessages((prevMessages) => {
            // Création d'une copie du tableau pour la manipulation
            const newMessages = [...prevMessages];

            const previous_message: ChatMessageProps | undefined =
              prevMessages[prevMessages.length - 1];

            // Ajout d'un nouveau message si c'est le premier fragment
            const current_sources: Source[] =
              sources[previous_message.messageId];
            const current_nTokensInput: number =
              nTokensInput[previous_message.messageId];
            newMessages.push({
              sender: "Bot",
              message: messageData.data,
              messageId: messageData.run_id,
              sources: current_sources,
              messageType: messageType,
              nTokensInput: current_nTokensInput,
              modelUsed: modelUsed,
            });
            return newMessages;
          });
        } else if (
          messageType === "final_response" ||
          messageType === "response_without_tool"
        ) {
          setMessages((prevMessages) => {
            // Création d'une copie du tableau pour la manipulation
            const newMessages = [...prevMessages];

            const previous_message: ChatMessageProps | undefined =
              prevMessages[prevMessages.length - 1];
            if (
              messageData.run_id != previous_message?.messageId ||
              messageData.type != previous_message?.messageType
            ) {
              // Ajout d'un nouveau message si c'est le premier fragment
              const current_sources: Source[] =
                sources[messageData.run_id] === undefined
                  ? sources[previous_message.messageId]
                  : sources[messageData.run_id];

              const current_nTokensInput: number =
                nTokensInput[messageData.run_id] === undefined
                  ? nTokensInput[previous_message.messageId]
                  : nTokensInput[messageData.run_id];

              newMessages.push({
                sender: "Bot",
                message: messageData.data,
                messageId: messageData.run_id,
                sources: current_sources,
                messageType: messageType,
                nTokensInput: current_nTokensInput,
                modelUsed: modelUsed,
              });
            } else {
              // Modification du dernier message si ce n'est pas le premier fragment
              const lastMessageIndex = newMessages.length - 1;
              if (lastMessageIndex >= 0) {
                const lastMessage = newMessages[lastMessageIndex];
                newMessages[lastMessageIndex] = {
                  ...lastMessage,
                  message: {
                    message_content:
                      lastMessage?.message?.message_content +
                      messageData.data.message_content,
                  },
                  nTokensInput: (lastMessage?.nTokensInput ?? 0) + 3,
                };
              }
            }

            return newMessages;
          });
        }
      };

      ws.onerror = (error) => {
        console.error("WebSocket Error:", error);
        reject(error);
      };

      ws.onclose = () => {
        console.log("WebSocket is closed now.");
      };

      setSocket(ws);
    } catch (error) {
      console.log('error: ', error);
      console.error(
        "Failed to create a ws_connection or open a WebSocket connection:",
        error
      );
      reject(error);
      return false; // Return false to indicate failure
    }
  });
};

export const create_ws_connection = async (
  selectedLibrary: string | undefined,
  selectedModel: string,
  nDocumentsSearchedNoLLM: number,
  nDocumentsSearched: number,
  setMessages: (
    value:
      | ChatMessageProps[]
      | ((prevMessages: ChatMessageProps[]) => ChatMessageProps[])
  ) => void,
  setSocket: (value: WebSocket) => void,
  base64Credentials: string,
  setBASE_URL: (url: string) => void,
  setBASE_URL_WS: (url: string) => void,
  selectedEmbeddingModel: string,
  setConnectionStatus: (
    status: "connected" | "disconnected" | "connecting"
  ) => void,
  BASE_URL: string,
  BASE_URL_WS: string,
  setConversationID: (value: string) => void,
  openaiKey?: string,
  mistralKey?: string,
  groqKey?: string,
  interaction_type?: string,
  rerank?: string,
  setOpenaiKeyStatus?: (status: "missing" | "invalid" | "valid") => void,
  conversationID?: string,
) => {
  var sources: { [key: string]: Source[] } = {};
  var connection: boolean = false;
  var nTokensInput: { [key: string]: number } = {};
  setConnectionStatus("connecting");

  const fullURL = window.location.href;
  try {
    connection = await connectToWebSocket(
      BASE_URL,
      BASE_URL_WS,
      selectedLibrary,
      selectedModel,
      nDocumentsSearchedNoLLM,
      nDocumentsSearched,
      sources,
      nTokensInput,
      setMessages,
      setSocket,
      base64Credentials,
      selectedEmbeddingModel,
      setConversationID,
      openaiKey,
      mistralKey,
      groqKey,
      interaction_type,
      rerank,
      setOpenaiKeyStatus,
      conversationID,
      
    );
    setConnectionStatus("connected");
  } catch (error) {
    connection = false;
    console.error("Failed to connect to local WebSocket server:", error);
    setConnectionStatus("disconnected");
  }
};
