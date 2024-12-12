import { create } from 'zustand';
// Import your external types
import { ChatMessageProps } from "./types";

interface LibraryProgress {
  progress: number;
  price: number;
}

// Define the structure for the entire progress array
type LibraryCreationProgress = Array<{
  [library_name: string]: LibraryProgress;
}>;

interface StoreState {
  messages: ChatMessageProps[];
  setMessages: (value: ChatMessageProps[] | ((prevMessages: ChatMessageProps[]) => ChatMessageProps[])) => void;
  socket: WebSocket | null;
  setSocket: (socket: WebSocket | null) => void;
  isDrawerOpen: boolean;
  setIsDrawerOpen: (isOpen: boolean) => void;
  mistralKey: string;
  setMistralKey: (key: string) => void;
  groqKey: string;
  setGroqKey: (key: string) => void;
  pdfId: number;
  setPdfId: (pdfId: number) => void;
  pdfPageNumber: number;
  setPdfPageNumber: (pageNumber: number) => void;
  selectedModel: string;
  setSelectedModel: (model: string) => void;
  tokensPerInteraction: number,
  setTokensPerInteraction: (tokens: number) => void;
  BASE_URL: string;
  setBASE_URL: (url: string) => void;
  BASE_URL_WS: string;
  setBASE_URL_WS: (url: string) => void;
  selectedEmbeddingModel: string;
  setSelectedEmbeddingModel: (embedding: string) => void;
  base64Credentials: string;
  price_token: { [key: string]: number }
  max_documents: { [key: string]: number }
  nDocumentsSearched: number;
  setNDocumentsSearched: (n: number) => void;
  nDocumentsSearchedNoLLM: number;
  setNDocumentsSearchedNoLLM: (n: number) => void;
  nDocumentsSearchedDebounced: number;
  setNDocumentsSearchedDebounced: (n: number) => void;
  nDocumentsSearchedNoLLMDebounced: number;
  setNDocumentsSearchedNoLLMDebounced: (n: number) => void;
  groqKeyDebounced: string;
  setGroqKeyDebounced: (key: string) => void;
  mistralKeyDebounced: string;
  setMistralKeyDebounced: (key: string) => void;
  openaiKeyDebounced: string;
  setOpenaiKeyDebounced: (key: string) => void;
  isInitialLoadComplete: boolean;
  setIsInitialLoadComplete: (value: boolean) => void;
  connectionStatus: 'connected' | 'disconnected' | 'connecting';
  setConnectionStatus: (status: 'connected' | 'disconnected' | 'connecting') => void;
  BASE_URL_LOCAL: string;
  BASE_URL_ONLINE: string;
  BASE_URL_ONLINE_WS: string;
  BASE_URL_ONLINE_TEST: string;
  BASE_URL_ONLINE_WS_TEST: string;
  BASE_URL_WS_LOCAL:string;
  selectedLibrary: string | undefined;
  setSelectedLibrary: (library: string | undefined) => void;
  isLoggedIn: boolean;
  setIsLoggedIn: (value: boolean) => void;
  rerank: string;
  setRerank: (value: string) => void;
  isLeftPanelOpen: boolean;
  setIsLeftPanelOpen: (value: boolean) => void;
  interaction_type: string;
  setInteractionType: (value: string) => void;
  filesAttached: File[];
  setFilesAttached: (files: File[]) => void;
  libraryCreationProgress: LibraryCreationProgress;
  setLibraryCreationProgress: (progress: LibraryCreationProgress) => void;
  openaiKey: string;
  setOpenaiKey: (key: string) => void;
  openaiKeyStatus: 'missing' | 'invalid' | 'valid';
  setOpenaiKeyStatus: (status: 'missing' | 'invalid' | 'valid') => void;
  conversationID: string;
  setConversationID: (id: string) => void;
  libraries: string[];
  setLibraries: (libraries: string[]) => void;
  username: string;
  setUsername: (username: string) => void;
}


const useStore = create<StoreState>((set) => ({
    messages: [],
    setMessages: (value) => set((state) => ({
        messages: typeof value === 'function' ? value(state.messages) : value
      })),
    socket: null,
    setSocket: (socket: WebSocket | null) => set({ socket }),
    isDrawerOpen: false,
    setIsDrawerOpen: (isOpen: boolean) => set({ isDrawerOpen: isOpen }),
    mistralKey: "",
    setMistralKey: (key: string) => set({ mistralKey: key }),
    groqKey: "",
    setGroqKey: (key: string) => set({ groqKey: key }),
    pdfId: -1,
    setPdfId: (pdfId: number) => set({ pdfId: pdfId }),
    pdfPageNumber: -1,
    setPdfPageNumber: (pageNumber: number) => set({ pdfPageNumber: pageNumber }),
    selectedModel : "gpt-4o-mini",
    setSelectedModel: (model: string) => set({ selectedModel: model }),
    tokensPerInteraction: 5000,
    setTokensPerInteraction: (tokens: number) => set({ tokensPerInteraction: tokens }),
    base64Credentials: btoa("admin:"), // Initial value
    BASE_URL: "http://",
    setBASE_URL: (url: string) => set({ BASE_URL: url }),
    BASE_URL_WS: "ws://",
    setBASE_URL_WS: (url: string) => set({ BASE_URL_WS: url }),
    selectedEmbeddingModel: "openai",
    // selectedEmbeddingModel: "fr_long_context",
    setSelectedEmbeddingModel: (embedding: string) => set({ selectedEmbeddingModel: embedding }),
    price_token: {
      "gpt-4o-mini": 1.5e-7,
      "gpt-4o": 5e-6,
      "No_Model": 0,
      'llama-3.1-8b-instant': 5e-8,
      'llama-3.1-70b-versatile': 8e-7,
      'mixtral-8x7b-32768': 3e-7,
    },
    max_documents:{
      'gpt-4o-mini':33,
      'gpt-4o':33,
      'llama-3.1-8b-instant': 33,
      'llama-3.1-70b-versatile': 33,
      'mixtral-8x7b-32768': 10,
    },
    nDocumentsSearched: 5,
    setNDocumentsSearched: (n: number) => set({ nDocumentsSearched: n }),
    nDocumentsSearchedNoLLM: 10,
    setNDocumentsSearchedNoLLM: (n: number) => set({ nDocumentsSearchedNoLLM: n }),
    nDocumentsSearchedDebounced: 5,
    setNDocumentsSearchedDebounced: (n: number) => set({ nDocumentsSearchedDebounced: n }),
    nDocumentsSearchedNoLLMDebounced: 10,
    setNDocumentsSearchedNoLLMDebounced: (n: number) => set({ nDocumentsSearchedNoLLMDebounced: n }),
    groqKeyDebounced: "",
    setGroqKeyDebounced: (key: string) => set({ groqKeyDebounced: key }),
    mistralKeyDebounced: "",
    setMistralKeyDebounced: (key: string) => set({ mistralKeyDebounced: key }),
    openaiKeyDebounced: "",
    setOpenaiKeyDebounced: (key: string) => set({ openaiKeyDebounced: key }),
    isInitialLoadComplete: false,
    setIsInitialLoadComplete: (value: boolean) => set({ isInitialLoadComplete: value }),
    connectionStatus: 'disconnected',
    setConnectionStatus: (status: 'connected' | 'disconnected' | 'connecting') => set({ connectionStatus: status }),
    BASE_URL_ONLINE_WS: `wss://lex-chatbot.epfl.ch/api/ws`,
    BASE_URL_ONLINE_WS_TEST: `wss://lex-chatbot-test.epfl.ch/api/ws`,
    BASE_URL_WS_LOCAL: "ws://127.0.0.1:8000/ws",
    BASE_URL_LOCAL: 'http://epfl-chatbot-backend-service-compose:8000',
    BASE_URL_ONLINE_TEST: '/api',
    BASE_URL_ONLINE: '/api',
    selectedLibrary: 'LEX',
    setSelectedLibrary: (library: string | undefined) => set({ selectedLibrary: library }),
    isLoggedIn: false,
    setIsLoggedIn: (value: boolean) => set({ isLoggedIn: value }),
    rerank: "false",
    setRerank: (value: string) => set({ rerank: value }),
    isLeftPanelOpen: false,
    setIsLeftPanelOpen: (value: boolean) => set({ isLeftPanelOpen: value }),
    interaction_type: "chat",
    setInteractionType: (value: string) => set({ interaction_type: value }),
    filesAttached: [],
    setFilesAttached: (files: File[]) => set({ filesAttached: files }),
    libraryCreationProgress: [],
    setLibraryCreationProgress: (progress: LibraryCreationProgress) => set({ libraryCreationProgress: progress }),
    openaiKey: "",
    setOpenaiKey: (key: string) => set({ openaiKey: key }),
    openaiKeyStatus: 'missing',
    setOpenaiKeyStatus: (status: 'missing' | 'invalid' | 'valid') => set({ openaiKeyStatus: status }),
    conversationID: "",
    setConversationID: (id: string) => set({ conversationID: id }),
    libraries: ['LEX', 'RH'],
    setLibraries: (libraries: string[]) => set({ libraries: libraries }),
    username: "",
    setUsername: (username: string) => set({ username: username }),
  }));
  
export {useStore};
