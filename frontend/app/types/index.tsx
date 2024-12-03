// src/types/index.ts

  
export interface ParametersProps {
  extend?: boolean;
  sx?: any;
  chatOnly?: boolean;
}

export interface Source {
  page_number: string;
  pdf_id: any;
  title: string;
  page: string;
  document_index: number;
  url:string;
}


export type ChatMessageProps = {
  sender: string;
  message: {message_content: string; tool_input?: string; tool_name?: string;};
  sources: Source[];
  modelUsed: string;
  messageType?: string;
  messageId: string;
  nTokensInput?: number;
};

export interface ChatContainerProps {
  messages:  ChatMessageProps[]
  sx?: any;
  chatOnly?: boolean;
  // onClickSource: (pdfUrl: string, pdfPageNumber: number) => void; // Assuming the onClickSource function signature
}

export type ChatInputProps = {
  onSendMessage: (message: string) => void;
  onFileAttach: (files: File[]) => void;
  sx?: any;
};

export type ChatLayoutProps = {
  onSendMessage: (message: string) => void;
  messages: ChatMessageProps[];
  extend?: boolean;
  onLogout: () => void;
  chatOnly?: boolean;
  onFileAttach: (files: File[]) => void;
}

type ConnectionStatus = 'connected' | 'disconnected' | 'connecting'; // Example statuses
type ButtonColor = 'inherit' | 'primary' | 'secondary' | 'error' | 'info' | 'success' | 'warning';
