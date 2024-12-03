import React from 'react';

export interface ParametersContext {
  selectedModel: "gpt-4o-mini" | "gpt-4o";
  tokensPerInteraction: number;
  onModelChange: (model: "gpt-4o-mini" | "gpt-4o") => void;
  onTokensChange: (tokens: number) => void;
}

const ParametersContext = React.createContext<ParametersContext | null>(null);

export default ParametersContext;
