import React, { useState, useContext, useEffect } from "react";
import Grid from "@mui/material/Grid";
import TextField from "@mui/material/TextField";
import Select from "@mui/material/Select";
import MenuItem from "@mui/material/MenuItem";
import InputLabel from "@mui/material/InputLabel";
import FormControl from "@mui/material/FormControl";
import { Button, Tooltip, Box } from "@mui/material";

import { SelectChangeEvent } from "@mui/material/Select"; // Important!
import ParametersContext from "../../Context";
import Cookies from "js-cookie"; // Import js-cookie
import { IconButton, Drawer } from "@mui/material";
import MenuIcon from "@mui/icons-material/Menu";
import { ParametersProps } from "../../types"; // Adjust the import path as needed
import { useStore } from "../../store"; // Adjust the import path as needed
import OpenAIKeyUpload from "../OpenAIKeyUpload/OpenAIKeyUpload"; // Adjust the import path as needed
import { Trash2 } from "lucide-react";
import { RotateCcw } from "lucide-react";
import { create_ws_connection } from "../../utils";

// In your render method or functional component
const Parameters: React.FC<ParametersProps> = ({
  sx,
  extend,
  chatOnly = false,
}) => {
  const {
    price_token,
    selectedModel,
    setSelectedModel,
    tokensPerInteraction,
    setTokensPerInteraction,
    mistralKey,
    setMistralKey,
    groqKey,
    setGroqKey,
    selectedEmbeddingModel,
    setSelectedEmbeddingModel,
    nDocumentsSearched,
    setNDocumentsSearched,
    nDocumentsSearchedNoLLM,
    setNDocumentsSearchedNoLLM,
    setNDocumentsSearchedDebounced,
    setNDocumentsSearchedNoLLMDebounced,
    setGroqKeyDebounced,
    setMistralKeyDebounced,
    setOpenaiKeyDebounced,
    setIsInitialLoadComplete,
    rerank,
    setRerank,
    max_documents,
    setMessages,
    isInitialLoadComplete,
    selectedLibrary,
    nDocumentsSearchedNoLLMDebounced,
    nDocumentsSearchedDebounced,
    setSocket,
    base64Credentials,
    setBASE_URL,
    setBASE_URL_WS,
    setConnectionStatus,
    BASE_URL,
    BASE_URL_WS,
    openaiKeyDebounced,
    mistralKeyDebounced,
    groqKeyDebounced,
    interaction_type,
    setFilesAttached,
    setOpenaiKeyStatus,
    conversationID,
    setConversationID
  } = useStore();

  const [drawerOpen, setDrawerOpen] = useState(false);
  const resetChat = () => {
    setMessages([]);
    setFilesAttached([]);
    if (isInitialLoadComplete) {
      create_ws_connection(
        selectedLibrary,
        selectedModel,
        nDocumentsSearchedNoLLMDebounced,
        nDocumentsSearchedDebounced,
        setMessages,
        setSocket,
        base64Credentials,
        setBASE_URL,
        setBASE_URL_WS,
        selectedEmbeddingModel,
        setConnectionStatus,
        BASE_URL,
        BASE_URL_WS,
        setConversationID,
        openaiKeyDebounced,
        mistralKeyDebounced,
        groqKeyDebounced,
        interaction_type,
        rerank,
        setOpenaiKeyStatus,
        
      );
    }
  };


  //set debounced values 300 ms after the user stops typing
  useEffect(() => {
    const handler = setTimeout(() => {
      setNDocumentsSearchedDebounced(nDocumentsSearched);
      setNDocumentsSearchedNoLLMDebounced(nDocumentsSearchedNoLLM);
      setGroqKeyDebounced(groqKey);
      setMistralKeyDebounced(mistralKey);
    }, 600);
    return () => {
      clearTimeout(handler);
    };
  }, [
    nDocumentsSearched,
    nDocumentsSearchedNoLLM,
    setNDocumentsSearchedDebounced,
    setNDocumentsSearchedNoLLMDebounced,
    setGroqKeyDebounced,
    setMistralKeyDebounced,
    groqKey,
    mistralKey,
  ]);

  useEffect(() => {
    const loadKeysFromCookies = () => {
      const mistralKey = Cookies.get("mistralKey") || "";
      const groqKey = Cookies.get("groqKey") || "";

      setMistralKey(mistralKey);
      setGroqKey(groqKey);

      setIsInitialLoadComplete(true); // Set to true once all keys are loaded
    };

    loadKeysFromCookies();
  }, [setMistralKey, setGroqKey]); // Depend on setter functions if they are indeed dependencies

  const handleModelChange = (event: SelectChangeEvent<string>) => {
    setSelectedModel(event.target.value);
  };

  const handleMistralApiKeyChange = (
    event: React.ChangeEvent<HTMLInputElement>
  ) => {
    const newKey = event.target.value;
    setMistralKey(newKey);
    Cookies.set("mistralKey", newKey, { expires: 365 }); // Similarly save mistralKey
  };

  const handleGroqApiKeyChange = (
    event: React.ChangeEvent<HTMLInputElement>
  ) => {
    const newKey = event.target.value;
    setGroqKey(newKey);
    Cookies.set("groqKey", newKey, { expires: 365 });
  };

  const handleEmbeddingModelChange = (event: SelectChangeEvent<string>) => {
    setSelectedEmbeddingModel(event.target.value);
  };

  const handleNdocumentChange = (
    event: React.ChangeEvent<HTMLInputElement>
  ) => {
    const newNDocuments = Number(event.target.value);

    if (
      selectedModel !== "No_Model" &&
      Number(newNDocuments) > max_documents[selectedModel]
    ) {
      alert(
        `The maximum number of documents searched for ${selectedModel} is ${max_documents[selectedModel]}`
      );
      return;
    }
    if (Number(newNDocuments) < 1) {
      return;
    }
    setNDocumentsSearched(Number(newNDocuments));
    setTokensPerInteraction(calculateNumberOfTokens(Number(newNDocuments)));
  };

  function calculateInteractionPrice(model: string, n_tokens_sent: number) {
    // Implement your calculation logic
    // Example:
    if (model && n_tokens_sent) {
      return (n_tokens_sent * price_token[model]).toFixed(3);
    } else {
      return 50;
    }
  }

  function calculateNumberOfTokens(nDocumentsSearched: number) {
    // Implement your calculation logic
    // Example:

    return nDocumentsSearched * 3000;
  }

  function handleNdocumentChangeNoLLM(
    event: React.ChangeEvent<HTMLInputElement>
  ) {
    const newNDocuments = Number(event.target.value);
    if (Number(newNDocuments) < 1) {
      return;
    }
    setNDocumentsSearchedNoLLM(Number(newNDocuments));
  }

  function handleRerankChange(event: SelectChangeEvent<string>) {
    setRerank(event.target.value);
  }
  if (chatOnly) {
    return (
      <>
        <Box position="relative" mb={2}>
          <Grid container spacing={1}>
            <Grid item xs={10} sm={4}>
              <FormControl fullWidth>
                <InputLabel id="model-select-label">LLM</InputLabel>
                <Select
                  labelId="model-select-label"
                  value={selectedModel}
                  label="Model"
                  onChange={handleModelChange}
                >
                  <MenuItem value="gpt-4o-mini">GPT-4o-mini</MenuItem>
                  <MenuItem value="gpt-4o">GPT-4o</MenuItem>
                  {extend && (
                    <MenuItem value="llama3.1-8b-instant">llama3.1-8b</MenuItem>
                  )}
                  {extend && (
                    <MenuItem value="llama3.1-70b-versatile">
                      llama3.1-70b
                    </MenuItem>
                  )}
                  {extend && (
                    <MenuItem value="mixtral-8x7b-32768">mixtral-8x7b</MenuItem>
                  )}
                </Select>
              </FormControl>
            </Grid>
            <Grid item xs={10} sm={8}>
              <OpenAIKeyUpload />
            </Grid>
          </Grid>
          <Box position="absolute" top={0} right={0}>
            <Tooltip title="Reset Chat">
              <IconButton
                onClick={resetChat}
                color="default"
                aria-label="reset chat"
                size="small"
              >
                <RotateCcw size={20} />
              </IconButton>
            </Tooltip>
          </Box>
        </Box>
      </>
    );
  }

  return (
    <>
      <Box position="relative" mb={2}>
        <Grid container spacing={1}>
          <Grid item xs={10} sm={4}>
            <FormControl fullWidth>
              <InputLabel id="model-select-label">LLM</InputLabel>
              <Select
                labelId="model-select-label"
                value={selectedModel}
                label="Model"
                onChange={handleModelChange}
              >
                <MenuItem value="gpt-4o-mini">GPT-4o-mini</MenuItem>
                <MenuItem value="gpt-4o">GPT-4o</MenuItem>
                {extend && (
                  <MenuItem value="llama3.1-8b-instant">llama3.1-8b</MenuItem>
                )}
                {extend && (
                  <MenuItem value="llama3.1-70b-versatile">
                    llama3.1-70b
                  </MenuItem>
                )}
                {extend && (
                  <MenuItem value="mixtral-8x7b-32768">mixtral-8x7b</MenuItem>
                )}
                <MenuItem value="No_Model">No LLM</MenuItem>
              </Select>
            </FormControl>
          </Grid>
          {extend && (
            <Grid item xs={10} sm={6}>
              <FormControl fullWidth>
                <InputLabel id="rerank">
                  rerank (slower, better results)
                </InputLabel>
                <Select
                  labelId="rerank"
                  value={rerank}
                  label="rerank (slower, better results)"
                  onChange={handleRerankChange}
                >
                  <MenuItem value="true">ON (slower, better results)</MenuItem>
                  <MenuItem value="false">OFF</MenuItem>
                </Select>
              </FormControl>
            </Grid>
          )}
          {selectedModel != "No_Model" && (
            <>
              <Grid item xs={10} sm={3} lg={2}>
                <TextField
                  fullWidth
                  label="Price per call"
                  type="string"
                  value={
                    String(
                      calculateInteractionPrice(
                        selectedModel,
                        tokensPerInteraction
                      )
                    ) + " CHF"
                  }
                />
              </Grid>
              <Grid item xs={10} sm={3} lg={2}>
                <TextField
                  fullWidth
                  label="N_docs_searched"
                  type="number"
                  value={nDocumentsSearched}
                  onChange={handleNdocumentChange}
                />
              </Grid>
            </>
          )}
          {
            <Grid item xs={10} sm={10} lg={10}>
              <OpenAIKeyUpload />
            </Grid>
          }
          {extend && (
            <Grid item xs={10} sm={6}>
              <FormControl fullWidth>
                <InputLabel id="embedding">moteur de recherche</InputLabel>
                <Select
                  labelId="embedding"
                  value={selectedEmbeddingModel}
                  label="embedding_model"
                  onChange={handleEmbeddingModelChange}
                >
                  <MenuItem value="gte">gte</MenuItem>
                  <MenuItem value="camembert">Camembert</MenuItem>
                  <MenuItem value="embaas">embaas </MenuItem>
                  <MenuItem value="fr_long_context">fr_long_context</MenuItem>
                  {extend && (
                    <MenuItem value="mpnet">Bert-mpnet (en only)</MenuItem>
                  )}
                  {extend && (
                    <MenuItem value="mistral">Mistral (bilingue)</MenuItem>
                  )}
                  <MenuItem value="openai">OpenAi (bilingue)</MenuItem>
                </Select>
              </FormControl>
            </Grid>
          )}
          {selectedEmbeddingModel === "mistral" && (
            <Grid item xs={10} sm={3} lg={3}>
              <TextField
                fullWidth
                label="mistral_key"
                type="password"
                value={mistralKey}
                onChange={handleMistralApiKeyChange}
              />
            </Grid>
          )}
          {["llama3-70b", "mixtral-8x7b", "llama3-8b"].includes(
            selectedModel
          ) && (
            <Grid item xs={10} sm={3} lg={3}>
              <TextField
                fullWidth
                label="groq_key"
                type="password"
                value={groqKey}
                onChange={handleGroqApiKeyChange}
              />
            </Grid>
          )}
          {selectedModel == "No_Model" && (
            <Grid item xs={10} sm={3} lg={2}>
              <TextField
                fullWidth
                label="N_resuts_no_LLM"
                type="number"
                value={nDocumentsSearchedNoLLM}
                onChange={handleNdocumentChangeNoLLM}
              />
            </Grid>
          )}
        </Grid>
        <Box position="absolute" top={0} right={0}>
          <Tooltip title="Reset Chat">
            <IconButton
              onClick={resetChat}
              color="default"
              aria-label="reset chat"
              size="small"
            >
              <RotateCcw size={20} />
            </IconButton>
          </Tooltip>
        </Box>
      </Box>
    </>
  );
};

export default Parameters;
