import React, { useState, useEffect } from 'react';
import { Grid, Button, TextField, Dialog, DialogActions, DialogContent, DialogContentText, DialogTitle } from '@mui/material';
import { styled } from '@mui/material/styles';
import { useStore } from '../../store';
import { Link } from "@mui/material";
import { create_ws_connection } from "../../utils";

interface StyledButtonProps {
  status?: 'missing' | 'invalid' | 'valid';
}

const StyledButton = styled(Button, {
  shouldForwardProp: (prop) => prop !== 'status',
})<StyledButtonProps>(({ theme, status }) => ({
  backgroundColor: 'transparent',
  color: 'black',
  textTransform: 'none',
  padding: '16px',
  borderRadius: '4px',
  border: '1px solid lightgrey',
  fontSize: '14px',
  '&:hover': {
    backgroundColor: status === 'valid' ? '#e6ffe6' : status === 'missing' || status === 'invalid' ? '#ffe6e6' : 'grey',
  },
  marginBottom: theme.spacing(2),
  marginTop: 0,
  whiteSpace: 'nowrap',
  overflow: 'hidden',
  textOverflow: 'ellipsis',
  width: '100%',
  minHeight: '48px',
}));

const OpenAIKeyUpload: React.FC = () => {
  const { 
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
    openaiKeyDebounced,
    mistralKeyDebounced,
    groqKeyDebounced,
    interaction_type,
    rerank, 
    openaiKey,
    setOpenaiKey,
    openaiKeyStatus, setOpenaiKeyStatus,
    conversationID,
    setConversationID

  } = useStore();

  const [isModalOpen, setIsModalOpen] = useState(false);
  const [isConfirmationOpen, setIsConfirmationOpen] = useState(false);
  // const [openaiKey, setOpenaiKey] = useState('');
  // const [openaiKeyStatus, setOpenaiKeyStatus] = useState<'missing' | 'invalid' | 'valid'>('missing');

  useEffect(() => {
    checkOpenAIKeyStatus();
  }, []);

  const checkOpenAIKeyStatus = async () => {
    try {
      const response = await fetch(`${BASE_URL}/auth/check-openai-key`, {
        method: 'GET',
        credentials: 'include',
      });
      const data = await response.json();
      console.log('checkOpenAIKeyStatus:', data);
      setOpenaiKeyStatus(data.openaiKeyStatus);
    } catch (error) {
      console.error('Error checking OpenAI key status:', error);
      setOpenaiKeyStatus('missing');
    }
  };

  const handleOpenModal = () => setIsModalOpen(true);
  const handleCloseModal = () => setIsModalOpen(false);
  const handleOpenConfirmation = () => setIsConfirmationOpen(true);
  const handleCloseConfirmation = () => setIsConfirmationOpen(false);

  const handleOpenAiKeyChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    setOpenaiKey(event.target.value);
  };

  const handleSubmit = async () => {
    handleCloseConfirmation();
    handleCloseModal();
    
    try {
      const upload_key_url = `${BASE_URL}/auth/upload-openai-key`;
      const response = await fetch(upload_key_url, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ 'openai_key': openaiKey }),
        credentials: 'include',
      });

      if (response.ok) {
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
          conversationID,
        );
        alert('OpenAI key uploaded successfully');
        checkOpenAIKeyStatus();
      } 
      else {
        const errorData = await response.json();
        console.log('mytest Error uploading OpenAI key:', errorData);
        checkOpenAIKeyStatus();
        alert(errorData.detail);
      }
    } catch (error) {
      console.error('Error uploading OpenAI key:', error);
      checkOpenAIKeyStatus();
      alert(error instanceof Error ? error.message : 'Failed to upload OpenAI key');
    }
  };

  const getButtonText = () => {
    if (openaiKeyStatus === 'valid') return 'OpenAI Key';
    return 'Upload OpenAI Key';
  };

  const getHoverText = () => {
    switch (openaiKeyStatus) {
      case 'missing':
        return 'No OpenAI key detected, please upload key';
      case 'invalid':
        return 'OpenAI key invalid, please upload a new key';
      case 'valid':
        return 'Uploaded OpenAI key is valid';
      default:
        return '';
    }
  };
  return (
    <Grid item xs={12} sm={6} lg={6}>
      <StyledButton
        onClick={handleOpenModal}
        status={openaiKeyStatus}
        title={getHoverText()}
      >
        {getButtonText()}
      </StyledButton>
      
      <Dialog open={isModalOpen} onClose={handleCloseModal}>
        <DialogTitle>Upload OpenAI Key</DialogTitle>
        <DialogContent>
          <DialogContentText>
            Please enter your OpenAI API key below.
          </DialogContentText>
          <TextField
            autoFocus
            margin="dense"
            label="OpenAI Key"
            type="password"
            fullWidth
            variant="outlined"
            value={openaiKey}
            onChange={handleOpenAiKeyChange}
          />
          <DialogContentText>
            <Link href="https://platform.openai.com/api-keys" target="_blank">
              Create new key
            </Link>
          </DialogContentText>
          <DialogContentText>
            <Link href="https://platform.openai.com/settings/organization/billing/overview" target="_blank">
              Add money on the key
            </Link>
          </DialogContentText>
        </DialogContent>
        <DialogActions>
          <Button onClick={handleCloseModal}>Cancel</Button>
          <Button onClick={handleOpenConfirmation}>Upload</Button>
        </DialogActions>
      </Dialog>

      <Dialog open={isConfirmationOpen} onClose={handleCloseConfirmation}>
        <DialogTitle>Confirm Upload</DialogTitle>
        <DialogContent>
          <DialogContentText>
            Are you sure you want to upload this key? If you upload an invalid key, you won&apos;t be able to use the LLM anymore.
          </DialogContentText>
        </DialogContent>
        <DialogActions>
          <Button onClick={handleCloseConfirmation}>Cancel</Button>
          <Button onClick={handleSubmit}>Confirm Upload</Button>
        </DialogActions>
      </Dialog>
    </Grid>
  );
};

export default OpenAIKeyUpload;