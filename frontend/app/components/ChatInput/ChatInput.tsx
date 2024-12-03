import React, { useState, useRef } from "react";
import { TextField, InputAdornment, IconButton, Chip, Box, CircularProgress } from '@mui/material';
import SendIcon from '@mui/icons-material/Send';
import AttachFileIcon from '@mui/icons-material/AttachFile';
import CloseIcon from '@mui/icons-material/Close';
import { ChatInputProps } from "../../types";
import { useStore } from "../../store";

export const ChatInput: React.FC<ChatInputProps> = ({ onSendMessage, onFileAttach, sx }) => {
  const [inputValue, setInputValue] = useState("");
  const [processingFiles, setProcessingFiles] = useState<Set<string>>(new Set());
  const fileInputRef = useRef<HTMLInputElement>(null);

  const { filesAttached, setFilesAttached, socket } = useStore();

  const isProcessing = processingFiles.size > 0;

  const sendMessage = () => {
    if (isProcessing) return;
    const trimmedInput = inputValue.trim();
    if (!trimmedInput) return;
    onSendMessage(trimmedInput);
    setInputValue("");
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !isProcessing) {
      e.preventDefault();
      sendMessage();
    }
  };

  const handleFileChange = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const files = event.target.files;
    if (files && files.length > 0) {
      const newFiles = Array.from(files).filter(file => file.type === 'application/pdf');
      
      if (newFiles.length === 0) {
        alert("Please select only PDF files.");
        return;
      }
      
      // Mark files as processing
      setProcessingFiles(prev => {
        const updated = new Set(prev);
        newFiles.forEach(file => updated.add(file.name));
        return updated;
      });

      try {
        await onFileAttach(newFiles);
      } catch (error) {
        console.error("Error processing files:", error);
      } finally {
        setProcessingFiles(prev => {
          const updated = new Set(prev);
          newFiles.forEach(file => updated.delete(file.name));
          return updated;
        });
        // Reset the file input
        if (fileInputRef.current) {
          fileInputRef.current.value = '';
        }
      }
    }
  };

  const triggerFileInput = () => {
    fileInputRef.current?.click();
  };

  const handleRemoveFile = async (fileToRemove: File) => {
    try {
      if (socket && socket.readyState === WebSocket.OPEN) {
        const removeFileMessage = JSON.stringify({
          type: 'remove_file',
          filename: fileToRemove.name
        });
        socket.send(removeFileMessage);

        const response = await new Promise((resolve, reject) => {
          const messageHandler = (event: MessageEvent) => {
            const data = JSON.parse(event.data);
            if (data.type === 'file_removed' && data.filename === fileToRemove.name) {
              socket.removeEventListener('message', messageHandler);
              resolve(data);
            }
          };
          socket.addEventListener('message', messageHandler);
          
          setTimeout(() => {
            socket.removeEventListener('message', messageHandler);
            reject(new Error('Server did not respond to file removal request'));
          }, 5000);
        });

        console.log('Server response:', response);
        setFilesAttached(filesAttached.filter(file => file !== fileToRemove));
      } else {
        throw new Error('WebSocket is not open');
      }
    } catch (error) {
      console.error('Error removing file:', error);
    }
  };

  return (
    <Box sx={{ width: '100%', ...sx }}>
      {filesAttached.length > 0 && (
        <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1, mb: 2 }}>
          {filesAttached.map((file, index) => (
            <Chip
              key={index}
              label={
                <span>
                  {file.name}
                  {processingFiles.has(file.name) && <CircularProgress size={16} sx={{ ml: 1 }} />}
                </span>
              }
              onDelete={processingFiles.has(file.name) ? undefined : () => handleRemoveFile(file)}
              deleteIcon={<CloseIcon />}
              size="small"
            />
          ))}
        </Box>
      )}
      <div style={{ display: 'flex', alignItems: 'center', width: '100%' }}>
        <input
          type="file"
          ref={fileInputRef}
          onChange={handleFileChange}
          style={{ display: 'none' }}
          multiple
          accept=".pdf,application/pdf"
        />
        <TextField
          sx={{ mt: 3, flexGrow: 1, borderRadius: 2 }}
          variant="outlined"
          label="Type your message..."
          value={inputValue}
          onChange={(e) => setInputValue(e.target.value)}
          onKeyDown={handleKeyDown}
          InputProps={{
            startAdornment: (
              <InputAdornment position="start">
                <IconButton onClick={triggerFileInput} edge="start" aria-label="attach file" disabled={isProcessing}>
                  <AttachFileIcon />
                </IconButton>
              </InputAdornment>
            ),
            endAdornment: (
              <InputAdornment position="end">
                <IconButton onClick={sendMessage} edge="end" aria-label="send message" disabled={isProcessing}>
                  <SendIcon />
                </IconButton>
              </InputAdornment>
            )
          }}
        />
      </div>
    </Box>
  );
};