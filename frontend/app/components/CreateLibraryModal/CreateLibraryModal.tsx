import React, { useCallback, useState, useEffect } from 'react';
import { Modal, Box, Typography, Button, LinearProgress, Tooltip } from '@mui/material';
import { useDropzone } from 'react-dropzone';
import { useStore } from "../../store";

interface CreateLibraryModalProps {
  isOpen: boolean;
  onClose: () => void;
  onCreateLibrary: (library_name: string, files: File[], selectedEmbeddingModel: string, special_prompt: string) => Promise<void>;
}

const CreateLibraryModal: React.FC<CreateLibraryModalProps> = ({ isOpen, onClose, onCreateLibrary }) => {
  const [libraryName, setLibraryName] = useState('');
  const [files, setFiles] = useState<File[]>([]);
  const [isUploading, setIsUploading] = useState(false);
  const [specialPrompt, setSpecialPrompt] = useState('');
  const [progress, setProgress] = useState(0);
  const [libraryPrice, setLibraryPrice] = useState(0);

  const { selectedEmbeddingModel, libraryCreationProgress } = useStore();

  const onDrop = useCallback((acceptedFiles: File[]) => {
    setFiles(prevFiles => [...prevFiles, ...acceptedFiles]);
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'application/pdf': ['.pdf']
    }
  });

  useEffect(() => {
    if (isUploading && libraryName) {
      const currentLibraryProgress = libraryCreationProgress.find(item => Object.keys(item)[0] === libraryName);
      if (currentLibraryProgress) {
        const currentProgressObject = currentLibraryProgress[libraryName];
        const currentProgress = currentProgressObject.progress;
        setLibraryPrice(currentProgressObject.price);
        setProgress(currentProgress);
        
        console.log('Progress updated:', currentProgress);
        
        if (currentProgress >= 100) {
          resetState();
        }
      }
    }
  }, [libraryCreationProgress, libraryName, isUploading]);

  const resetState = () => {
    setIsUploading(false);
    setFiles([]);
    setLibraryName('');
    setSpecialPrompt('');
    setProgress(0);
    onClose();
    setLibraryPrice(0);
  };

  const handleCreate = async () => {
    if (libraryName && files.length > 0) {
      setIsUploading(true);
      try {
        await onCreateLibrary(libraryName, files, selectedEmbeddingModel, specialPrompt);
        // The modal will be closed by resetState when progress reaches 100%
      } catch (error) {
        console.error('Error creating library:', error);
        alert('An error occurred while creating the library: ' + error);
        resetState();
      }
    }
  };

  return (
    <Modal open={isOpen} onClose={onClose}>
      <Box sx={{
        position: 'absolute',
        top: '50%',
        left: '50%',
        transform: 'translate(-50%, -50%)',
        width: 400,
        bgcolor: 'background.paper',
        boxShadow: 24,
        p: 4,
        borderRadius: 2,
      }}>
        <Typography variant="h6" component="h2" gutterBottom>
          Create New Library
        </Typography>
        <input
          type="text"
          placeholder="Library Name"
          value={libraryName}
          onChange={(e) => setLibraryName(e.target.value)}
          style={{ width: '100%', marginBottom: '1rem', padding: '0.5rem', borderRadius: '4px', border: '1px solid #cccccc' }}
        />
        <Tooltip title="Instruct LLM for specific behaviour (funny, serious...)" arrow placement="top">
          <input
            type="text"
            placeholder="General Instructions (optional)"
            value={specialPrompt}
            onChange={(e) => setSpecialPrompt(e.target.value)}
            style={{ width: '100%', marginBottom: '1rem', padding: '0.5rem', borderRadius: '4px', border: '1px solid #cccccc' }}
          />
        </Tooltip>
        <div {...getRootProps()} style={{
          border: '2px dashed #cccccc',
          borderRadius: '4px',
          padding: '20px',
          textAlign: 'center',
          cursor: 'pointer',
          marginBottom: '1rem'
        }}>
          <input {...getInputProps()} />
          {
            isDragActive ?
              <p>Drop the PDF files here ...</p> :
              <p>Drag n drop some PDF files here, or click to select files</p>
          }
        </div>
        {files.length > 0 && (
          <Typography variant="body2" gutterBottom>
            {files.length} file(s) selected
          </Typography>
        )}
        {isUploading && (
          <Box sx={{ width: '100%', marginBottom: '1rem' }}>
            <LinearProgress variant="determinate" value={progress} />
            <Typography variant="body2" color="text.secondary" align="center">
              {`${Math.round(progress)}%`}
            </Typography>
          </Box>
        )}
        <Button 
          variant="contained" 
          onClick={handleCreate} 
          disabled={!libraryName || files.length === 0 || isUploading}
        >
          Create Library
        </Button>
        {isUploading && libraryPrice && (
          <Typography variant="body2" align="center" sx={{ mt: 2 }}>
            {`Price: $${libraryPrice}`}
          </Typography>
        )}
      </Box>
    </Modal>
  );
};

export default CreateLibraryModal;