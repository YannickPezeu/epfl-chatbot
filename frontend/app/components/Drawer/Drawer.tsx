import React from 'react';
import Drawer from '@mui/material/Drawer';
import Box from '@mui/material/Box';
import { PDFViewer } from '@react-pdf/renderer'; // Si vous utilisez react-pdf
import { useStore } from '../../store';

type PDFDrawerProps = {
  isOpen: boolean;
  onClose: () => void;
  library_name: string;
};

const PDFDrawer: React.FC<PDFDrawerProps> = ({ 
  isOpen, 
  onClose,  
  library_name }) => {

    const { BASE_URL, pdfId, pdfPageNumber } = useStore();
  
  const fullPdfUrl = `${BASE_URL}/source_docs/view-source-doc/${pdfId}?library_name=${library_name}#page=${pdfPageNumber}`;
  // console.log('fullPdfUrl', fullPdfUrl)

  return (
    <Drawer anchor={'right'} open={isOpen} onClose={onClose} PaperProps={{
        sx: {
          width: '80%', // This will make the Drawer take up 80% of the screen width
        },
      }}>
      <Box
        sx={{ width: '100%' }}
        role="presentation"
      >
        <iframe src={fullPdfUrl} style={{ width: '100%', height: '100vh' }} title="PDF Viewer"></iframe>
      </Box>
    </Drawer>
  );
};

export default PDFDrawer;
