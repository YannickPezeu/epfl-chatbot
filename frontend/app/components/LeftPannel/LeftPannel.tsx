import React from 'react';
import { Drawer, List, ListItem, ListItemText, Button, IconButton } from '@mui/material';
import AddIcon from '@mui/icons-material/Add';
import ChevronLeftIcon from '@mui/icons-material/ChevronLeft';
import CloseIcon from '@mui/icons-material/Close';

interface LeftPanelProps {
  isOpen: boolean;
  onClose: () => void;
  libraries: string[];
  onLibrarySelect: (library: string) => void;
  onCreateNewLibrary: () => void;
  selectedLibrary: string | null;
  onDeleteLibrary: (library: string) => void;
}

const LeftPanel: React.FC<LeftPanelProps> = ({
  isOpen,
  onClose,
  libraries,
  onLibrarySelect,
  onCreateNewLibrary,
  selectedLibrary,
  onDeleteLibrary,
}) => {
  return (
    <Drawer
      variant="temporary"
      anchor="left"
      open={isOpen}
      onClose={onClose}
      ModalProps={{
        keepMounted: true, // Better open performance on mobile
      }}
      sx={{
        '& .MuiDrawer-paper': {
          width: 240,
          boxSizing: 'border-box',
        },
      }}
    >
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '10px' }}>
        <h2>Libraries</h2>
        <IconButton onClick={onClose}>
          <ChevronLeftIcon />
        </IconButton>
      </div>
      <List>
        <Button
          variant="contained"
          startIcon={<AddIcon />}
          onClick={onCreateNewLibrary}
          sx={{ margin: '10px' }}
        >
          Create New Library
        </Button>
        {libraries.map((db, index) => (
          <ListItem
            key={index}
            sx={{
              backgroundColor: selectedLibrary === db ? 'lightblue' : 'white',
              '&:hover': {
                backgroundColor: selectedLibrary === db ? 'lightblue' : 'rgba(0, 0, 0, 0.04)',
              },
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center',
            }}
          >
            <ListItemText 
              primary={db} 
              onClick={() => onLibrarySelect(db)}
              sx={{ cursor: 'pointer', flexGrow: 1 }}
            />
            {!['RH','LEX', 'LEX AND RH', 'no_library'].includes(db) &&<IconButton
              onClick={(e) => {
                e.stopPropagation();
                onDeleteLibrary(db);
              }}
              size="small"
            >
              <CloseIcon />
            </IconButton>}
          </ListItem>
        ))}
      </List>
    </Drawer>
  );
};

export default LeftPanel;