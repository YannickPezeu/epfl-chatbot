// Your main component file
import React, { useState } from 'react';
import { IconButton, Drawer, Box } from '@mui/material';
import MenuIcon from '@mui/icons-material/Menu';
import Parameters from './Parameters';  // Assuming the component is exported
import { ParametersProps } from "../../types"; // Adjust the import path as needed

const ParametersForm: React.FC<ParametersProps> = ({sx, extend, chatOnly=false}) => {
  const [drawerOpen, setDrawerOpen] = useState(false);

  return (
    <>
      <IconButton
        edge="start"
        color="inherit"
        aria-label="menu"
        sx={{ mr: 2, display: { sm: 'none' } }}
        onClick={() => setDrawerOpen(true)}
      >
        <MenuIcon />
      </IconButton>

      <Drawer
        anchor="left"
        open={drawerOpen}
        onClose={() => setDrawerOpen(false)}
      >
       <Box
          sx={{ width: 250, padding: 2}}
          role="presentation"
        >
          <Parameters extend={extend} chatOnly={chatOnly} />
        </Box>
      </Drawer>

      {/* Display full layout on larger screens */}
      <Box sx={{ display: { xs: 'none', sm: 'block' } }}>
        <Parameters extend={extend} chatOnly={chatOnly} />
      </Box>
    </>
  );
};

export default ParametersForm;
