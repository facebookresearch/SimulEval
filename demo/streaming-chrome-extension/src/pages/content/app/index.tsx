import ScopedCssBaseline from '@mui/material/ScopedCssBaseline';

// import * as React from 'react';
import {createRoot} from 'react-dom/client';
import App from '@src/pages/content/app/app';
import refreshOnUpdate from 'virtual:reload-on-update-in-view';
import {ThemeProvider} from '@mui/material/styles';
import theme from './theme';

// Roboto font for mui ui library
// import '@fontsource/roboto/300.css';
// import '@fontsource/roboto/400.css';
// import '@fontsource/roboto/500.css';
// import '@fontsource/roboto/700.css';

refreshOnUpdate('pages/content');

const root = document.createElement('div');
root.id = 'seamless-translation-content-view-root';

console.log('Appending react root to body');
document.body.append(root);

createRoot(root).render(
  <ThemeProvider theme={theme}>
    <ScopedCssBaseline>
      <App />
    </ScopedCssBaseline>
  </ThemeProvider>,
);
