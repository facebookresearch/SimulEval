import {createTheme} from '@mui/material/styles';
// import {red} from '@mui/material/colors';

const Z_INDEX_BASE = 9999999;

function getHtmlFontSize(): number | null {
  let fontSize = null;
  try {
    // NOTE: Even when this is not explicitly set it still returns a value
    const fontSizeString = window
      .getComputedStyle(document.getElementsByTagName('html')[0])
      .getPropertyValue('font-size');
    fontSize = parseInt(fontSizeString, 10);
  } catch (e) {
    console.error('Error getting font size', e);
  }

  return fontSize;
}

const htmlFontSize = getHtmlFontSize();

const htmlFontSizeObject =
  htmlFontSize == null ? {} : {htmlFontSize: htmlFontSize};

const themeObject = {
  palette: {
    primary: {
      main: '#8595A4',
    },
    text: {primary: '#1C2A33'},
  },
  typography: {
    ...htmlFontSizeObject,
    fontFamily: [
      'Optimistic Text',
      'Roboto',
      '"Helvetica Neue"',
      'Arial',
      'sans-serif',
    ].join(','),
    h1: {fontSize: 16, fontWeight: '500'},
    body1: {fontSize: 16},
  },
  // Because our chrome extension uses a high z-index, we need to
  // provide that base here so MUI stuff renders correctly
  zIndex: {
    mobileStepper: Z_INDEX_BASE + 1000,
    fab: Z_INDEX_BASE + 1050,
    speedDial: Z_INDEX_BASE + 1050,
    appBar: Z_INDEX_BASE + 1100,
    drawer: Z_INDEX_BASE + 1200,
    modal: Z_INDEX_BASE + 1300,
    snackbar: Z_INDEX_BASE + 1400,
    tooltip: Z_INDEX_BASE + 1500,
  },
};

console.log('themeObject', themeObject);

// A custom theme for this app
const theme = createTheme(themeObject);

export default theme;
