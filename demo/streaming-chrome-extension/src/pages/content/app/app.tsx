import {useEffect, useLayoutEffect, useRef, useState} from 'react';
import Button from '@mui/material/Button';
import Typography from '@mui/material/Typography';
import InputLabel from '@mui/material/InputLabel';
import FormControl from '@mui/material/FormControl';
import Select, {SelectChangeEvent} from '@mui/material/Select';
import MenuItem from '@mui/material/MenuItem';
import Stack from '@mui/material/Stack';
// import SvgIcon from '@mui/material/SvgIcon';
import SeamlessLogoPath from '@assets/img/seamless.svg';
// import logo from '@assets/img/logo.svg';
// import SeamlessLogo from './SeamlessLogo';
import Accordion from '@mui/material/Accordion';
import AccordionSummary from '@mui/material/AccordionSummary';
import AccordionDetails from '@mui/material/AccordionDetails';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import TextField from '@mui/material/TextField';

function sendRuntimeMessage(message) {
  return new Promise((resolve) => {
    chrome.runtime.sendMessage(message, (res) => {
      resolve(res);
    });
  });
}

function getStorage(key) {
  return new Promise((resolve) => {
    chrome.storage.local.get([key], (result) => {
      resolve(result[key]);
    });
  });
}

type StreamingStatus = 'stopped' | 'running' | 'starting';

const buttonLabelMap: {[key in StreamingStatus]: string} = {
  stopped: 'Start Streaming',
  running: 'Stop Streaming',
  starting: 'Starting...',
};

const INPUT_LANGUAGES = {
  'English (US)': 'en-US',
  'Spanish (Spain)': 'es-ES',
};

const OUTPUT_LANGUAGES = {
  'English (US)': 'en-US',
  'Spanish (Spain)': 'es-ES',
};

type ServerData = {
  eos: boolean;
  event: string;
  payload: string;
  sample_rate?: number;
  server_active_connections?: number;
};

export default function App() {
  const [streamingStatus, setStreamingStatus] =
    useState<StreamingStatus>('stopped');
  const [wsConnOpen, setWsConnOpen] = useState<boolean>(false);
  const [serverReady, setServerReady] = useState<boolean>(false);
  const [websocketServerUrl, setWebsocketServerUrl] = useState<string>('');
  const [inputLang, setInputLang] = useState('es-ES');
  const [targetLang, setTargetLang] = useState('en-US');
  const [receivedData, setReceivedData] = useState<Array<ServerData>>([]);
  const lastTranslationResultRef = useRef(null);

  const translationSentences: Array<string> = receivedData
    .reduce(
      (acc, data) => {
        // TODO: Add special handling if the payload starts/ends with an apostrophe?
        const newAcc = [
          ...acc.slice(0, -1),
          acc[acc.length - 1].trim() + ' ' + data.payload,
        ];
        if (data.eos) {
          newAcc.push('');
        }

        return newAcc;
      },
      [''],
    )
    .filter((s) => s.trim().length !== 0);

  console.log('translationSentences', translationSentences);

  useEffect(() => {
    console.log('content view loaded');
  }, []);

  useEffect(() => {
    // Receive data from Option Tab or Background
    const onMessage = (request, _sender, sendResponse) => {
      const {type} = request;
      const data: ServerData = request.data;

      console.log('Data received in react from option tab:', request);

      switch (type) {
        case 'FROM_OPTION_TRANSLATION_TEXT':
          console.log(data);

          setReceivedData((prev) => [...prev, data]);

          if (
            data?.server_active_connections != null &&
            data.server_active_connections > 1
          ) {
            console.warn(
              `WARNING: The server currently has ${data.server_active_connections} active connections. Multiple concurrent connections can significantly degrade performance.`,
            );
          }

          break;
        default:
          console.warn('Unknown message type', type);
          break;
      }

      sendResponse({status: 'OK'});
    };

    console.log('Adding listener for runtime message from options tab');
    chrome.runtime.onMessage.addListener(onMessage);

    return () => {
      chrome.runtime.onMessage.removeListener(onMessage);
    };
  }, []);

  useLayoutEffect(() => {
    if (lastTranslationResultRef.current != null) {
      // Scroll the div to the most recent entry
      lastTranslationResultRef.current.scrollIntoView();
    }
    // Run the effect every time data is received, so that
    // we scroll to the bottom even if we're just adding text to
    // a pre-existing chunk
  }, [receivedData]);

  const startServerAsync = () => {
    return new Promise<void>((resolve) => {
      const wsConnOpenListener = (request) => {
        console.log('wsConnOpenListener received message', request);
        const {type} = request;
        if (type === 'WS_CONN_OPEN') {
          console.log('WS_CONN_OPEN received');
          setWsConnOpen(true);
          chrome.runtime.onMessage.removeListener(wsConnOpenListener);
          resolve();
        }
      };

      chrome.runtime.onMessage.addListener(wsConnOpenListener);
      sendRuntimeMessage({
        type: 'START_WS',
        data: {websocketServerUrl},
      });
    });
  };

  const configureStreamAsync = () => {
    return new Promise<void>((resolve) => {
      const serverReadyListener = (request) => {
        console.log('serverReadyListener received message', request);
        const {type} = request;
        if (type === 'SERVER_READY') {
          console.log('SERVER_READY received');
          setServerReady(true);
          setStreamingStatus('running');
          chrome.runtime.onMessage.removeListener(serverReadyListener);
          resolve();
        }
      };

      chrome.runtime.onMessage.addListener(serverReadyListener);
      sendRuntimeMessage({
        type: 'START_STREAM',
        data: {inputLang, outputLang: targetLang},
      });
    });
  };

  const startStreaming = async () => {
    if (streamingStatus !== 'stopped') {
      console.warn(
        `Attempting to start stream when status is ${streamingStatus}`,
      );
      return;
    }

    setStreamingStatus('starting');

    console.log('Awaiting WS_CONN_OPEN');
    await startServerAsync();
    console.log('startServerAsync has returned');

    await configureStreamAsync();
    console.log('configureStreamAsync has returned');
  };

  const stopStreaming = async () => {
    if (streamingStatus === 'stopped') {
      console.warn(
        `Attempting to stop stream when status is ${streamingStatus}`,
      );
      return;
    }

    const response = await sendRuntimeMessage({type: 'STOP_STREAM'});

    console.log('STOP_STREAM response', response);

    setStreamingStatus('stopped');
  };

  return (
    <div className="modal-wrapper">
      <div className="main-container">
        <div className="top-section horizontal-padding">
          <div className="header-container">
            <img
              src={chrome.runtime.getURL(SeamlessLogoPath)}
              className="header-icon"
              alt="Seamless Translation Logo"
            />

            <div>
              <Typography variant="h1" sx={{color: '#65676B'}}>
                Seamless Translation
              </Typography>
            </div>
          </div>
          <Stack spacing="22px" direction="column">
            <Stack spacing="12px" direction="row">
              <FormControl fullWidth sx={{minWidth: '14em'}}>
                <InputLabel id="source-selector-input-label">Source</InputLabel>
                <Select
                  labelId="source-selector-input-label"
                  defaultValue={INPUT_LANGUAGES['Spanish (Spain)']}
                  label="Source"
                  onChange={(e: SelectChangeEvent) =>
                    setInputLang(e.target.value)
                  }>
                  {Object.entries(INPUT_LANGUAGES).map(([label, value]) => (
                    <MenuItem value={value} key={value}>
                      {label}
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>
              <FormControl fullWidth sx={{minWidth: '14em'}}>
                <InputLabel id="target-selector-input-label">
                  Translation
                </InputLabel>
                <Select
                  labelId="target-selector-input-label"
                  defaultValue={OUTPUT_LANGUAGES['English (US)']}
                  label="Translation"
                  onChange={(e: SelectChangeEvent) =>
                    setTargetLang(e.target.value)
                  }>
                  {Object.entries(OUTPUT_LANGUAGES).map(([label, value]) => (
                    <MenuItem value={value} key={value}>
                      {label}
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>
            </Stack>
            <div>
              {streamingStatus === 'stopped' ? (
                <Button variant="contained" onClick={startStreaming}>
                  {buttonLabelMap[streamingStatus]}
                </Button>
              ) : (
                <Button variant="contained" onClick={stopStreaming}>
                  {buttonLabelMap[streamingStatus]}
                </Button>
              )}
            </div>
            <Accordion>
              <AccordionSummary
                expandIcon={<ExpandMoreIcon />}
                id="options-accordion-header">
                <Typography>Options</Typography>
              </AccordionSummary>
              <AccordionDetails>
                <TextField
                  fullWidth
                  disabled={streamingStatus !== 'stopped'}
                  label="Websocket Server URL"
                  variant="outlined"
                  value={websocketServerUrl}
                  onChange={(e) => setWebsocketServerUrl(e.target.value)}
                />
              </AccordionDetails>
            </Accordion>
          </Stack>
        </div>

        <div className="translation-text-container horizontal-padding">
          <div className="translation-container-header">
            <Typography variant="h1" sx={{fontWeight: 700}}>
              Transcript
            </Typography>
          </div>
          <div className="translation-text">
            {translationSentences.map((sentence, index) => {
              const maybeRef =
                index === translationSentences.length - 1
                  ? {ref: lastTranslationResultRef}
                  : {};
              return (
                <div className="text-chunk" key={index} {...maybeRef}>
                  <Typography variant="body1">{sentence}</Typography>
                </div>
              );
            })}
          </div>
        </div>
      </div>
    </div>
  );
}
