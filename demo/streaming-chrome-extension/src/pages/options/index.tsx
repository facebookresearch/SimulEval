import React from 'react';
import {createRoot} from 'react-dom/client';
import Options from '@pages/options/Options';
import '@pages/options/index.css';
import refreshOnUpdate from 'virtual:reload-on-update-in-view';
import {StartStreamingData} from '@src/types/StreamingTypes';

refreshOnUpdate('pages/options');

console.log('options/index.tsx loaded');

const BUFFER_LIMIT = 1;

// function init() {
//   const appContainer = document.querySelector('#app-container');
//   if (!appContainer) {
//     throw new Error('Can not find #app-container');
//   }
//   const root = createRoot(appContainer);
//   root.render(<Options />);
// }

// init();

/**
 * From options.js
 */

function tabCapture(): Promise<MediaStream> {
  return new Promise((resolve) => {
    chrome.tabCapture.capture(
      {
        audio: true,
        video: false,
      },
      (stream) => {
        resolve(stream);
      },
    );
  });
}

let globalActiveTabId = null;

// function to16BitPCM(input) {
//   const dataLength = input.length * (16 / 8);
//   const dataBuffer = new ArrayBuffer(dataLength);
//   const dataView = new DataView(dataBuffer);
//   let offset = 0;
//   for (let i = 0; i < input.length; i++, offset += 2) {
//     const s = Math.max(-1, Math.min(1, input[i]));
//     dataView.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7fff, true);
//   }
//   return dataView;
// }

// function to16kHz(audioData, sampleRate = 44100) {
//   const data = new Float32Array(audioData);
//   const fitCount = Math.round(data.length * (16000 / sampleRate));
//   const newData = new Float32Array(fitCount);
//   const springFactor = (data.length - 1) / (fitCount - 1);
//   newData[0] = data[0];
//   for (let i = 1; i < fitCount - 1; i++) {
//     const tmp = i * springFactor;
//     const before = Math.floor(tmp).toFixed();
//     const after = Math.ceil(tmp).toFixed();
//     const atPoint = tmp - before;
//     newData[i] = data[before] + (data[after] - data[before]) * atPoint;
//   }
//   newData[fitCount - 1] = data[data.length - 1];
//   return newData;
// }

function sendMessageToTab(tabId, data) {
  return new Promise((resolve) => {
    chrome.tabs.sendMessage(tabId, data, (res) => {
      resolve(res);
    });
  });
}

const initWsClient = async (
  url,
  onOpenCallback,
  onCloseCallbck,
  onErrorCallback,
  onMessageCallback,
) => {
  return new Promise((resolve, reject) => {
    const wsClient = new WebSocket(url);
    wsClient.onopen = () => {
      resolve(wsClient);
      if (onOpenCallback) {
        onOpenCallback();
      }
    };
    wsClient.onclose = () => {
      if (onCloseCallbck) {
        onCloseCallbck();
      }
    };
    wsClient.onerror = (error) => {
      if (onErrorCallback) {
        onErrorCallback(error);
      }
      reject(error);
    };
    wsClient.onmessage = (e) => {
      if (onMessageCallback) {
        onMessageCallback(e);
      }
    };
  });
};

const WS_HOST_DEFAULT = 'wss://seamless-api.dev.metademolab.com';
// const WS_HOST_DEFAULT = 'ws://52.32.188.142:8000';

const onWsOpen = () => {
  console.log('WS conn open');
  sendMessageToTab(globalActiveTabId, {type: 'WS_CONN_OPEN'});
};

const onWsClose = () => {
  console.log('WS conn closed');
};

const onWsError = (err) => {
  console.log(`WS conn error: ${JSON.stringify(err)}`);
};

const onWsMessage = (e) => {
  const message = JSON.parse(e.data);
  console.log('WS message', message);

  if (message.event === 'translation_text') {
    // You can pass some data to current tab
    console.log('currentTabId is', globalActiveTabId);
    sendMessageToTab(globalActiveTabId, {
      type: 'FROM_OPTION_TRANSLATION_TEXT',
      data: message,
    });
    // if (message.latency) {
    //   todo: display latency
    // }
  } else if (message.event === 'server_ready') {
    // todo: change the button color
    console.log('received server_ready');
    sendMessageToTab(globalActiveTabId, {
      type: 'SERVER_READY',
    });
  } else {
    console.log(`unsupported incoming event ${message.event}`);
  }
};

const float32To16BitPCM = (float32Arr) => {
  var pcm16bit = new Int16Array(float32Arr.length);
  for (var i = 0; i < float32Arr.length; ++i) {
    // force number in [-1,1]
    var s = Math.max(-1, Math.min(1, float32Arr[i]));

    /**
     * convert 32 bit float to 16 bit int pcm audio
     * 0x8000 = minimum int16 value, 0x7fff = maximum int16 value
     */
    pcm16bit[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
  }
  return pcm16bit;
};

const ttsSpeakOut = (txt, voice, language) => {
  var msg = new SpeechSynthesisUtterance();
  msg.text = txt;
  const voices = window.speechSynthesis.getVoices();
  const filteredVoice = voices.find((v) => v.name === voice);
  if (filteredVoice) {
    msg.voice = filteredVoice;
  }
  msg.lang = language;
  window.speechSynthesis.speak(msg);
};

let wsSocket;
let shouldStream = false;
let context;
let streamForTabCapture;
async function startWSConnection(websocketServerUrlFromClient) {
  const wsHostToUse =
    websocketServerUrlFromClient != null &&
    websocketServerUrlFromClient.trim().length > 0
      ? websocketServerUrlFromClient.trim()
      : WS_HOST_DEFAULT;

  if (wsSocket != null && wsSocket.url === wsHostToUse) {
    console.log(`websocket connection to ${wsHostToUse} is already open.`);
    sendMessageToTab(globalActiveTabId, {type: 'WS_CONN_OPEN'});
    return;
  }

  /**
   * Set up tab capture and audio context.
   * If we already have a stream open, reuse it.
   * Calling await tabCapture() when capture is already active will cause an error.
   *
   * NOTE: This must be done before starting the websocket connection so that
   * context.sampleRate is set correctly.
   */
  if (streamForTabCapture == null) {
    streamForTabCapture = await tabCapture();

    if (streamForTabCapture == null) {
      console.log('stream is null. this is bad!');
      window.close();
      return;
    }

    console.log('creating new audio context');
    context = new AudioContext();
    const mediaStream = context.createMediaStreamSource(streamForTabCapture);
    const recorder = context.createScriptProcessor(16384, 1, 1);

    recorder.onaudioprocess = async (event) => {
      if (!context) return;
      if (!shouldStream) return;
      const float32Audio = event.inputBuffer.getChannelData(0);
      const pcm16Audio = float32To16BitPCM(float32Audio);
      if (wsSocket != null) {
        wsSocket.send(pcm16Audio);
      } else {
        console.warn('wsSocket is null in onaudioprocess');
      }
    };

    mediaStream.connect(recorder);
    recorder.connect(context.destination);
    mediaStream.connect(context.destination);
  }

  // If we already have a websocket connection open, close it
  if (wsSocket != null) {
    wsSocket.close();
  }

  console.log('starting WS connection');
  wsSocket = await initWsClient(
    `${wsHostToUse}/api/seamless_stream_es_en_s2t`,
    onWsOpen,
    onWsClose,
    onWsError,
    onWsMessage,
  );
}

function stopStream() {
  shouldStream = false;
}

async function startStream(option: StartStreamingData, tabId: number) {
  console.log('startRecord');
  globalActiveTabId = tabId;
  console.log('wsSocket', wsSocket);
  if (!wsSocket) {
    console.log('wsSocket is NOT ready');
    return;
  }
  await wsSocket.send(
    JSON.stringify({
      event: 'config',
      rate: context.sampleRate,
      target_language: option.outputLang,
      source_language: option.inputLang,
      // TODO: don't hardcode this
      debug: false,
      // TODO: don't hardcode this
      async_processing: true,
      // Hardcode for
      buffer_limit: BUFFER_LIMIT,
      model_type: 's2t',
    }),
  );
  shouldStream = true;
}

// Receive data from Current Tab or Background
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  const {type, data} = request;

  switch (type) {
    case 'START_STREAM':
      startStream(data, sender.tab.id);
      sendResponse({text: 'START_STREAM received'});
      break;
    case 'STOP_STREAM':
      stopStream();
      sendResponse({text: 'Stream stopped', status: true});
      break;
    case 'START_WS':
      globalActiveTabId = sender.tab.id;
      startWSConnection(data?.websocketServerUrl);
      sendResponse({});
      break;
    default:
      break;
  }

  return true;
});
