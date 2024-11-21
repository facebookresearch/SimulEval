console.log('content index.ts loaded');

/**
 * @description
 * Chrome extensions don't support modules in content scripts.
 */
// import('./components/Demo');

console.log('Importing Demo component...');
import('./app');

// chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
//   console.log('From content/index.ts: Message received:', {
//     request,
//     sender,
//     sendResponse,
//   });
//   const {type, data} = request;

//   switch (type) {
//     case 'INJECT_PAGE_CONTENT':
//       console.log(request);
//       // console.log('Setting secondary div to display none');
//       // document
//       //   .getElementById('secondary')
//       //   .setAttribute('style', 'display: none;');
//       console.log('Importing Demo component...');
//       import('./app');

//       break;
//     default:
//       break;
//   }

//   // sendResponse({status: 'Confirmed'});
// });

// function sendRuntimeMessage(message) {
//   return new Promise((resolve) => {
//     chrome.runtime.sendMessage(message, (res) => {
//       resolve(res);
//     });
//   });
// }

// function getStorage(key) {
//   return new Promise((resolve) => {
//     chrome.storage.local.get([key], (result) => {
//       resolve(result[key]);
//     });
//   });
// }

// let newParagraph = true;
// let mainDiv = document.getElementById('secondary');
// const inject_url = chrome.runtime.getURL('src/pages/content/inject.html');
// let streamingStarted = false;
// fetch(inject_url)
//   .then((resp) => {
//     return resp.text();
//   })
//   .then((txt) => {
//     mainDiv.innerHTML = txt;
//     let startButton = document.getElementById('start_streaming_button');
//     startButton.onclick = async () => {
//       if (!streamingStarted) {
//         startButton.value = '...';
//         const inputLang = document.getElementById('input_lang').value;
//         const outputLang = document.getElementById('output_lang').value;
//         const bufferVal = document.getElementById('buffer_slider').value;
//         const optionTabId = await getStorage('optionTabId');
//         const currentTabId = await getStorage('currentTabId');
//         const res = await sendRuntimeMessage({
//           type: 'START_STREAM',
//           data: {currentTabId, inputLang, outputLang, bufferVal},
//         });
//         console.log('res is', res);
//         streamingStarted = true;
//         startButton.value = 'STOP';
//         startButton.innerHTML = 'STOP';
//       } else {
//         const res = await sendRuntimeMessage({
//           type: 'STOP_STREAM',
//         });
//         console.log('res is', res);
//         streamingStarted = false;
//         startButton.value = 'START';
//         startButton.innerHTML = 'START';
//       }
//     };
//   })
//   .catch((err) => console.log('err is', err));

// let voices, enVoice, esVoice;
// window.speechSynthesis.addEventListener('voiceschanged', () => {
//   voices = window.speechSynthesis.getVoices();
//   enVoice = voices.find((v) => {
//     return v.lang === 'en-US' && v.name.startsWith('Google');
//   });
//   esVoice = voices.find((v) => {
//     return v.lang === 'es-ES' && v.name.startsWith('Google');
//   });
// });

// const ttsSpeakOut = (txt, language) => {
//   var msg = new SpeechSynthesisUtterance();
//   msg.text = txt;
//   if (language == 'en-US') {
//     msg.voice = enVoice;
//   } else if (language == 'es-ES') {
//     msg.voice = esVoice;
//   }

//   msg.lang = language;
//   window.speechSynthesis.speak(msg);
// };

// const capitalize = (s) => (s && s[0].toUpperCase() + s.slice(1)) || '';

// // Receive data from Option Tab or Background
// chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
//   const {type, data} = request;

//   switch (type) {
//     case 'FROM_OPTION_TRANSLATION_TEXT':
//       console.log(data);
//       if (data.payload) {
//         let translation_feed = document.getElementById('translation_feed');
//         let txt = newParagraph
//           ? `\n\n${capitalize(data.payload)}`
//           : ` ${data.payload}`;
//         if (data.eos) {
//           txt = txt + '.';
//           newParagraph = true;
//         } else {
//           newParagraph = false;
//         }
//         translation_feed.appendChild(document.createTextNode(txt));
//         translation_feed.scrollTop = translation_feed.scrollHeight;
//         if (document.getElementById('speech').checked) {
//           ttsSpeakOut(data.payload, 'en-US');
//         }
//       }

//       break;
//     default:
//       break;
//   }

//   sendResponse({});
// });
