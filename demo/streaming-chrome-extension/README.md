# Seamless Translation Streaming Chrome Extension

## Prerequisites

This extension is meant to be developed locally on your macbook.

_NOTE: While it is theoretically possible to develop it remotely (ie on a dev server) and mount the `/dist/` folder over SSH so that your local browser can load/reload the extension, we do not currently have instructions for this and some functionality like the live-reload feature won't work without additional configuration to forward ports to your local computer._

The following are prerequisites for your dev environment

1. Node.js environment: Follow the instructions in the "Getting Started" section of the main monorepo [README.md](/README.md) to set up Node.js
2. Other project-wide prerequisites: Follow any _other_ instructions in that same "Getting Started" section [README.md](/README.md) to set up any project-wide tooling/dependencies
3. Yarn package manager: see [installation instructions](https://yarnpkg.com/getting-started/install)

## Getting started

1. Run `yarn` to install the relevant packages
2. Run `yarn run dev` to start a dev server that watches for changes and tries to live update the file in your browser. Note that this does not always work perfectly, and sometimes you will have to refresh the page/extension
3. Load the extension within Chrome
   1. Within Chrome go to `chrome://extensions`
   2. Ensure "Developer mode" is enabled
   3. Click "Load unpacked"
   4. Select the `/dist/` folder in this project (NOTE: `/dist/` is populated when you run `yarn run dev` and/or `yarn run build`. You must run one of the two for the `/dist/` folder to contain an up-to-date build)

There are three main code entry points, and three places you'll want to look for console logs, stack traces, etc.

1. **The Background JS** - This is a bit of javascript that handles the overall extension state and triggers the content js to inject the react app into the page.
2. **The Content JS** - This code renders the UI on top of the web page where the chrome extension was clicked. It receives messages from the background worker and the options tab.
3. **The Options Tab** - In the current implementation this is where the audio is actually captured. I believe there was a reason for this workaround (as opposed to capturing it directly in the background js), but I'm not clear if it's still relevant.

## Building the extension for production

1. Run `yarn` to install the relevant packages
2. Run `yarn build`, which will build the extension in the `/dist/` folder.
3. Follow the instructions above to load the extension within Chrome.

## Using the extension

While this extension is in development, there are a few important things to do to make sure it works well while you're using it.

**TL;DR:** _Your video MUST be stopped before you start using the extension. If something isn't working, hard refresh the page (`cmd+shift+R` on mac) and try again._

NOTE: As of 2023-05-04 our streaming server does not support multiple simultaneous connections, so to the best of your ability you'll want to make sure others are not using it. If you see poor performance, this could be the reason. We aim to address this soon.

1. Ensure your video is stopped. **Your video MUST be stopped before proceeding to the next steps.** If your video is playing you may lose audio, and the extension won't work.
2. After loading the extension using the instructions above ('Load the extension within Chrome'), ensure that it's enabled. You can try it on any website, but it's been tested on YouTube
3. Once you click the chrome extension icon you should see a panel appear in the upper right corner of the screen that says "Seamless Translation". **If you don't see it, or if it looks messed up, hard refresh the webpage (`cmd+shift+r` on mac) and try again.**
4. Select your source and target languages (we currently support spanish->english, and possibly others) and then press "Start Streaming". This starts sending your tab's audio to our streaming server.
5. WAIT 3 seconds before proceeding to the next step
6. **Now press play on your video.**
   1. Reminder: You MUST start the streaming before you start the video playing
   2. After a few seconds you should see the text translation come in. It may take longer depending on the audio and how congested the server is. If you don't see anything after 20 seconds, try hard refreshing the page. If it still doesn't work, open the chrome developer console ([instructions](https://developer.chrome.com/docs/devtools/open/#shortcuts)) and look for any relevant messages. Share the issue with the engineering team. Screenshots of the browser window and console are very useful!

A few other notes:

- The extension does not currently support navigating to other pages while it's open. If you go to a new video on YouTube the extension will likely stay open with the previous translation text. To solve this, hard refresh the page before navigating to a new one.
- If you are not hearing audio after using the extension, look for the tab that was created when you opened the extension -- usually it's directly adjacent to your current tab. Close that, refresh your page, and then try again.
- Make sure the volume of your video is all the way up (ie the volume slider on the video itself, not your computer's audio volume).
