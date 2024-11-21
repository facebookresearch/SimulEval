import reloadOnUpdate from 'virtual:reload-on-update-in-background-script';

reloadOnUpdate('pages/background');

/**
 * Extension reloading is necessary because the browser automatically caches the css.
 * If you do not use the css of the content script, please delete it.
 */
reloadOnUpdate('pages/content/style.scss');

console.log('background loaded');

function openOptions(): Promise<chrome.tabs.Tab> {
  return new Promise((resolve) => {
    chrome.tabs.create(
      {
        pinned: true,
        active: false, // <--- Important
        url: `chrome-extension://${chrome.runtime.id}/src/pages/options/index.html`,
      },
      (tab) => {
        resolve(tab);
      },
    );
  });
}

function removeTab(tabId) {
  return new Promise((resolve) => {
    chrome.tabs.remove(tabId).then(resolve).catch(resolve);
  });
}

function executeScript(tabId, file): Promise<void> {
  return new Promise((resolve) => {
    chrome.scripting.executeScript(
      {
        target: {tabId},
        files: [file],
      },
      () => {
        resolve();
      },
    );
  });
}

function insertCSS(tabId, file): Promise<void> {
  return new Promise((resolve) => {
    chrome.scripting.insertCSS(
      {
        target: {tabId},
        files: [file],
      },
      () => {
        resolve();
      },
    );
  });
}

function sendMessageToTab(tabId, data) {
  return new Promise((resolve) => {
    chrome.tabs.sendMessage(tabId, data, (res) => {
      resolve(res);
    });
  });
}

function sleep(ms = 0) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function getStorage(key) {
  return new Promise((resolve) => {
    chrome.storage.local.get([key], (result) => {
      resolve(result[key]);
    });
  });
}

function setStorage(key, value) {
  return new Promise((resolve) => {
    chrome.storage.local.set(
      {
        [key]: value,
      },
      () => {
        resolve(value);
      },
    );
  });
}

chrome.action.onClicked.addListener(async (currentTab) => {
  console.log('From background/index.ts: Chrome extension clicked!');
  console.log('Current Tab', currentTab);

  // Is it record only one tab at the same time?
  const optionTabId = await getStorage('optionTabId');
  if (optionTabId) {
    await removeTab(optionTabId);
  }

  // You can save the current tab id to cache
  await setStorage('currentTabId', currentTab.id);

  // Inject the actual content react app into the page
  // await sendMessageToTab(currentTab.id, {type: 'INJECT_PAGE_CONTENT'});
  // await executeScript(currentTab.id, 'content.js');
  // await insertCSS(currentTab.id, 'content.css');
  await chrome.scripting.executeScript({
    target: {tabId: currentTab.id},
    files: ['src/pages/content/index.js'],
  });
  console.log('Injected content script into current tab');

  await sleep(500);

  // Open the option tab
  const optionTab = await openOptions();
  console.log('Option tab', optionTab);

  // You can save the option tab id to cache
  await setStorage('optionTabId', optionTab.id);

  await sleep(500);

  // You can pass some data to option tab
  // await sendMessageToTab(optionTab.id, {
  //   type: "START_RECORD",
  //   data: { currentTabId: currentTab.id },
  // });

  // Start the websocket connection
  // await sendMessageToTab(optionTab.id, {
  //   type: 'START_WS',
  // });
});

chrome.tabs.onRemoved.addListener(async (tabId) => {
  const currentTabId = await getStorage('currentTabId');
  const optionTabId = await getStorage('optionTabId');

  // When the current tab is closed, the option tab is also closed by the way
  if (currentTabId === tabId && optionTabId) {
    await removeTab(optionTabId);
  }
});
