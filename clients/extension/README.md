# ArcShield Chrome Extension

This is the Chrome extension used to trigger and observe ArcShield benchmark actions from the browser.

## How it works

- The extension is a popup-based Chrome extension.
- It communicates with the local ArcShield API server.
- It checks whether the backend is online, then sends browser-side actions to the server.
- It is loaded unpacked in Chrome during development.

## Prerequisites

- The Python backend must be running locally
- Chrome or another Chromium-based browser
- Developer mode enabled in the browser extension page

## Run It

Start the backend from the project root:

```bash
uvicorn src.api.server:app --reload
```

Then load the extension in Chrome:

1. Open `chrome://extensions/`.
2. Enable Developer mode.
3. Click Load unpacked.
4. Select the `clients/extension` folder.

## Build or Package

This extension is a plain manifest-based extension, so there is no separate build step in this repository. Chrome loads the files directly from the folder.

## Main Files

- `manifest.json`: extension metadata and permissions
- `popup.html`: popup UI shell
- `popup.js`: extension behavior and API calls
- `icon.png`: toolbar and popup icon
