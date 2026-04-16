# Chrome Extension Benchmark Tool

This tool allows you to run the AI Safety Benchmark directly from your browser.

## Components
1.  **Backend Server (`src/server.py`)**: A local Python server that runs the benchmark logic.
2.  **Chrome Extension (`extension/`)**: A browser popup that controls the server.

## Setup & Running

### 1. Start the Backend Server
The extension needs the local server to be running to work.
Open a terminal in the project root and run:

```bash
uvicorn src.server:app --reload
```
or
```bash
python -m uvicorn src.server:app --reload
```

You should see: `Uvicorn running on http://127.0.0.1:8000`.

### 2. Load the Extension in Chrome
1.  Open Chrome and navigate to `chrome://extensions/`.
2.  Enable **Developer mode** (toggle in top right).
3.  Click **Load unpacked**.
4.  Select the `extension` folder inside this project (`.../PromptEngInCyberSec/extension`).
5.  The "CyberSec Prompt Benchmark" icon should appear in your toolbar.

### 3. Usage
1.  Click the extension icon.
2.  Enter the number of samples to test (e.g., 5).
3.  Click **Run Benchmark**.
4.  Watch the progress bar and results stream in!
