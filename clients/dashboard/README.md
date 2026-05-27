# ArcShield Dashboard

This is the Vite + TypeScript dashboard for ArcShield. It provides the researcher HUD for launching benchmarks, monitoring progress, and reviewing summary results after a run finishes.

## How it works

- The dashboard is a standalone frontend app.
- It talks to the Python API in `src/api/server.py`.
- When a benchmark starts, the UI streams progress and displays results as they arrive.
- After the benchmark ends, the summary section becomes visible so you can inspect the final metrics.

## Prerequisites

- Python backend dependencies installed with `pip install -r requirements.txt`
- Node.js installed for the frontend
- A `.env` file if you use cloud models such as Gemini

## Run It

From the project root, start the backend:

```bash
uvicorn src.api.server:app --reload
```

In a second terminal, start the dashboard:

```bash
cd clients/dashboard
npm install
npm run dev
```

Open the local URL shown by Vite, usually `http://localhost:5173`.

## Build It

```bash
cd clients/dashboard
npm run build
```

## Main Files

- `src/main.ts`: app bootstrap and UI behavior
- `src/style.css`: dashboard styling
- `index.html`: application shell
