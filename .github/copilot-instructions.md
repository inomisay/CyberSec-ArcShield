# Copilot Workspace Instructions for CyberSec Prompt Engineering Benchmark Suite

## Purpose
This file provides workspace-specific instructions and conventions for GitHub Copilot and AI agents working in this repository. It is designed to maximize productivity, ensure safe automation, and avoid common pitfalls in this security-focused, multi-language project.

---

## Key Principles
- **Link, don't embed:** Reference documentation in `docs/` and code in `src/` rather than duplicating content in this file.
- **Respect modular boundaries:** Do not mix logic between `src/core/`, `src/benchmarks/`, `clients/dashboard/`, and `clients/extension/`.
- **Test before commit:** All code changes must pass the test suite (`pytest` for Python, `npm run build` for dashboard) before merging.
- **Environment variables:** Sensitive API keys (e.g., `GEMINI_API_KEY`) must be loaded from `.env` and never hardcoded.
- **Frontend/backend separation:** Treat `clients/dashboard/` as a standalone Vite+TypeScript app. Do not import backend code into frontend.

---

## Build & Test Commands
- **Python backend:**
  - Install: `pip install -r requirements.txt`
  - Run API: `uvicorn src.api.server:app --reload`
  - Test: `pytest src/tests/`
- **Frontend dashboard:**
  - Install: `cd clients/dashboard && npm install`
  - Dev server: `npm run dev`
  - Build: `npm run build`
- **Chrome extension:**
  - Load unpacked from `clients/extension/` in Chrome extensions page
  - Backend must be running for extension to function

---

## Documentation
- See `README.md` for project overview and execution guide
- See `docs/CLASSIFICATION_REFERENCE.md` for classification label meanings
- See `docs/TEST_STRUCTURE_GUIDE.md` and `docs/TEST_SCENARIOS.md` for test structure and scenarios

---

## Common Pitfalls
- **API server import path:** Always use `src.api.server:app` for Uvicorn, not `src/server.py`
- **.env required for cloud models:** Ensure `.env` is present and loaded for Gemini/Groq providers
- **Test data location:** All datasets must be under `dataset/` and referenced by relative path
- **Frontend/backend port mismatch:** Dashboard runs on 5173, backend on 8000 by default

---

## Example Prompts
- "Run all benchmarks for Ollama and Gemini."
- "Add a new dataset to the pipeline."
- "Summarize test coverage gaps."
- "Update dashboard UI to show new metric."

---

## For Agent Customization
- Use `applyTo` patterns for instructions targeting only `clients/dashboard/`, `src/`, or `clients/extension/` as needed.
- For new agent hooks or skills, see `README.md` and reference the modular structure above.
