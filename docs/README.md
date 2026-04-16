# CyberSec Prompt Engineering Benchmark Suite

This project is a comprehensive research and evaluation suite designed to test LLM (Large Language Model) robustness against **Adversarial Prompt Engineering**, **Jailbreaks**, and **Prompt Injections**. It features a local backend, a futuristic researcher dashboard, and a browser extension for active monitoring.

## 🛡️ Project Summary
The suite enables security researchers to:
- Run massive benchmarks across multiple datasets.
- Compare local models (Ollama) against cloud-based models (Gemini).
- Measure **Attack Success Rate (ASR)** and **Defense Impact** using research-grade metrics.
- Evaluate the effectiveness of **System Instruction Defenses** in real-time.

## 🤖 Models Supported
The system is designed to be model-agnostic and currently supports:
- **Local (Ollama)**: Llama 3.2 (Default).
- **Google Gemini (AI Studio)**: 
    - Gemini 2.0 Flash (Latest/Fastest)
    - Gemini 1.5 Flash
    - Gemini 1.5 Pro

## 📊 Datasets Integrated
The benchmark utilizes several academic and community-curated datasets:
1. **Prompt Injection Malignant**: Focused on harmful adversarial injections designed to bypass core safety filters.
2. **LLM Jailbreak + Safety Data**: A broad dataset for chatbot safety, covering multiple techniques like roleplay and hypothetical scenarios.
3. **Kaggle Dataset**: General purpose LLM security datasets for broad evaluation.
4. **Prompt Fruit Injection**: Specialized datasets analyzing fruit-themed injection analysis patterns.

## 🚀 Execution Guide

### 1. Terminal Evaluation (Core Engine)
To run a raw benchmark in your terminal with academic metrics:
```bash
# Basic run (Llama 3.2 via Ollama)
python src/benchmarks/benchmark_multi.py --sample 5

# Run with specific Gemini model
python src/benchmarks/benchmark_multi.py --sample 10 --provider gemini --model gemini-2.0-flash
```

### 2. Cybersecurity Dashboard (Modern HUD)
The dashboard provides a visual "HUD" for real-time monitoring and aggregate reporting.

**Backend Setup:**
```bash
# Ensure dependencies are installed
pip install -r requirements.txt

# Start the API Server
uvicorn src.api.server:app --reload
```

**Frontend Setup:**
```bash
cd dashboard
npm install
npm run dev
```
Open `http://localhost:5173` to access the dashboard. 
*Note: Ensure `GEMINI_FLASH_LITE_API_KEY` or `GEMINI_API_KEY` is set in your `.env` file for cloud models.*

### 3. Browser Extension (Security Assistant)
The extension allows you to test prompts directly within your browser.
1. Open Chrome and go to `chrome://extensions/`.
2. Enable **Developer Mode**.
3. Click **Load Unpacked**.
4. Select the `extension` folder from this project.
5. Ensure the Backend (Step 2) is running for the extension to communicate.

## 📁 Project Structure
- `src/core/`: The "Brain" - Models, Defender, and Judicial evaluation logic.
- `src/benchmarks/`: Multi-dataset execution engines.
- `dashboard/`: Vite + TypeScript frontend.
- `extension/`: Chrome extension files.
- `data/`: Integrated datasets.

---
**Academic Note**: This project utilizes evaluation principles from Liu et al. (2024) and Radcliffe et al. (2024) regarding adversarial complexity and robustness metrics.
