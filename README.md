# ArcShield: CyberSec Prompt Engineering Benchmark Suite

[![GitHub Repo](https://img.shields.io/badge/GitHub-inomisay/CyberSec--ArcShield-blue?logo=github)](https://github.com/inomisay/CyberSec-ArcShield)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**ArcShield** is a state-of-the-art research and evaluation suite designed to analyze LLM robustness against **Adversarial Prompt Engineering**, **Jailbreaks**, and **Prompt Injections**. It features a high-performance evaluation engine, a futuristic researcher dashboard, and advanced strategic visualizations for security hardening.

---

## 🛡️ Key Features
- **Massive Benchmark Execution**: Run hundreds of attack vectors across multiple curated datasets.
- **Multi-Provider Integration**: Seamlessly compare local infrastructure (Ollama) with global cloud providers (OpenAI, Gemini, Groq, Mistral, HF).
- **Strategic Visualization Pipeline**: Generate publication-ready diagrams including Pareto Efficiency, Defense Dumbbells, and Vulnerability Heatmaps.
- **Chrome Security Extension**: Active monitoring and intercept testing directly in the browser.
- **Futuristic HUD Dashboard**: Real-time monitoring of attack trends and model resilience.

## 🤖 Models & Providers Supported
The suite is provider-agnostic, with command templates for the following evaluated models:
- **OpenAI**: `gpt-5-mini`.
- **Google Gemini**: `gemini-flash-lite-latest`, `gemma`.
- **Groq**: `llama-3.1-8b-instant`.
- **Mistral AI**: `mistral-small-latest` (currently resolving to `mistral-small-2603`).
- **Cloudflare Workers AI**: `@cf/qwen/qwen3-30b-a3b-fp8`, `@cf/deepseek-ai/deepseek-r1-distill-qwen-32b`, `@cf/google/gemma-3-12b-it`.
- **Local (Ollama)**: `llama3.1:8b`, `qwen3:latest`, `deepseek-r1:latest`, `mistral:latest`, `gemma3:latest`.

## 📊 Strategic Analysis Tools
ArcShield goes beyond simple ASR (Attack Success Rate) metrics to provide strategic security insights:

1. **Strategic Pareto Efficiency**: Analyze the trade-off between **Security Breach Rate** and **Inference Latency**. Identify models that offer the "Strategic Optimum."
2. **Security Hardening (ASR Reduction)**: Visualize the effectiveness of system-level defenses using dumbbell plots that show the delta between 'Raw' and 'Hardened' model configurations.
3. **Tactical Vulnerability Heatmaps**: Heat-mapped visualization of residual vulnerabilities categorized by attack source and strategy.
4. **Radar Profiling**: Holistic 4-axis profiling of model capabilities across Security, Hardening, Helpfulness, and Resilience.

---

## 🚀 Quick Start Guide

### 1. Installation
```bash
# Clone the repository
git clone https://github.com/inomisay/CyberSec-ArcShield.git
cd CyberSec-ArcShield

# Install dependencies
pip install -r requirements.txt
```

### 2. Running Benchmarks
Edit the `.env` file with your API keys, then run:
```bash
# Run a sample benchmark on Gemini
python src/benchmarks/benchmark_multi.py --provider gemini --model gemini-2.5-flash-lite --sample 10

# Run on OpenAI
python src/benchmarks/benchmark_multi.py --provider openai --model gpt-5-mini --sample 5
```

### 3. Generating Strategic Visualizations
Once results are collected in the `output/` directory:
```bash
python logs/diagrams.py
```
This will generate high-resolution PNGs in `output/provider_comparison_plots/`.

### 4. Researcher Dashboard
```bash
# Start API Server
uvicorn src.api.server:app --reload

# Start Frontend (in separate terminal)
cd clients/dashboard
npm install && npm run dev
```

### 5. Browser Extension
Load the unpacked extension from `clients/extension` in Chrome's extensions page after starting the API server.

---

## 📁 Project Structure
- `src/core/`: The "Brain" containing judicial evaluation and defender logic.
- `src/benchmarks/`: Core execution engines for multi-dataset evaluation.
- `output/`: Diagnostic output and the `diagrams.py` visualization suite.
- `clients/dashboard/`: React-based visual analytics platform.
- `clients/extension/`: Chrome extension for real-time prompt analysis.
- `dataset/`: Integrated security datasets (Prompt Injection, Jailbreaks, etc).
- `src/pipeline/prep/`: Data preparation, sampling, and benchmark support scripts.
- `src/pipeline/analysis/`: Coverage and provenance analysis scripts.
- `src/pipeline/models/`: Model inventory and hosting metadata scripts.

## 🎓 Academic Credits & Research
This project implements metrics and evaluation methodologies inspired by:
- **Liu et al. (2024)**: On the robustness of large language models to adversarial prompt engineering.
- **Radcliffe et al. (2024)**: Measuring impact reduction in system-level instruction defenses.

---
© 2026 ArcShield Security Research Group. Licensed under MIT.
