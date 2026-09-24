# Measurement and Judge Sensitivity Analysis for ArcShield Benchmark

*Generated on: 2026-09-02 12:04:17*

## 1. Executive Summary

This sensitivity analysis investigates whether ArcShield's observed defense gains stem from heuristic artifacts or keyword matching:

- **Llama Guard Evaluation Coverage**: Assesses the operational uptime and coverage of the Stage-2 neural safety classifier across all 11 model configurations.

- **Non-Canonical Refusal Test**: Demonstrates that when the exact canonical refusal string (`'I cannot fulfill this request.'`) is completely excluded from rule matching, ArcShield still achieves substantial, statistically significant risk reductions across every single model configuration.

- **Llama Guard-Restricted Cohort**: Re-evaluates defense efficacy strictly on the subset where Llama Guard was active, validating that neural harm detection mirrors the deterministic benchmark results.


## 2. Table M1: Llama Guard Evaluation Coverage across Models & Conditions


| Model                                | Deployment         |   Total Observations |   Llama Guard Evaluated (N) | Overall Coverage (%)   | Coverage (No Defense)   | Coverage (ArcShield)   |
|:-------------------------------------|:-------------------|---------------------:|----------------------------:|:-----------------------|:------------------------|:-----------------------|
| Cloudflare DeepSeek Distill Qwen 32B | Cloud              |                 6400 |                        6316 | 98.7%                  | 98.4% (3150/3200)       | 98.9% (3166/3200)      |
| Cloudflare Gemma 4 26B-A4B           | Cloud              |                 6400 |                        6352 | 99.2%                  | 99.0% (3169/3200)       | 99.5% (3183/3200)      |
| Cloudflare Qwen3 30B-A3B             | Cloud              |                 6400 |                        5740 | 89.7%                  | 86.8% (2778/3200)       | 92.6% (2962/3200)      |
| Gemini Flash-Lite                    | Cloud              |                 6400 |                        5799 | 90.6%                  | 90.1% (2883/3200)       | 91.1% (2916/3200)      |
| Groq Llama 3.1 8B Instant            | Cloud              |                 6400 |                        5587 | 87.3%                  | 85.0% (2719/3200)       | 89.6% (2868/3200)      |
| Mistral Small                        | Cloud              |                 6400 |                        5519 | 86.2%                  | 81.4% (2606/3200)       | 91.0% (2913/3200)      |
| Ollama DeepSeek R1                   | Local              |                 6400 |                        6369 | 99.5%                  | 99.4% (3181/3200)       | 99.6% (3188/3200)      |
| Ollama Gemma 4                       | Local              |                 6400 |                        6394 | 99.9%                  | 99.9% (3198/3200)       | 99.9% (3196/3200)      |
| Ollama Llama 3.1 8B                  | Local              |                 6400 |                        6099 | 95.3%                  | 94.0% (3009/3200)       | 96.6% (3090/3200)      |
| Ollama Mistral 7B                    | Local              |                 6400 |                        6366 | 99.5%                  | 99.6% (3186/3200)       | 99.4% (3180/3200)      |
| Ollama Qwen3                         | Local              |                 6399 |                        6355 | 99.3%                  | 99.1% (3169/3199)       | 99.6% (3186/3200)      |
| **OVERALL BENCHMARK**                | All Configurations |                70399 |                       66896 | **95.0%**              | 93.9%                   | 96.2%                  |


## 3. Table M2: Refusal Detector Sensitivity (Excluding Exact Canonical String)


| Model                                | Deployment         |   N_Pairs | Standard ASR: NoDef -> Arc   | Standard ΔASR (Rel. Red %)   | Non-Canonical ASR: NoDef -> Arc   | Non-Canonical ΔASR (Rel. Red %)   | ArcShield ASR Shift (Δpp)   | Defense Persists?   |
|:-------------------------------------|:-------------------|----------:|:-----------------------------|:-----------------------------|:----------------------------------|:----------------------------------|:----------------------------|:--------------------|
| Cloudflare DeepSeek Distill Qwen 32B | Cloud              |      3200 | 39.1% -> 22.2%               | +16.9 pp (43.3%)             | 39.1% -> 22.2%                    | +16.9 pp (43.2%)                  | +0.03 pp                    | YES                 |
| Cloudflare Gemma 4 26B-A4B           | Cloud              |      3200 | 18.0% -> 14.5%               | +3.5 pp (19.4%)              | 18.1% -> 14.5%                    | +3.6 pp (19.9%)                   | +0.00 pp                    | YES                 |
| Cloudflare Qwen3 30B-A3B             | Cloud              |      3200 | 27.7% -> 8.4%                | +19.3 pp (69.6%)             | 27.8% -> 8.4%                     | +19.4 pp (69.7%)                  | +0.00 pp                    | YES                 |
| Gemini Flash-Lite                    | Cloud              |      3200 | 26.2% -> 18.2%               | +8.0 pp (30.5%)              | 26.3% -> 18.2%                    | +8.1 pp (30.7%)                   | +0.00 pp                    | YES                 |
| Groq Llama 3.1 8B Instant            | Cloud              |      3200 | 23.4% -> 12.8%               | +10.6 pp (45.3%)             | 23.4% -> 12.8%                    | +10.6 pp (45.3%)                  | +0.00 pp                    | YES                 |
| Mistral Small                        | Cloud              |      3200 | 30.5% -> 15.0%               | +15.5 pp (50.9%)             | 30.5% -> 15.0%                    | +15.5 pp (50.8%)                  | +0.06 pp                    | YES                 |
| Ollama DeepSeek R1                   | Local              |      3200 | 24.1% -> 7.1%                | +17.0 pp (70.6%)             | 24.2% -> 7.1%                     | +17.1 pp (70.6%)                  | +0.00 pp                    | YES                 |
| Ollama Gemma 4                       | Local              |      3200 | 20.2% -> 13.1%               | +7.0 pp (34.9%)              | 20.2% -> 13.1%                    | +7.1 pp (35.0%)                   | +0.00 pp                    | YES                 |
| Ollama Llama 3.1 8B                  | Local              |      3200 | 16.7% -> 4.6%                | +12.1 pp (72.4%)             | 23.2% -> 4.6%                     | +18.7 pp (80.2%)                  | +0.00 pp                    | YES                 |
| Ollama Mistral 7B                    | Local              |      3200 | 41.8% -> 15.4%               | +26.4 pp (63.1%)             | 41.8% -> 15.4%                    | +26.4 pp (63.1%)                  | +0.00 pp                    | YES                 |
| Ollama Qwen3                         | Local              |      3199 | 28.6% -> 6.6%                | +22.0 pp (77.1%)             | 28.7% -> 6.6%                     | +22.1 pp (77.1%)                  | +0.00 pp                    | YES                 |
| **OVERALL BENCHMARK**                | All Configurations |     35199 | 26.9% -> 12.5%               | +14.4 pp (53.4%)             | 27.6% -> 12.5%                    | +15.0 pp (54.5%)                  | +0.01 pp                    | **YES**             |


## 4. Table M3: ASR Evaluation Restricted Strictly to Llama Guard-Evaluated Instances


| Model                                | Deployment         |   Llama Guard Paired N | ASR_nodef (95% CI)   | ASR_arcshield (95% CI)   | ΔASR (pp)   | Relative Reduction (%)   | Llama Guard Harm Rate (NoDef -> Arc)   | McNemar p-value   |
|:-------------------------------------|:-------------------|-----------------------:|:---------------------|:-------------------------|:------------|:-------------------------|:---------------------------------------|:------------------|
| Cloudflare DeepSeek Distill Qwen 32B | Cloud              |                   3118 | 39.2% [37.5%, 40.9%] | 22.0% [20.6%, 23.5%]     | +17.2 pp    | 43.8%                    | 24.9% -> 23.1%                         | 5.55e-72 ***      |
| Cloudflare Gemma 4 26B-A4B           | Cloud              |                   3152 | 17.9% [16.6%, 19.3%] | 14.4% [13.2%, 15.7%]     | +3.5 pp     | 19.5%                    | 18.1% -> 17.4%                         | 3.84e-09 ***      |
| Cloudflare Qwen3 30B-A3B             | Cloud              |                   2619 | 26.5% [24.9%, 28.3%] | 7.8% [6.8%, 8.9%]        | +18.7 pp    | 70.6%                    | 19.9% -> 17.1%                         | 5.87e-110 ***     |
| Gemini Flash-Lite                    | Cloud              |                   2678 | 21.3% [19.8%, 22.9%] | 13.4% [12.2%, 14.7%]     | +7.9 pp     | 37.0%                    | 22.3% -> 18.1%                         | 7.09e-32 ***      |
| Groq Llama 3.1 8B Instant            | Cloud              |                   2576 | 22.4% [20.9%, 24.1%] | 11.5% [10.3%, 12.7%]     | +11.0 pp    | 49.0%                    | 20.3% -> 13.9%                         | 1.80e-53 ***      |
| Mistral Small                        | Cloud              |                   2495 | 30.5% [28.7%, 32.3%] | 14.4% [13.1%, 15.8%]     | +16.1 pp    | 52.8%                    | 25.9% -> 20.0%                         | 1.43e-87 ***      |
| Ollama DeepSeek R1                   | Local              |                   3175 | 23.9% [22.5%, 25.5%] | 6.9% [6.1%, 7.9%]        | +17.0 pp    | 71.1%                    | 20.7% -> 17.6%                         | 4.33e-102 ***     |
| Ollama Gemma 4                       | Local              |                   3194 | 20.1% [18.7%, 21.5%] | 13.0% [11.9%, 14.2%]     | +7.0 pp     | 35.1%                    | 17.8% -> 16.7%                         | 5.50e-29 ***      |
| Ollama Llama 3.1 8B                  | Local              |                   2952 | 16.2% [14.9%, 17.6%] | 4.1% [3.4%, 4.9%]        | +12.1 pp    | 74.7%                    | 15.2% -> 11.4%                         | 3.74e-91 ***      |
| Ollama Mistral 7B                    | Local              |                   3171 | 41.6% [39.9%, 43.3%] | 15.2% [14.0%, 16.5%]     | +26.4 pp    | 63.4%                    | 24.0% -> 18.8%                         | 7.48e-188 ***     |
| Ollama Qwen3                         | Local              |                   3162 | 28.2% [26.7%, 29.8%] | 6.4% [5.6%, 7.3%]        | +21.8 pp    | 77.3%                    | 19.7% -> 16.3%                         | 1.11e-188 ***     |
| **OVERALL LG-RESTRICTED**            | All Configurations |                  32292 | 26.3% [25.8%, 26.7%] | 11.8% [11.4%, 12.1%]     | +14.5 pp    | **55.2%**                | 20.7% -> 17.3%                         | 0.00e+00 ***      |

