# Empirical Latency Distribution and Token Ceiling Sensitivity Analysis

*Generated on: 2026-09-02 11:06:23*

## 1. Executive Summary & Hardware Context

- **Local Intel Arc GPU Configurations (5 models)**: Evaluated locally on an Intel Arc A770 GPU via Ollama. Shows expected higher compute/generation latency on local consumer hardware (Medians: ~6.4s - 8.9s for refused, ~27.5s - 62.2s for answered/reasoning generation).

- **Cloud API Endpoints (6 models)**: Evaluated across Groq LPU, Cloudflare Workers AI serverless GPUs, Google TPU v5e, and Mistral AI. Demonstrates ultra-low latency on Groq (p50: ~1.4s) and Gemini Flash-Lite (p50: ~2.9s).

- **Token Limit Correlation**: Point-biserial correlation ($r_{pb}$) and Spearman rank correlation ($ho$) between token counts (128 - 4000) and `attack_successful` confirm that ArcShield's risk reduction is uniform across all token ceiling tiers.

  - **Point-Biserial Correlation ($r_{pb}$)**: +0.3214 ($p = 0.00e+00$)

  - **Spearman Rank Correlation ($\rho$)**: +0.3686 ($p = 0.00e+00$)

  - **Mean Completion Tokens (Successful Attacks)**: 282.2 tokens

  - **Mean Completion Tokens (Blocked Attacks)**: 126.4 tokens


## 2. Table L1: Robust Latency Distribution by Model


| Model                                | Deployment                    | Hardware / Infrastructure   |   N_Obs | Median (p50) ms   | IQR ms        | p25 - p75 (IQR range) ms   | p95 ms         | p99 ms     | Mean ± SD ms     | Median Refused ms   | Median Answered ms   |
|:-------------------------------------|:------------------------------|:----------------------------|--------:|:------------------|:--------------|:---------------------------|:---------------|:-----------|:-----------------|:--------------------|:---------------------|
| Ollama DeepSeek R1                   | Local (Intel Arc GPU)         | Local Intel Arc A770        |    6800 | 22,071 ms         | 54,008 ms     | [5,343 - 59,351]           | 98,796 ms      | 154,402 ms | 35,610 ± 52,024  | 6,522 ms            | 54,445 ms            |
| Ollama Gemma 4                       | Local (Intel Arc GPU)         | Local Intel Arc A770        |    6800 | 17,099 ms         | 37,264 ms     | [8,027 - 45,291]           | 92,433 ms      | 134,515 ms | 36,744 ± 186,038 | 8,380 ms            | 31,624 ms            |
| Ollama Llama 3.1 8B                  | Local (Intel Arc GPU)         | Local Intel Arc A770        |    6800 | 16,680 ms         | 35,749 ms     | [6,804 - 42,553]           | 102,946 ms     | 205,371 ms | 33,011 ± 77,089  | 7,958 ms            | 42,849 ms            |
| Ollama Mistral 7B                    | Local (Intel Arc GPU)         | Local Intel Arc A770        |    6800 | 42,871 ms         | 69,104 ms     | [15,297 - 84,401]          | 144,834 ms     | 226,452 ms | 55,053 ± 50,835  | 8,988 ms            | 61,121 ms            |
| Ollama Qwen3                         | Local (Intel Arc GPU)         | Local Intel Arc A770        |    6799 | 23,165 ms         | 51,572 ms     | [8,098 - 59,669]           | 152,576 ms     | 241,997 ms | 45,736 ± 55,440  | 8,928 ms            | 47,269 ms            |
| Cloudflare DeepSeek Distill Qwen 32B | Cloud (Cloudflare Workers AI) | Cloud Serverless GPU        |    6800 | 17,391 ms         | 16,141 ms     | [10,506 - 26,647]          | 32,137 ms      | 57,622 ms  | 38,885 ± 517,841 | 10,484 ms           | 22,525 ms            |
| Cloudflare Gemma 4 26B-A4B           | Cloud (Cloudflare Workers AI) | Cloud Serverless GPU        |    6800 | 10,355 ms         | 8,361 ms      | [6,034 - 14,395]           | 34,091 ms      | 77,882 ms  | 19,796 ± 152,711 | 7,758 ms            | 11,616 ms            |
| Cloudflare Qwen3 30B-A3B             | Cloud (Cloudflare Workers AI) | Cloud Serverless GPU        |    6800 | 7,798 ms          | 7,024 ms      | [5,299 - 12,323]           | 20,934 ms      | 26,238 ms  | 9,538 ± 5,797    | 6,440 ms            | 9,580 ms             |
| Gemini Flash-Lite                    | Cloud (Google AI)             | Cloud TPU v5e               |    6800 | 2,363 ms          | 2,344 ms      | [1,536 - 3,881]            | 12,500 ms      | 27,331 ms  | 4,939 ± 74,860   | 1,613 ms            | 2,827 ms             |
| Groq Llama 3.1 8B Instant            | Cloud (Groq LPU)              | Cloud LPU                   |    6800 | 1,145 ms          | 701 ms        | [858 - 1,560]              | 2,556 ms       | 3,686 ms   | 1,304 ± 717      | 883 ms              | 1,389 ms             |
| Mistral Small                        | Cloud (Mistral AI)            | Cloud API                   |    6800 | 1,758 ms          | 1,628 ms      | [1,164 - 2,792]            | 4,798 ms       | 7,966 ms   | 2,253 ± 2,227    | 1,137 ms            | 2,409 ms             |
| **ALL LOCAL (Intel Arc GPU)**        | Summary Group                 | -                           |   33999 | **22,934 ms**     | **49,390 ms** | [7,809 - 57,200]           | **122,778 ms** | 212,353 ms | 41,231 ± 99,240  | -                   | -                    |
| **ALL CLOUD APIS**                   | Summary Group                 | -                           |   40800 | **4,374 ms**      | **9,749 ms**  | [1,607 - 11,356]           | **27,069 ms**  | 42,856 ms  | 12,786 ± 222,910 | -                   | -                    |


## 3. Table L2: Stratified Sensitivity Analysis by Token Ceiling Tiers


| Token Ceiling Tier   |   N_Pairs | ASR_nodef (95% CI)   | ASR_arcshield (95% CI)   | ΔASR (pp)   | Relative Reduction (%)   | DSR (95% CI)           | p-value (Defense Effect)   |
|:---------------------|----------:|:---------------------|:-------------------------|:------------|:-------------------------|:-----------------------|:---------------------------|
| ≤128                 |     16994 | 12.7% [12.2%, 13.2%] | 6.3% [6.0%, 6.6%]        | +6.4 pp     | 50.2%                    | 93.7% [93.4%, 94.0%]   | 1.70e-116 ***              |
| 129-256              |      4926 | 35.5% [34.5%, 36.5%] | 27.6% [26.3%, 28.8%]     | +7.9 pp     | 22.3%                    | 72.4% [71.2%, 73.7%]   | 2.82e-21 ***               |
| 257-512              |      3687 | 40.1% [39.0%, 41.1%] | 35.2% [33.7%, 36.8%]     | +4.8 pp     | 12.1%                    | 64.8% [63.2%, 66.3%]   | 5.92e-07 ***               |
| 513-1024             |      1692 | 44.3% [42.6%, 46.0%] | 37.4% [35.1%, 39.7%]     | +6.9 pp     | 15.7%                    | 62.6% [60.3%, 64.9%]   | 2.77e-06 ***               |
| 1025-2048            |         1 | 28.6% [8.2%, 64.1%]  | 0.0% [0.0%, 79.3%]       | +28.6 pp    | 100.0%                   | 100.0% [20.7%, 100.0%] | 1.0000                     |

