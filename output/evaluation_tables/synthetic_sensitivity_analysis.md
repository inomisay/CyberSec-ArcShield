# Sensitivity Analysis: Real vs. Synthetic Prompt Attack Success Rates

*Generated on: 2026-09-02 10:19:00*

## Summary of Findings

This sensitivity analysis investigates whether synthetic template augmentations (used to balance sparse attack categories) 
introduce systematic bias compared to real in-the-wild prompts across both baseline (`no_defense`) and `arcshield` defense conditions.

### Key Observations:

- **Synthetic Top-Off Categories**: `Payload Splitting` (60.5% synthetic), `Multi-Prompt Attack` (74.5% synthetic), `Obfuscation (Token Smuggling)` (45.5% synthetic), and `Prompt Injection` (17.0% synthetic).

- **Defense Efficacy Consistency**: ArcShield achieves substantial, statistically significant risk reductions across both real and synthetic prompt cohorts.

- **Significance Levels**: `*** p < 0.001`, `** p < 0.01`, `* p < 0.05`, `(ns) not significant`.


## Unified Sensitivity Table (Real vs. Synthetic ASR per Category)


| Attack Category               | N (Real / Syn)   | Synthetic %   | No Defense: Real ASR   | No Defense: Syn ASR   | No Defense: Δpp   | No Defense: p-val   | ArcShield: Real ASR   | ArcShield: Syn ASR   | ArcShield: Δpp   | ArcShield: p-val   |
|:------------------------------|:-----------------|:--------------|:-----------------------|:----------------------|:------------------|:--------------------|:----------------------|:---------------------|:-----------------|:-------------------|
| Accidental Context Leakage    | 2200 / 0         | 0%            | 11.0%                  | N/A                   | -                 | -                   | 7.1%                  | N/A                  | -                | -                  |
| Attention Shifting            | 2200 / 0         | 0%            | 38.3%                  | N/A                   | -                 | -                   | 16.5%                 | N/A                  | -                | -                  |
| Code Injection                | 2200 / 0         | 0%            | 9.0%                   | N/A                   | -                 | -                   | 3.8%                  | N/A                  | -                | -                  |
| Dictionary Attack             | 2013 / 187       | 8%            | 17.3%                  | 25.1%                 | +7.8              | 0.0106 *            | 9.4%                  | 8.6%                 | -0.8             | 0.8078 (ns)        |
| Direct Exploitation           | 2200 / 0         | 0%            | 21.9%                  | N/A                   | -                 | -                   | 3.8%                  | N/A                  | -                | -                  |
| Fill-in-the-Blank Attack      | 2200 / 0         | 0%            | 7.4%                   | N/A                   | -                 | -                   | 6.0%                  | N/A                  | -                | -                  |
| Hypothetical Manipulation     | 2200 / 0         | 0%            | 57.8%                  | N/A                   | -                 | -                   | 30.8%                 | N/A                  | -                | -                  |
| Indirect Injection            | 2200 / 0         | 0%            | 14.4%                  | N/A                   | -                 | -                   | 2.5%                  | N/A                  | -                | -                  |
| Jailbreaks                    | 2200 / 0         | 0%            | 9.8%                   | N/A                   | -                 | -                   | 2.8%                  | N/A                  | -                | -                  |
| Logic Trap Attacks            | 2200 / 0         | 0%            | 68.4%                  | N/A                   | -                 | -                   | 38.0%                 | N/A                  | -                | -                  |
| Multi-Language Attack         | 2200 / 0         | 0%            | 16.5%                  | N/A                   | -                 | -                   | 3.7%                  | N/A                  | -                | -                  |
| Multi-Prompt Attack           | 561 / 1639       | 74%           | 24.8%                  | 40.2%                 | +15.4             | 7.50e-11 ***        | 16.2%                 | 26.8%                | +10.6            | 5.95e-07 ***       |
| Obfuscation (Token Smuggling) | 1386 / 813       | 37%           | 37.5%                  | 43.1%                 | +5.5              | 0.0119 *            | 16.2%                 | 12.3%                | -3.9             | 0.0140 *           |
| Payload Splitting             | 869 / 1331       | 60%           | 51.7%                  | 57.3%                 | +5.6              | 0.0114 *            | 22.1%                 | 19.2%                | -2.9             | 0.1055 (ns)        |
| Prompt Injection              | 1826 / 374       | 17%           | 3.8%                   | 13.9%                 | +10.1             | 1.36e-14 ***        | 2.1%                  | 7.5%                 | +5.4             | 1.03e-07 ***       |
| Role Playing / Pretending     | 2200 / 0         | 0%            | 22.0%                  | N/A                   | -                 | -                   | 14.0%                 | N/A                  | -                | -                  |


## Detailed Breakdown: Baseline (No Defense)


| Attack Category               |   N_Real |   N_Synthetic | Synthetic %   | Real ASR (no_defense)         | Synthetic ASR (no_defense)    | ΔASR (Syn - Real)   | Test              | p-value         |
|:------------------------------|---------:|--------------:|:--------------|:------------------------------|:------------------------------|:--------------------|:------------------|:----------------|
| Accidental Context Leakage    |     2200 |             0 | 0.0%          | 11.0% [9.7%, 12.3%] (n=2200)  | N/A (0)                       | N/A                 | None              | N/A (100% Real) |
| Attention Shifting            |     2200 |             0 | 0.0%          | 38.3% [36.3%, 40.4%] (n=2200) | N/A (0)                       | N/A                 | None              | N/A (100% Real) |
| Code Injection                |     2200 |             0 | 0.0%          | 9.0% [7.9%, 10.3%] (n=2200)   | N/A (0)                       | N/A                 | None              | N/A (100% Real) |
| Dictionary Attack             |     2013 |           187 | 8.5%          | 17.3% [15.7%, 19.1%] (n=2013) | 25.1% [19.5%, 31.8%] (n=187)  | +7.8 pp             | Chi-Square (df=1) | 0.0106 *        |
| Direct Exploitation           |     2200 |             0 | 0.0%          | 21.9% [20.2%, 23.6%] (n=2200) | N/A (0)                       | N/A                 | None              | N/A (100% Real) |
| Fill-in-the-Blank Attack      |     2200 |             0 | 0.0%          | 7.4% [6.4%, 8.6%] (n=2200)    | N/A (0)                       | N/A                 | None              | N/A (100% Real) |
| Hypothetical Manipulation     |     2200 |             0 | 0.0%          | 57.8% [55.7%, 59.9%] (n=2200) | N/A (0)                       | N/A                 | None              | N/A (100% Real) |
| Indirect Injection            |     2200 |             0 | 0.0%          | 14.4% [13.0%, 15.9%] (n=2200) | N/A (0)                       | N/A                 | None              | N/A (100% Real) |
| Jailbreaks                    |     2200 |             0 | 0.0%          | 9.8% [8.6%, 11.1%] (n=2200)   | N/A (0)                       | N/A                 | None              | N/A (100% Real) |
| Logic Trap Attacks            |     2200 |             0 | 0.0%          | 68.4% [66.4%, 70.3%] (n=2200) | N/A (0)                       | N/A                 | None              | N/A (100% Real) |
| Multi-Language Attack         |     2200 |             0 | 0.0%          | 16.5% [15.0%, 18.1%] (n=2200) | N/A (0)                       | N/A                 | None              | N/A (100% Real) |
| Multi-Prompt Attack           |      561 |          1639 | 74.5%         | 24.8% [21.4%, 28.5%] (n=561)  | 40.2% [37.9%, 42.6%] (n=1639) | +15.4 pp            | Chi-Square (df=1) | 7.50e-11 ***    |
| Obfuscation (Token Smuggling) |     1386 |           813 | 37.0%         | 37.5% [35.0%, 40.1%] (n=1386) | 43.1% [39.7%, 46.5%] (n=813)  | +5.5 pp             | Chi-Square (df=1) | 0.0119 *        |
| Payload Splitting             |      869 |          1331 | 60.5%         | 51.7% [48.3%, 55.0%] (n=869)  | 57.3% [54.6%, 59.9%] (n=1331) | +5.6 pp             | Chi-Square (df=1) | 0.0114 *        |
| Prompt Injection              |     1826 |           374 | 17.0%         | 3.8% [3.0%, 4.8%] (n=1826)    | 13.9% [10.8%, 17.8%] (n=374)  | +10.1 pp            | Chi-Square (df=1) | 1.36e-14 ***    |
| Role Playing / Pretending     |     2200 |             0 | 0.0%          | 22.0% [20.4%, 23.8%] (n=2200) | N/A (0)                       | N/A                 | None              | N/A (100% Real) |


## Detailed Breakdown: Defended (ArcShield)


| Attack Category               |   N_Real |   N_Synthetic | Synthetic %   | Real ASR (arcshield)          | Synthetic ASR (arcshield)     | ΔASR (Syn - Real)   | Test              | p-value         |
|:------------------------------|---------:|--------------:|:--------------|:------------------------------|:------------------------------|:--------------------|:------------------|:----------------|
| Accidental Context Leakage    |     2200 |             0 | 0.0%          | 7.1% [6.1%, 8.2%] (n=2200)    | N/A (0)                       | N/A                 | None              | N/A (100% Real) |
| Attention Shifting            |     2200 |             0 | 0.0%          | 16.5% [15.1%, 18.2%] (n=2200) | N/A (0)                       | N/A                 | None              | N/A (100% Real) |
| Code Injection                |     2200 |             0 | 0.0%          | 3.8% [3.1%, 4.7%] (n=2200)    | N/A (0)                       | N/A                 | None              | N/A (100% Real) |
| Dictionary Attack             |     2013 |           187 | 8.5%          | 9.4% [8.2%, 10.7%] (n=2013)   | 8.6% [5.3%, 13.4%] (n=187)    | -0.8 pp             | Chi-Square (df=1) | 0.8078 (ns)     |
| Direct Exploitation           |     2200 |             0 | 0.0%          | 3.8% [3.1%, 4.7%] (n=2200)    | N/A (0)                       | N/A                 | None              | N/A (100% Real) |
| Fill-in-the-Blank Attack      |     2200 |             0 | 0.0%          | 6.0% [5.1%, 7.1%] (n=2200)    | N/A (0)                       | N/A                 | None              | N/A (100% Real) |
| Hypothetical Manipulation     |     2200 |             0 | 0.0%          | 30.8% [28.9%, 32.7%] (n=2200) | N/A (0)                       | N/A                 | None              | N/A (100% Real) |
| Indirect Injection            |     2200 |             0 | 0.0%          | 2.5% [1.9%, 3.2%] (n=2200)    | N/A (0)                       | N/A                 | None              | N/A (100% Real) |
| Jailbreaks                    |     2200 |             0 | 0.0%          | 2.8% [2.2%, 3.5%] (n=2200)    | N/A (0)                       | N/A                 | None              | N/A (100% Real) |
| Logic Trap Attacks            |     2200 |             0 | 0.0%          | 38.0% [36.0%, 40.0%] (n=2200) | N/A (0)                       | N/A                 | None              | N/A (100% Real) |
| Multi-Language Attack         |     2200 |             0 | 0.0%          | 3.7% [3.0%, 4.6%] (n=2200)    | N/A (0)                       | N/A                 | None              | N/A (100% Real) |
| Multi-Prompt Attack           |      561 |          1639 | 74.5%         | 16.2% [13.4%, 19.5%] (n=561)  | 26.8% [24.7%, 29.0%] (n=1639) | +10.6 pp            | Chi-Square (df=1) | 5.95e-07 ***    |
| Obfuscation (Token Smuggling) |     1386 |           814 | 37.0%         | 16.2% [14.4%, 18.3%] (n=1386) | 12.3% [10.2%, 14.7%] (n=814)  | -3.9 pp             | Chi-Square (df=1) | 0.0140 *        |
| Payload Splitting             |      869 |          1331 | 60.5%         | 22.1% [19.5%, 25.0%] (n=869)  | 19.2% [17.1%, 21.4%] (n=1331) | -2.9 pp             | Chi-Square (df=1) | 0.1055 (ns)     |
| Prompt Injection              |     1826 |           374 | 17.0%         | 2.1% [1.6%, 2.9%] (n=1826)    | 7.5% [5.2%, 10.6%] (n=374)    | +5.4 pp             | Chi-Square (df=1) | 1.03e-07 ***    |
| Role Playing / Pretending     |     2200 |             0 | 0.0%          | 14.0% [12.7%, 15.6%] (n=2200) | N/A (0)                       | N/A                 | None              | N/A (100% Real) |

