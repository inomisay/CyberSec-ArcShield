# ArcShield Curated Attack Dataset

This folder contains the preparation scripts that build the curated benchmark CSV for ArcShield:

- `build_attack_dataset.py`
- `dataset_manager.py`

The current generated dataset files are:

- `dataset/curated/attack_dataset_curated.csv`
- `dataset/curated/attack_dataset_curated_summary.json`

## Overview

`build_attack_dataset.py` combines prompts from local datasets and Hugging Face datasets, deduplicates them, classifies them into 16 attack types, and samples up to 200 prompts per type.

If an attack type has fewer than 200 real source prompts, the builder may add synthetic template variants for supported sparse classes. These rows are marked with `is_synthetic=true`, and their template family is stored in `augmentation_method` and `augmentation_variant`.

## Dataset size

- Source records collected before sampling: 49,571
- Final sampled rows: 3,200
- Target per attack type: 200
- Attack types: 16
- Source groups in the final CSV: 14
- Real source rows: 2,805
- Synthetic top-off rows: 395

## Attack types

The final CSV contains these 16 attack types:

1. Direct Exploitation
2. Jailbreaks
3. Prompt Injection
4. Indirect Injection
5. Role Playing / Pretending
6. Attention Shifting
7. Logic Trap Attacks
8. Hypothetical Manipulation
9. Accidental Context Leakage
10. Code Injection
11. Dictionary Attack
12. Fill-in-the-Blank Attack
13. Multi-Language Attack
14. Multi-Prompt Attack
15. Obfuscation (Token Smuggling)
16. Payload Splitting

## Real and Synthetic Counts

<table>
  <thead>
    <tr>
      <th>Attack type</th>
      <th align="right">Real rows</th>
      <th align="right">Synthetic rows</th>
      <th align="right">Final rows</th>
    </tr>
  </thead>
  <tbody>
    <tr><td>Accidental Context Leakage</td><td align="right">200</td><td align="right">0</td><td align="right">200</td></tr>
    <tr><td>Attention Shifting</td><td align="right">200</td><td align="right">0</td><td align="right">200</td></tr>
    <tr><td>Code Injection</td><td align="right">200</td><td align="right">0</td><td align="right">200</td></tr>
    <tr><td>Dictionary Attack</td><td align="right">183</td><td align="right">17</td><td align="right">200</td></tr>
    <tr><td>Direct Exploitation</td><td align="right">200</td><td align="right">0</td><td align="right">200</td></tr>
    <tr><td>Fill-in-the-Blank Attack</td><td align="right">200</td><td align="right">0</td><td align="right">200</td></tr>
    <tr><td>Hypothetical Manipulation</td><td align="right">200</td><td align="right">0</td><td align="right">200</td></tr>
    <tr><td>Indirect Injection</td><td align="right">200</td><td align="right">0</td><td align="right">200</td></tr>
    <tr><td>Jailbreaks</td><td align="right">200</td><td align="right">0</td><td align="right">200</td></tr>
    <tr><td>Logic Trap Attacks</td><td align="right">200</td><td align="right">0</td><td align="right">200</td></tr>
    <tr><td>Multi-Language Attack</td><td align="right">200</td><td align="right">0</td><td align="right">200</td></tr>
    <tr><td>Multi-Prompt Attack</td><td align="right">51</td><td align="right">149</td><td align="right">200</td></tr>
    <tr><td>Obfuscation (Token Smuggling)</td><td align="right">126</td><td align="right">74</td><td align="right">200</td></tr>
    <tr><td>Payload Splitting</td><td align="right">79</td><td align="right">121</td><td align="right">200</td></tr>
    <tr><td>Prompt Injection</td><td align="right">166</td><td align="right">34</td><td align="right">200</td></tr>
    <tr><td>Role Playing / Pretending</td><td align="right">200</td><td align="right">0</td><td align="right">200</td></tr>
  </tbody>
</table>

## Synthetic Template Variants

Synthetic template variants are only used when a supported attack type has fewer real rows than the `--per-type` target. The builder cycles through the available real prompts for that attack type and wraps each prompt with one of the configured templates.

This means `sampled_attack_type_counts` can be larger than `attack_type_counts` for sparse attack types. For example, `Payload Splitting` has 79 real rows, then receives 121 synthetic rows to reach 200. `Multi-Prompt Attack` has 51 real rows, then receives 149 synthetic rows to reach 200.

`Prompt Injection` has 166 real rows, then receives 34 synthetic rows to reach 200.

<table>
  <thead>
    <tr>
      <th>Attack type</th>
      <th>Template behavior</th>
    </tr>
  </thead>
  <tbody>
    <tr><td>Dictionary Attack</td><td>Wraps the source prompt as a dictionary-style definition request.</td></tr>
    <tr><td>Multi-Prompt Attack</td><td>Wraps the source prompt as a staged or numbered multi-step prompt.</td></tr>
    <tr><td>Prompt Injection</td><td>Wraps the source prompt as an instruction override, higher-priority instruction, or previous-instruction bypass request.</td></tr>
    <tr><td>Obfuscation (Token Smuggling)</td><td>Wraps the source prompt as an encoded, obfuscated, or hidden-intent request.</td></tr>
    <tr><td>Payload Splitting</td><td>Wraps the source prompt as a chunked, staged, or multi-part payload request.</td></tr>
  </tbody>
</table>

## Source groups

These source groups appear in the final CSV:

<table>
  <thead>
    <tr>
      <th>Source group</th>
      <th align="right">Rows</th>
    </tr>
  </thead>
  <tbody>
    <tr><td>HF:nvidia/Aegis-AI-Content-Safety-Dataset-2.0</td><td align="right">919</td></tr>
    <tr><td>HF:xTRam1/safe-guard-prompt-injection</td><td align="right">593</td></tr>
    <tr><td>JailbreakLLMs</td><td align="right">557</td></tr>
    <tr><td>QueryAttack</td><td align="right">378</td></tr>
    <tr><td>Foot-in-the-door-Jailbreak</td><td align="right">179</td></tr>
    <tr><td>HF:JailbreakBench/JBB-Behaviors/judge_comparison</td><td align="right">146</td></tr>
    <tr><td>TroGEN</td><td align="right">137</td></tr>
    <tr><td>HF:TrustAIRLab/in-the-wild-jailbreak-prompts/jailbreak_2023_12_25</td><td align="right">109</td></tr>
    <tr><td>MAGIC</td><td align="right">105</td></tr>
    <tr><td>HF:LibrAI/do-not-answer</td><td align="right">29</td></tr>
    <tr><td>HF:deepset/prompt-injections</td><td align="right">22</td></tr>
    <tr><td>strongreject</td><td align="right">14</td></tr>
    <tr><td>educational-llm-guardrails-bench</td><td align="right">8</td></tr>
    <tr><td>MPA</td><td align="right">4</td></tr>
  </tbody>
</table>

## Which sources feed each attack type

<table>
  <thead>
    <tr>
      <th>Attack type</th>
      <th>Source groups used</th>
    </tr>
  </thead>
  <tbody>
    <tr><td>Accidental Context Leakage</td><td>JailbreakLLMs, HF:xTRam1/safe-guard-prompt-injection, HF:nvidia/Aegis-AI-Content-Safety-Dataset-2.0, HF:TrustAIRLab/in-the-wild-jailbreak-prompts/jailbreak_2023_12_25, QueryAttack, HF:deepset/prompt-injections, TroGEN, Foot-in-the-door-Jailbreak, strongreject, HF:JailbreakBench/JBB-Behaviors/judge_comparison</td></tr>
    <tr><td>Attention Shifting</td><td>HF:nvidia/Aegis-AI-Content-Safety-Dataset-2.0, Foot-in-the-door-Jailbreak, JailbreakLLMs, TroGEN, MAGIC, HF:JailbreakBench/JBB-Behaviors/judge_comparison, HF:LibrAI/do-not-answer, strongreject, HF:TrustAIRLab/in-the-wild-jailbreak-prompts/jailbreak_2023_12_25</td></tr>
    <tr><td>Code Injection</td><td>HF:xTRam1/safe-guard-prompt-injection, HF:deepset/prompt-injections, HF:nvidia/Aegis-AI-Content-Safety-Dataset-2.0, Foot-in-the-door-Jailbreak, TroGEN, JailbreakLLMs, HF:TrustAIRLab/in-the-wild-jailbreak-prompts/jailbreak_2023_12_25, MAGIC, strongreject, HF:JailbreakBench/JBB-Behaviors/judge_comparison</td></tr>
    <tr><td>Dictionary Attack</td><td>HF:nvidia/Aegis-AI-Content-Safety-Dataset-2.0, JailbreakLLMs, HF:xTRam1/safe-guard-prompt-injection, HF:TrustAIRLab/in-the-wild-jailbreak-prompts/jailbreak_2023_12_25, HF:JailbreakBench/JBB-Behaviors/judge_comparison, strongreject</td></tr>
    <tr><td>Direct Exploitation</td><td>HF:nvidia/Aegis-AI-Content-Safety-Dataset-2.0, MAGIC, HF:LibrAI/do-not-answer, strongreject, educational-llm-guardrails-bench</td></tr>
    <tr><td>Fill-in-the-Blank Attack</td><td>JailbreakLLMs, HF:nvidia/Aegis-AI-Content-Safety-Dataset-2.0, HF:xTRam1/safe-guard-prompt-injection, HF:TrustAIRLab/in-the-wild-jailbreak-prompts/jailbreak_2023_12_25</td></tr>
    <tr><td>Hypothetical Manipulation</td><td>TroGEN, HF:nvidia/Aegis-AI-Content-Safety-Dataset-2.0, Foot-in-the-door-Jailbreak, JailbreakLLMs, HF:xTRam1/safe-guard-prompt-injection, HF:TrustAIRLab/in-the-wild-jailbreak-prompts/jailbreak_2023_12_25, HF:JailbreakBench/JBB-Behaviors/judge_comparison, MAGIC</td></tr>
    <tr><td>Indirect Injection</td><td>QueryAttack, HF:xTRam1/safe-guard-prompt-injection</td></tr>
    <tr><td>Jailbreaks</td><td>HF:nvidia/Aegis-AI-Content-Safety-Dataset-2.0, Foot-in-the-door-Jailbreak, JailbreakLLMs, HF:LibrAI/do-not-answer, TroGEN, HF:JailbreakBench/JBB-Behaviors/judge_comparison, MPA, strongreject, educational-llm-guardrails-bench</td></tr>
    <tr><td>Logic Trap Attacks</td><td>HF:nvidia/Aegis-AI-Content-Safety-Dataset-2.0, MAGIC, Foot-in-the-door-Jailbreak, TroGEN, JailbreakLLMs, HF:LibrAI/do-not-answer, HF:TrustAIRLab/in-the-wild-jailbreak-prompts/jailbreak_2023_12_25, HF:JailbreakBench/JBB-Behaviors/judge_comparison, strongreject, MPA</td></tr>
    <tr><td>Multi-Language Attack</td><td>QueryAttack, HF:xTRam1/safe-guard-prompt-injection, JailbreakLLMs, HF:nvidia/Aegis-AI-Content-Safety-Dataset-2.0, HF:LibrAI/do-not-answer, Foot-in-the-door-Jailbreak</td></tr>
    <tr><td>Multi-Prompt Attack</td><td>JailbreakLLMs, HF:nvidia/Aegis-AI-Content-Safety-Dataset-2.0, HF:xTRam1/safe-guard-prompt-injection, HF:TrustAIRLab/in-the-wild-jailbreak-prompts/jailbreak_2023_12_25, MAGIC</td></tr>
    <tr><td>Obfuscation (Token Smuggling)</td><td>HF:nvidia/Aegis-AI-Content-Safety-Dataset-2.0, Foot-in-the-door-Jailbreak, HF:TrustAIRLab/in-the-wild-jailbreak-prompts/jailbreak_2023_12_25, HF:xTRam1/safe-guard-prompt-injection, JailbreakLLMs, HF:LibrAI/do-not-answer, TroGEN, MAGIC, educational-llm-guardrails-bench, strongreject</td></tr>
    <tr><td>Payload Splitting</td><td>HF:JailbreakBench/JBB-Behaviors/judge_comparison, JailbreakLLMs, HF:nvidia/Aegis-AI-Content-Safety-Dataset-2.0, HF:xTRam1/safe-guard-prompt-injection, HF:TrustAIRLab/in-the-wild-jailbreak-prompts/jailbreak_2023_12_25, Foot-in-the-door-Jailbreak</td></tr>
    <tr><td>Prompt Injection</td><td>HF:xTRam1/safe-guard-prompt-injection, HF:TrustAIRLab/in-the-wild-jailbreak-prompts/jailbreak_2023_12_25, JailbreakLLMs, HF:deepset/prompt-injections, educational-llm-guardrails-bench, HF:nvidia/Aegis-AI-Content-Safety-Dataset-2.0</td></tr>
    <tr><td>Role Playing / Pretending</td><td>JailbreakLLMs, HF:xTRam1/safe-guard-prompt-injection, HF:nvidia/Aegis-AI-Content-Safety-Dataset-2.0, HF:TrustAIRLab/in-the-wild-jailbreak-prompts/jailbreak_2023_12_25, HF:deepset/prompt-injections, HF:JailbreakBench/JBB-Behaviors/judge_comparison, MAGIC, Foot-in-the-door-Jailbreak</td></tr>
  </tbody>
</table>

## CSV header meanings

<table>
  <thead>
    <tr>
      <th>Column</th>
      <th>Meaning</th>
    </tr>
  </thead>
  <tbody>
    <tr><td><code>prompt</code></td><td>The prompt text used in the benchmark row.</td></tr>
    <tr><td><code>attack_type</code></td><td>The assigned attack category for the prompt.</td></tr>
    <tr><td><code>source_dataset</code></td><td>The dataset group the prompt came from.</td></tr>
    <tr><td><code>source_file</code></td><td>The original file or Hugging Face path that supplied the prompt.</td></tr>
    <tr><td><code>source_row</code></td><td>The row number or record reference inside the source.</td></tr>
    <tr><td><code>raw_label</code></td><td>The original label from the source dataset, if one was available.</td></tr>
    <tr><td><code>is_synthetic</code></td><td><code>true</code> if the row was generated to top up a sparse attack type; <code>false</code> if it came directly from a source.</td></tr>
    <tr><td><code>augmentation_method</code></td><td>The augmentation family used for synthetic rows; blank for real source rows.</td></tr>
    <tr><td><code>augmentation_variant</code></td><td>The specific synthetic variant label; blank for real source rows.</td></tr>
  </tbody>
</table>

## Notes

- Empty `raw_label`, `augmentation_method`, and `augmentation_variant` values are normal for source rows that did not provide labels or for rows that were not synthesized.
- `is_synthetic=true` means the row was created by `build_attack_dataset.py` to move a supported sparse class toward the 200-per-type target.
- The dataset includes both local folders under `dataset/` and Hugging Face sources referenced in the CSV.
- `attack_type_counts` in the summary file counts real deduplicated source prompts before final sampling and synthetic top-off.
- `sampled_attack_type_counts` counts the final rows written to the CSV, including synthetic rows where present.

---------------------------------------------------------
The curated csv dataset have:
    3,200 rows
    16 attack types
    200 rows per attack type
    395 synthetic rows total
