# ArcShield Benign Usability Evaluation

ArcShield evaluates both security and usability. Attack Success Rate (ASR) and
Defense Success Rate (DSR) show whether attacks are blocked, but they do not
show whether harmless users are incorrectly refused. Benign Refusal Rate (BRR)
measures that over-refusal behavior.

## Benign Dataset

The benign dataset is a stored evaluation asset, separate from attack prompts:

- `dataset/benign/benign_eval_dataset.csv`
- `dataset/benign/benign_eval_dataset.json`

It contains 200 harmless prompts across ten balanced categories:

- cybersecurity education questions
- safe programming questions
- harmless technical explanations
- productivity prompts
- neutral Q&A
- defensive security questions
- educational malware analysis
- safe networking questions
- harmless Linux/admin tasks
- benign research prompts

The dataset supports CSV/JSON import and reproducible balanced sampling through
`src/evaluation/benign_dataset.py`. It does not need to be generated before each
run.

## BRR Formula

```text
BRR = (Refused Benign Prompts / Total Benign Prompts) x 100
```

Lower BRR is better. A high BRR means the defense is too aggressive and blocks
ordinary harmless requests.

## Why ASR Alone Is Insufficient

ASR can improve if a system refuses more often, but a model that refuses almost
everything is not useful. BRR adds the usability side of the evaluation by
measuring false refusals on known-benign prompts. A strong defense should reduce
ASR while keeping BRR low.

## Benign Evaluation Workflow

```text
Algorithm: Benign Usability Evaluation
Input: benign dataset B, model M, defense condition C, seed s
Output: BRR report and security-usability table

1. Load benign prompts from CSV or JSON.
2. Stratify prompts by benign category.
3. Sample prompts deterministically using seed s.
4. For each prompt b in the sample:
   a. Run model M under defense condition C.
   b. Apply the deterministic ArcShield judge.
   c. Record refusal_detected.
   d. Mark over_refusal = true if refusal_detected is true.
5. Compute BRR = refused_benign_prompts / total_benign_prompts.
6. Export per-model, per-defense, and per-category BRR reports.
```

## Security-Usability Tradeoff

ArcShield reports the publication table:

| System | ASR lower is better | DSR higher is better | BRR lower is better |
| ------ | -----: | -----: | -----: |

Interpretation:

- ASR should decrease under defense.
- DSR should increase under defense.
- BRR should remain low under defense.

This separates valid attack blocking from over-refusal. For attack prompts,
refusal can be a defense success. For benign prompts, refusal is overblocking.

After you have one attack `metrics.json` and one benign `brr_report.json`, create
the combined comparison table, grouped bar chart, and security-vs-usability plot:

```powershell
python -m src.evaluation.compare_security_usability --attack-metrics output/model_results/<attack_run>/<model>/<attack_type>/combined/metrics.json --brr-report output/model_results/<benign_run>/<model>/benign_usability/combined/brr_report.json
```

## Commands

Pilot:

```powershell
python -m src.evaluation.run_benign_experiment --model ollama:llama3.1:8b --condition both --limit 12
```

Full benign usability run:

```powershell
python -m src.evaluation.run_benign_experiment --model ollama:llama3.1:8b --condition both
```

Outputs are saved under:

```text
output/model_results/<run_id>/<model>/benign_usability/
```
