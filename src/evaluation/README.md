# ArcShield Controlled Evaluation Runs

This folder is for slow, reproducible model testing when the full backend batch
run is too expensive for online providers.

Use it to run one model and one attack type at a time, then repeat for the next
attack type. Each run saves raw model responses, judge outputs, metrics, config
metadata, and an optional human annotation sheet under `output/model_results/`.

## Core Roles

- `src/core/attacker.py`: sends prompts to the selected target model and logs
  latency/token metadata.
- `src/core/defender.py`: defines benchmark defense conditions:
  `no_defense` and `arcshield`.
- `src/core/moderator.py`: optional defense layer for second-pass moderation.
  It is separate from the final judge and can be ablated later.
- `src/core/judge.py`: final offline deterministic judge using heuristic rules
  plus local Llama Guard when available.

## Recommended Workflow

Run a small pilot first:

```powershell
python -m src.evaluation.run_attack_type_experiment --model google:gemini-flash-lite-latest --attack-type "Prompt Injection" --condition both --limit 5
```

Run one full attack type for one model:

```powershell
python -m src.evaluation.run_attack_type_experiment --model google:gemini-flash-lite-latest --attack-type "Prompt Injection" --condition both
```

Run only the baseline:

```powershell
python -m src.evaluation.run_attack_type_experiment --model github:Phi-4 --attack-type "Jailbreaks" --condition no_defense
```

Run only ArcShield defense:

```powershell
python -m src.evaluation.run_attack_type_experiment --model github:Phi-4 --attack-type "Jailbreaks" --condition arcshield
```

List valid attack type names:

```powershell
python -m src.evaluation.run_attack_type_experiment --list-attack-types
```

## Output Layout

Default output:

```text
output/model_results/
  <run_id>/
    <model_slug>/
      <attack_type_slug>/
        no_defense/
          results.csv
          results.json
          metrics.json
          evaluation_config.json
        arcshield/
          results.csv
          results.json
          metrics.json
          evaluation_config.json
        combined/
          results.csv
          results.json
          metrics.json
          annotation_sheet.csv
```

Use the per-condition folders to compare baseline vs ArcShield ASR/DSR. Use the
combined folder for paper tables and human validation sampling.

## Benign Usability Evaluation

Attack metrics alone cannot show over-refusal. Run the benign evaluator to
measure Benign Refusal Rate (BRR) on harmless prompts:

```powershell
python -m src.evaluation.run_benign_experiment --model ollama:llama3.1:8b --condition both --limit 12
```

Full benign run:

```powershell
python -m src.evaluation.run_benign_experiment --model ollama:llama3.1:8b --condition both
```

See `README_benign_usability.md` for the BRR methodology and paper-ready
security-usability reporting format.

The benign CSV/JSON files are stored under `dataset/benign/`; they are not part
of the attack dataset and do not need to be regenerated for normal evaluation.
