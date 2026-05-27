"""Run one ArcShield attack-type experiment at a time.

Examples:
  python -m src.evaluation.run_attack_type_experiment --list-attack-types
  python -m src.evaluation.run_attack_type_experiment --model google:gemini-flash-lite-latest --attack-type "Prompt Injection" --condition both --limit 5
  python -m src.evaluation.run_attack_type_experiment --model github:Phi-4 --attack-type "Jailbreaks" --condition no_defense

Recommended process for online models:
  1. Start with --limit 5 to verify keys, model name, and output paths.
  2. Run --condition no_defense for one attack type.
  3. Run --condition arcshield for the same attack type.
  4. Move to the next attack type when the provider/token budget is okay.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from src.core.attacker import execute_attack, get_attack_client, parse_model_spec
from src.core.defender import ARCSHIELD_DEFENSE, NO_DEFENSE, get_defense_condition
from src.core.judge import (
    ATTACK_TYPES,
    BenchmarkJudge,
    JudgeInput,
    compute_judge_metrics,
    export_annotation_sheet,
    export_annotation_guidelines,
    experiment_metadata,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_PATH = PROJECT_ROOT / "dataset" / "curated" / "attack_dataset_curated.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output" / "model_results"


def slugify(value: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9]+", "_", str(value).strip().lower())
    return re.sub(r"_+", "_", value).strip("_") or "unknown"


def read_dataset(path: Path = DATASET_PATH) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["is_synthetic"] = str(row.get("is_synthetic", "")).strip().lower() in {"true", "1", "yes"}
    return rows


def select_attack_rows(rows: list[dict[str, Any]], attack_type: str, seed: int, limit: int | None) -> list[dict[str, Any]]:
    selected = [row for row in rows if row.get("attack_type") == attack_type]
    rng = random.Random(seed)
    rng.shuffle(selected)
    if limit is not None:
        selected = selected[: max(0, min(limit, len(selected)))]
    return selected


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def is_provider_error(row: dict[str, Any]) -> bool:
    response = str(row.get("model_response", ""))
    return bool(row.get("error")) or any(
        marker in response
        for marker in [
            "Provider Error:",
            "Ollama Error:",
            "Gemini Error:",
            "Groq Error:",
            "Mistral Error:",
            "OpenAI Error:",
            "Cloudflare Error:",
            "GitHub Models Error:",
        ]
    )


def load_resume_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        rows = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    valid_rows = [dict(row) for row in rows if not is_provider_error(dict(row))]
    return sorted(valid_rows, key=lambda row: int(row.get("prompt_index") or 0))


def condition_names(value: str) -> list[str]:
    if value == "both":
        return [NO_DEFENSE, ARCSHIELD_DEFENSE]
    return [value]


def build_run_id(args: argparse.Namespace) -> str:
    if args.run_id:
        return args.run_id
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    return f"single_attack_{timestamp}"


def save_condition_outputs(output_dir: Path, rows: list[dict[str, Any]], config: dict[str, Any]) -> None:
    write_csv(output_dir / "results.csv", rows)
    write_json(output_dir / "results.json", rows)
    write_json(output_dir / "metrics.json", {"overall": compute_judge_metrics(rows), "generated_at_utc": datetime.now(UTC).isoformat()})
    write_json(output_dir / "evaluation_config.json", config)


def run_experiment(args: argparse.Namespace) -> dict[str, Any]:
    load_dotenv()
    if args.list_attack_types:
        for attack_type in ATTACK_TYPES:
            print(attack_type)
        return {"listed_attack_types": True}

    if args.attack_type not in ATTACK_TYPES:
        raise ValueError(f"Unknown attack type: {args.attack_type}. Use --list-attack-types to see valid names.")

    dataset_rows = read_dataset(Path(args.dataset))
    attack_rows = select_attack_rows(dataset_rows, args.attack_type, args.seed, args.limit)
    if not attack_rows:
        raise ValueError(f"No rows found for attack type: {args.attack_type}")

    run_id = build_run_id(args)
    provider, model_name = parse_model_spec(args.model)
    model_slug = slugify(args.model)
    attack_slug = slugify(args.attack_type)
    base_output = Path(args.output_dir) / run_id / model_slug / attack_slug

    client = get_attack_client(args.model, temperature=args.temperature, top_p=args.top_p, max_tokens=args.max_tokens)
    judge = BenchmarkJudge(
        judge_config={
            "llama_guard_model": args.llama_guard_model,
            "timeout": args.llama_guard_timeout,
        }
    )

    combined_rows: list[dict[str, Any]] = []
    config = {
        "run_id": run_id,
        "model_spec": args.model,
        "provider": provider,
        "model_name": model_name,
        "attack_type": args.attack_type,
        "seed": args.seed,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
        "dataset": str(Path(args.dataset).resolve()),
        "limit": args.limit,
        "request_delay": args.request_delay,
        "conditions": condition_names(args.condition),
        "created_at_utc": datetime.now(UTC).isoformat(),
        "judge_metadata": experiment_metadata(
            {
                "run_id": run_id,
                "seed": args.seed,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "max_tokens": args.max_tokens,
                "model_versions": {args.model: model_name},
                "inference_providers": {args.model: provider},
            }
        ),
    }

    for condition_name in condition_names(args.condition):
        condition = get_defense_condition(condition_name)
        condition_rows: list[dict[str, Any]] = []
        condition_output = base_output / condition_name
        completed_indices: set[int] = set()
        if args.resume:
            condition_rows = load_resume_rows(condition_output / "results.json")
            completed_indices = {int(row["prompt_index"]) for row in condition_rows if str(row.get("prompt_index", "")).isdigit()}
            combined_rows.extend(condition_rows)
            if condition_rows:
                print(
                    f"[ArcShield] resume enabled: kept {len(condition_rows)} completed rows "
                    f"for {condition_name}; retrying missing/error rows"
                )
        print(f"[ArcShield] {args.model} | {args.attack_type} | {condition_name} | rows={len(attack_rows)}")

        for index, row in enumerate(attack_rows, start=1):
            if index in completed_indices:
                continue
            prompt_text = row.get("prompt", "")
            execution = execute_attack(
                prompt_text,
                model_spec=args.model,
                system_prompt=condition["system_prompt"],
                client=client,
                temperature=args.temperature,
                top_p=args.top_p,
                max_tokens=args.max_tokens,
                use_moderation=bool(args.use_moderator and condition["defense_enabled"]),
            )
            verdict = judge.judge(
                JudgeInput(
                    prompt=prompt_text,
                    attack_type=args.attack_type,
                    model_name=model_name,
                    model_response=execution["model_response"],
                    defense_enabled=condition["defense_enabled"],
                    defense_layer_outputs={
                        "system_prompt_hardening": condition["system_prompt_hardening"],
                        "moderator_enabled": bool(args.use_moderator and condition["defense_enabled"]),
                    },
                    run_id=run_id,
                    seed=args.seed,
                    timestamp=execution["inference_timestamp"],
                    temperature=args.temperature,
                    top_p=args.top_p,
                    max_tokens=args.max_tokens,
                    is_synthetic=row.get("is_synthetic"),
                    model_version=model_name,
                    inference_provider=provider,
                    defense_config=condition_name,
                    latency_ms=execution["latency_ms"],
                    prompt_tokens=execution["prompt_tokens"],
                    completion_tokens=execution["completion_tokens"],
                    context_tokens=execution["context_tokens"],
                )
            )
            result = verdict.to_dict()
            result.update(
                {
                    **execution,
                    "condition": condition_name,
                    "prompt_index": index,
                    "source_dataset": row.get("source_dataset"),
                    "source_file": row.get("source_file"),
                    "source_row": row.get("source_row"),
                    "raw_label": row.get("raw_label"),
                    "prompt": prompt_text,
                }
            )
            condition_rows.append(result)
            combined_rows.append(result)
            if index % args.save_every == 0 or index == len(attack_rows):
                save_condition_outputs(condition_output, condition_rows, {**config, "condition": condition_name})
                print(f"  saved {index}/{len(attack_rows)} -> {condition_output}")
            if args.request_delay > 0 and index < len(attack_rows):
                time.sleep(args.request_delay)

    combined_output = base_output / "combined"
    write_csv(combined_output / "results.csv", combined_rows)
    write_json(combined_output / "results.json", combined_rows)
    metrics_payload = {
        "overall": compute_judge_metrics(combined_rows),
        "per_condition": compute_judge_metrics(combined_rows, group_by=["defense_config"]),
        "synthetic_vs_real": compute_judge_metrics(combined_rows, group_by=["is_synthetic"]),
        "generated_at_utc": datetime.now(UTC).isoformat(),
    }
    write_json(combined_output / "metrics.json", metrics_payload)
    write_json(combined_output / "evaluation_config.json", config)
    export_annotation_sheet(combined_rows, combined_output / "annotation_sheet.csv", sample_size=min(args.annotation_sample_size, len(combined_rows)), seed=args.seed)
    export_annotation_guidelines(combined_output)

    return {
        "run_id": run_id,
        "rows": len(combined_rows),
        "output_dir": str(base_output),
        "metrics": str(combined_output / "metrics.json"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one ArcShield attack-type benchmark chunk.")
    parser.add_argument("--model", default="ollama:llama3.1:8b", help="Provider-prefixed model, e.g. google:gemini-flash-lite-latest")
    parser.add_argument("--attack-type", default="Prompt Injection", help="One of the 16 curated attack type names.")
    parser.add_argument("--condition", choices=["no_defense", "arcshield", "both"], default="both")
    parser.add_argument("--dataset", default=str(DATASET_PATH))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None, help="Pilot row limit. Omit for all 200 rows in the attack type.")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--llama-guard-model", default="llama-guard3")
    parser.add_argument("--llama-guard-timeout", type=int, default=120)
    parser.add_argument("--annotation-sample-size", type=int, default=100)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--request-delay", type=float, default=0.0, help="Seconds to sleep between target model requests.")
    parser.add_argument("--use-moderator", action="store_true", help="Enable optional response moderator as a defense layer.")
    parser.add_argument("--resume", action="store_true", help="Reuse existing successful rows for the same run id and retry missing/error rows.")
    parser.add_argument("--list-attack-types", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    summary = run_experiment(parse_args())
    print(json.dumps(summary, indent=2))
