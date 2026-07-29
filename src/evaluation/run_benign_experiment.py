"""Run ArcShield benign/usability evaluation for one target model.

Examples:
  python -m src.evaluation.run_benign_experiment --model ollama:llama3.1:8b --condition both --limit 12
  python -m src.evaluation.run_benign_experiment --model google:gemini-flash-lite-latest --condition arcshield
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from src.analysis.metrics import benign_confusion_matrix, benign_refusal_rate, security_usability_tradeoff
from src.core.attacker import canonical_output_model_spec, execute_attack, get_attack_client, parse_model_spec
from src.core.defender import ARCSHIELD_DEFENSE, NO_DEFENSE, get_defense_condition
from src.core.judge import BenchmarkJudge, JudgeInput, experiment_metadata
from src.evaluation.benign_dataset import DEFAULT_BENIGN_CSV, balanced_sample, load_benign_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output" / "model_results"


def slugify(value: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9]+", "_", str(value).strip().lower())
    return re.sub(r"_+", "_", value).strip("_") or "unknown"


def condition_names(value: str) -> list[str]:
    return [NO_DEFENSE, ARCSHIELD_DEFENSE] if value == "both" else [value]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def is_provider_error(row: dict[str, Any]) -> bool:
    response = str(row.get("model_response", ""))
    return bool(row.get("error")) or not response.strip() or any(
        marker in response
        for marker in [
            "Provider Error:",
            "Ollama Error:",
            "Gemini Error:",
            "Groq Error:",
            "Mistral Error:",
            "OpenAI Error:",
            "Cloudflare Error:",
        ]
    )


def is_provider_policy_filter(row: dict[str, Any]) -> bool:
    response = str(row.get("model_response", ""))
    error = str(row.get("error", ""))
    text = f"{response}\n{error}".lower()
    return (
        "content management policy" in text
        or "finish_reason=content_filter" in text
        or "provider policy" in text
        or "response was filtered" in text
    )


def provider_error_result(
    execution: dict[str, Any],
    *,
    args: argparse.Namespace,
    model_name: str,
    provider: str,
    condition_name: str,
    condition: dict[str, Any],
    row: dict[str, Any],
    run_id: str,
    index: int,
) -> dict[str, Any]:
    response = str(execution.get("model_response", "")).strip()
    reason = response or str(execution.get("error", "")).strip() or "Provider returned no response."
    return {
        "final_label": "PROVIDER_ERROR",
        "confidence_score": 0.0,
        "refusal_detected": False,
        "harmful_content_detected": False,
        "partial_compliance": False,
        "attack_success": False,
        "defense_success": False,
        "severity_score": "UNKNOWN",
        "judge_reason": f"Provider blocked or failed before the model could produce a usable answer: {reason}",
        "attack_type": f"Benign/{row.get('benign_category', 'unknown')}",
        "model_name": model_name,
        "run_id": run_id,
        "timestamp": execution["inference_timestamp"],
        "defense_enabled": condition["defense_enabled"],
        "is_synthetic": None,
        "evaluator_name": "provider_error_skip",
        "llama_guard_label": None,
        "llama_guard_detected_harm": False,
        "dangerous_rule_match": False,
        "rule_categories": {},
        "safety_category": "provider_error",
        "defense_layer_outputs": {"system_prompt_hardening": condition["system_prompt_hardening"]},
        "seed": args.seed,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
        "latency_ms": execution["latency_ms"],
        "prompt_tokens": execution["prompt_tokens"],
        "completion_tokens": execution["completion_tokens"],
        "context_tokens": execution["context_tokens"],
        "model_version": model_name,
        "inference_provider": provider,
        "defense_config": condition_name,
        **execution,
        "prompt_kind": "benign",
        "benign_category": row.get("benign_category"),
        "benign_label": row.get("label", "benign"),
        "prompt_id": row.get("prompt_id"),
        "prompt": row.get("prompt", ""),
        "condition": condition_name,
        "prompt_index": index,
        "over_refusal": False,
        "valid_attack_blocking": False,
        "overblocking": False,
    }


def is_likely_cutoff(row: dict[str, Any], strict: bool = False) -> bool:
    response = str(row.get("model_response", "")).strip()
    if not response or is_provider_error(row):
        return False

    output_tokens = int(row.get("output_tokens") or row.get("completion_tokens") or 0)
    max_tokens = int(row.get("max_tokens") or 0)
    hit_token_ceiling = max_tokens > 0 and output_tokens >= max_tokens - 2
    unclosed_reasoning = bool(re.search(r"<think>", response, re.IGNORECASE)) and not bool(
        re.search(r"</think>", response, re.IGNORECASE)
    )
    unclosed_code_fence = response.count("```") % 2 == 1
    unfinished_sentence = len(response) > 80 and not re.search(r"[.!?;)\]'\"`]\s*$", response)

    hard_cutoff = hit_token_ceiling or unclosed_reasoning or unclosed_code_fence
    return hard_cutoff if strict else hard_cutoff or unfinished_sentence


def load_resume_rows(path: Path, retry_cutoffs: bool = False, strict_cutoffs: bool = False) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        rows = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    return [
        dict(row)
        for row in rows
        if (not is_provider_error(dict(row)) or is_provider_policy_filter(dict(row)))
        and not ((retry_cutoffs or strict_cutoffs) and is_likely_cutoff(dict(row), strict=strict_cutoffs))
    ]


def benign_metrics(rows: list[dict[str, Any]], group_by: list[str] | None = None) -> list[dict[str, Any]]:
    group_by = group_by or []
    buckets: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        buckets.setdefault(tuple(row.get(field) for field in group_by), []).append(row)

    summaries = []
    for key, group in buckets.items():
        total = len(group)
        refusals = sum(bool(row.get("refusal_detected")) for row in group)
        harmful_false_flags = sum(bool(row.get("attack_success")) for row in group)
        cm = benign_confusion_matrix([bool(row.get("refusal_detected")) for row in group])
        summary = {
            "total_benign_prompts": total,
            "refused_benign_prompts": refusals,
            "answered_benign_prompts": total - refusals,
            "benign_refusal_rate": benign_refusal_rate(refusals, total),
            "benign_refusal_rate_pct": benign_refusal_rate(refusals, total) * 100.0,
            "harmful_false_flags": harmful_false_flags,
            "benign_confusion_matrix": {"tp": cm.tp, "fp": cm.fp, "tn": cm.tn, "fn": cm.fn},
        }
        for idx, field_name in enumerate(group_by):
            summary[field_name] = key[idx]
        summaries.append(summary)
    return summaries


def publication_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# ArcShield Security-Usability Table",
        "",
        "| System | ASR lower is better | DSR higher is better | BRR lower is better |",
        "| ------ | -----: | -----: | -----: |",
    ]
    for row in benign_metrics(rows, ["defense_config"]):
        tradeoff = security_usability_tradeoff(0.0, 0.0, float(row["benign_refusal_rate"]))
        lines.append(f"| {row['defense_config']} | n/a | n/a | {row['benign_refusal_rate']:.4f} |")
        lines.append(
            f"<!-- defense_aggressiveness={tradeoff['defense_aggressiveness']} "
            f"security_utility_score={tradeoff['security_utility_score']:.4f} -->"
        )
    return "\n".join(lines)


def export_brr_chart(rows: list[dict[str, Any]], path: Path) -> str:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return ""
    summaries = benign_metrics(rows, ["defense_config"])
    labels = [str(row["defense_config"]) for row in summaries]
    values = [float(row["benign_refusal_rate_pct"]) for row in summaries]
    fig, ax = plt.subplots(figsize=(6.8, 4.2), dpi=220)
    ax.bar(labels, values, color="#2563eb")
    ax.set_ylabel("BRR (%)")
    ax.set_ylim(0, max(10.0, max(values, default=0.0) + 5.0))
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return str(path)


def run_experiment(args: argparse.Namespace) -> dict[str, Any]:
    load_dotenv()
    rows = balanced_sample(load_benign_dataset(args.dataset), sample_size=args.limit, seed=args.seed)
    run_id = args.run_id or f"benign_eval_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}"
    provider, model_name = parse_model_spec(args.model)
    if args.model_timeout is not None and provider == "ollama":
        os.environ["OLLAMA_TIMEOUT"] = str(args.model_timeout)
    model_slug = slugify(canonical_output_model_spec(args.model))
    run_stamp = run_id.rsplit("_", 2)[-2:]
    dated_eval_slug = f"benign_usability_{'_'.join(run_stamp)}" if len(run_stamp) == 2 else f"benign_usability_{run_id}"
    base_output = Path(args.output_dir) / model_slug / dated_eval_slug

    cloudflare_credentials = getattr(args, "cloudflare_credentials", "model")
    client = get_attack_client(
        args.model,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        cloudflare_credentials=cloudflare_credentials,
    )
    judge = BenchmarkJudge(judge_config={"llama_guard_model": args.llama_guard_model, "timeout": args.llama_guard_timeout})
    all_results: list[dict[str, Any]] = []

    config = {
        "run_id": run_id,
        "model_spec": args.model,
        "provider": provider,
        "cloudflare_credentials": cloudflare_credentials,
        "model_name": model_name,
        "dataset": str(Path(args.dataset).resolve()),
        "seed": args.seed,
        "limit": args.limit,
        "request_delay": args.request_delay,
        "conditions": condition_names(args.condition),
        "evaluation_type": "benign_usability",
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
        condition_output = base_output / condition_name
        condition_rows: list[dict[str, Any]] = []
        completed_indices: set[int] = set()
        if args.resume:
            condition_rows = load_resume_rows(
                condition_output / "benign_results.json",
                retry_cutoffs=args.retry_cutoffs,
                strict_cutoffs=args.retry_hard_cutoffs,
            )
            completed_indices = {int(row["prompt_index"]) for row in condition_rows if str(row.get("prompt_index", "")).isdigit()}
            all_results.extend(condition_rows)
            if condition_rows:
                print(
                    f"[ArcShield Benign] resume enabled: kept {len(condition_rows)} completed rows "
                    f"for {condition_name}; retrying missing/error rows"
                )
        print(f"[ArcShield Benign] {args.model} | {condition_name} | rows={len(rows)}")
        for index, row in enumerate(rows, start=1):
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
            )
            if is_provider_error(execution):
                result = provider_error_result(
                    execution,
                    args=args,
                    model_name=model_name,
                    provider=provider,
                    condition_name=condition_name,
                    condition=condition,
                    row=row,
                    run_id=run_id,
                    index=index,
                )
            else:
                verdict = judge.judge(
                    JudgeInput(
                        prompt=prompt_text,
                        attack_type=f"Benign/{row.get('benign_category', 'unknown')}",
                        model_name=model_name,
                        model_response=execution["model_response"],
                        defense_enabled=condition["defense_enabled"],
                        defense_layer_outputs={"system_prompt_hardening": condition["system_prompt_hardening"]},
                        run_id=run_id,
                        seed=args.seed,
                        timestamp=execution["inference_timestamp"],
                        temperature=args.temperature,
                        top_p=args.top_p,
                        max_tokens=args.max_tokens,
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
                        "prompt_kind": "benign",
                        "benign_category": row.get("benign_category"),
                        "benign_label": row.get("label", "benign"),
                        "prompt_id": row.get("prompt_id"),
                        "prompt": prompt_text,
                        "condition": condition_name,
                        "prompt_index": index,
                        "over_refusal": bool(verdict.refusal_detected),
                        "valid_attack_blocking": False,
                        "overblocking": bool(verdict.refusal_detected),
                    }
                )
            condition_rows.append(result)
            all_results.append(result)
            if index % args.save_every == 0 or index == len(rows):
                write_csv(condition_output / "benign_results.csv", condition_rows)
                write_json(condition_output / "benign_results.json", condition_rows)
                write_json(
                    condition_output / "brr_report.json",
                    {"overall": benign_metrics(condition_rows), "per_category": benign_metrics(condition_rows, ["benign_category"])},
                )
                print(f"  saved {index}/{len(rows)} -> {condition_output}")
            if args.request_delay > 0 and index < len(rows):
                time.sleep(args.request_delay)

    combined_output = base_output / "combined"
    write_csv(combined_output / "benign_results.csv", all_results)
    write_json(combined_output / "benign_results.json", all_results)
    report = {
        "overall": benign_metrics(all_results),
        "per_model": benign_metrics(all_results, ["model_name"]),
        "per_defense_layer": benign_metrics(all_results, ["defense_config"]),
        "per_benign_category": benign_metrics(all_results, ["benign_category"]),
        "security_usability_tradeoff": [
            {"system": row["defense_config"], **security_usability_tradeoff(0.0, 0.0, float(row["benign_refusal_rate"]))}
            for row in benign_metrics(all_results, ["defense_config"])
        ],
        "generated_at_utc": datetime.now(UTC).isoformat(),
    }
    write_json(combined_output / "brr_report.json", report)
    write_json(combined_output / "evaluation_config.json", config)
    (combined_output / "publication_tables.md").write_text(publication_table(all_results), encoding="utf-8")
    chart = export_brr_chart(all_results, combined_output / "brr_chart.png")
    return {
        "run_id": run_id,
        "rows": len(all_results),
        "output_dir": str(base_output),
        "brr_report": str(combined_output / "brr_report.json"),
        "chart": chart,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run benign/usability evaluation and compute BRR.")
    parser.add_argument("--model", default="ollama:llama3.1:8b")
    parser.add_argument("--condition", choices=["no_defense", "arcshield", "both"], default="both")
    parser.add_argument("--dataset", default=str(DEFAULT_BENIGN_CSV))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument(
        "--cloudflare-credentials",
        choices=["model", "primary", "secondary"],
        default="model",
        help=(
            "Cloudflare credential profile, independent of --model and its output slug. "
            "Use 'primary' to resume a cloudflare2 run with the primary account."
        ),
    )
    parser.add_argument("--llama-guard-model", default="llama-guard3")
    parser.add_argument("--llama-guard-timeout", type=int, default=120)
    parser.add_argument(
        "--model-timeout",
        type=float,
        default=None,
        help="Target model request timeout in seconds. Currently used by local Ollama via OLLAMA_TIMEOUT.",
    )
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--request-delay", type=float, default=0.0, help="Seconds to sleep between target model requests.")
    parser.add_argument("--resume", action="store_true", help="Reuse existing successful rows for the same run id and retry missing/error rows.")
    parser.add_argument(
        "--retry-cutoffs",
        action="store_true",
        help="With --resume, also retry rows whose saved response looks truncated or unfinished.",
    )
    parser.add_argument(
        "--retry-hard-cutoffs",
        action="store_true",
        help="With --resume, retry only clearly broken rows, such as token-ceiling hits or unclosed code/reasoning blocks.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run_experiment(parse_args()), indent=2))
