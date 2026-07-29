"""Strict, reproducible cohort selection for final article analyses."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.core.judge import ATTACK_TYPES


RESULTS_DIR = Path("output/model_results")
EXCLUDED_MODEL_DIRS = {
    "cloudflare2_cf_google_gemma_3_12b_it",
    "cloudflare2_cf_moonshotai_kimi_k2_6",
}
ERROR_PREFIXES = (
    "cloudflare error:",
    "gemini error:",
    "groq error:",
    "mistral error:",
    "openai error:",
    "ollama error:",
    "provider error:",
)


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def condition_from_row(row: dict[str, Any]) -> str:
    configured = str(row.get("defense_config") or row.get("condition") or "").strip()
    if configured in {"no_defense", "arcshield"}:
        return configured
    return "arcshield" if bool(row.get("defense_enabled")) else "no_defense"


def is_provider_error(row: dict[str, Any]) -> bool:
    if row.get("final_label") == "PROVIDER_ERROR":
        return True
    if str(row.get("error") or "").strip():
        return True
    response = str(row.get("model_response") or "").strip().lower()
    return response.startswith(ERROR_PREFIXES)


def valid_metadata(row: dict[str, Any]) -> bool:
    model = str(row.get("model_name") or "").strip()
    provider = str(row.get("inference_provider") or row.get("provider") or "").strip()
    return bool(model and provider)


@dataclass
class CompleteCohort:
    report_rows: list[dict[str, Any]]
    attack_rows: list[dict[str, Any]]
    benign_rows: list[dict[str, Any]]
    excluded_models: list[dict[str, Any]]

    @property
    def included_models(self) -> list[str]:
        return [row["model"] for row in self.report_rows]


def _complete_attack_runs(model_dir: Path) -> tuple[dict[str, list[dict[str, Any]]], list[str]]:
    accepted: dict[str, list[dict[str, Any]]] = {}
    rejected_runs: list[str] = []
    for run_dir in sorted(model_dir.iterdir()):
        if not run_dir.is_dir() or run_dir.name.startswith("benign_"):
            continue
        no_defense = load_json(run_dir / "no_defense" / "results.json")
        arcshield = load_json(run_dir / "arcshield" / "results.json")
        if not isinstance(no_defense, list) or not isinstance(arcshield, list):
            rejected_runs.append(f"{run_dir.name}: missing condition JSON")
            continue
        if len(no_defense) != 200 or len(arcshield) != 200:
            rejected_runs.append(
                f"{run_dir.name}: no_defense={len(no_defense)}, arcshield={len(arcshield)}"
            )
            continue
        data = no_defense + arcshield
        attack_types = {str(row.get("attack_type") or "").strip() for row in data}
        if len(attack_types) != 1 or "" in attack_types:
            rejected_runs.append(f"{run_dir.name}: invalid attack_type metadata")
            continue
        attack_type = next(iter(attack_types))
        if attack_type not in ATTACK_TYPES:
            rejected_runs.append(f"{run_dir.name}: unknown attack type {attack_type!r}")
            continue
        if attack_type in accepted:
            rejected_runs.append(f"{run_dir.name}: duplicate complete run for {attack_type}")
            accepted.pop(attack_type, None)
            continue
        accepted[attack_type] = data
    return accepted, rejected_runs


def _complete_benign_runs(model_dir: Path) -> tuple[list[dict[str, Any]] | None, list[str]]:
    accepted: list[list[dict[str, Any]]] = []
    rejected: list[str] = []
    for run_dir in sorted(model_dir.iterdir()):
        if not run_dir.is_dir() or not run_dir.name.startswith("benign_"):
            continue
        no_defense = load_json(run_dir / "no_defense" / "benign_results.json")
        arcshield = load_json(run_dir / "arcshield" / "benign_results.json")
        if not isinstance(no_defense, list) or not isinstance(arcshield, list):
            rejected.append(f"{run_dir.name}: missing benign condition JSON")
            continue
        if len(no_defense) != 200 or len(arcshield) != 200:
            rejected.append(
                f"{run_dir.name}: no_defense={len(no_defense)}, arcshield={len(arcshield)}"
            )
            continue
        accepted.append(no_defense + arcshield)
    if len(accepted) != 1:
        rejected.append(f"expected one complete benign run, found {len(accepted)}")
        return None, rejected
    return accepted[0], rejected


def load_complete_article_cohort(results_dir: Path = RESULTS_DIR) -> CompleteCohort:
    report_rows: list[dict[str, Any]] = []
    attack_rows: list[dict[str, Any]] = []
    benign_rows: list[dict[str, Any]] = []
    excluded_models: list[dict[str, Any]] = []
    expected = set(ATTACK_TYPES)

    for model_dir in sorted(results_dir.iterdir()):
        if (
            not model_dir.is_dir()
            or model_dir.name.startswith("github_")
            or model_dir.name in EXCLUDED_MODEL_DIRS
        ):
            continue
        attacks, rejected_attack_runs = _complete_attack_runs(model_dir)
        benign, rejected_benign_runs = _complete_benign_runs(model_dir)
        reasons: list[str] = []
        if set(attacks) != expected:
            missing = sorted(expected - set(attacks))
            extra = sorted(set(attacks) - expected)
            reasons.append(f"attack categories mismatch; missing={missing}, extra={extra}")
        if benign is None:
            reasons.append("no unique complete benign evaluation")

        selected_attacks = [row for attack_type in ATTACK_TYPES for row in attacks.get(attack_type, [])]
        selected_benign = benign or []
        invalid_metadata = sum(
            not valid_metadata(row) for row in selected_attacks + selected_benign
        )
        if invalid_metadata:
            reasons.append(f"{invalid_metadata} rows have invalid model/provider metadata")
        if reasons:
            excluded_models.append(
                {
                    "model": model_dir.name,
                    "reasons": reasons,
                    "rejected_attack_runs": rejected_attack_runs,
                    "rejected_benign_runs": rejected_benign_runs,
                }
            )
            continue

        for row in selected_attacks:
            row = dict(row)
            row["_article_model"] = model_dir.name
            attack_rows.append(row)
        for row in selected_benign:
            row = dict(row)
            row["_article_model"] = model_dir.name
            benign_rows.append(row)

        attack_errors = sum(is_provider_error(row) for row in selected_attacks)
        benign_errors = sum(is_provider_error(row) for row in selected_benign)
        report_rows.append(
            {
                "model": model_dir.name,
                "deployment": "Local" if model_dir.name.startswith("ollama_") else "Cloud",
                "completed_attack_categories": len(attacks),
                "adversarial_no_defense_rows": sum(
                    condition_from_row(row) == "no_defense" for row in selected_attacks
                ),
                "adversarial_arcshield_rows": sum(
                    condition_from_row(row) == "arcshield" for row in selected_attacks
                ),
                "benign_no_defense_rows": sum(
                    condition_from_row(row) == "no_defense" for row in selected_benign
                ),
                "benign_arcshield_rows": sum(
                    condition_from_row(row) == "arcshield" for row in selected_benign
                ),
                "adversarial_provider_errors": attack_errors,
                "benign_provider_errors": benign_errors,
                "provider_errors_total": attack_errors + benign_errors,
                "rejected_incomplete_pilot_runs": len(rejected_attack_runs),
            }
        )
    return CompleteCohort(report_rows, attack_rows, benign_rows, excluded_models)


def write_completeness_report(
    cohort: CompleteCohort,
    output_dir: Path = Path("output/plots/article_plots"),
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    adversarial = len(cohort.attack_rows)
    benign = len(cohort.benign_rows)
    payload = {
        "included_models": cohort.report_rows,
        "excluded_models": cohort.excluded_models,
        "final_included_model_count": len(cohort.report_rows),
        "final_adversarial_observation_count": adversarial,
        "final_benign_observation_count": benign,
        "final_total_observation_count": adversarial + benign,
        "contains_eleven_complete_model_configurations": len(cohort.report_rows) == 11,
    }
    (output_dir / "article_completeness_report.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    return payload


if __name__ == "__main__":
    report = write_completeness_report(load_complete_article_cohort())
    print(json.dumps(report, indent=2))
