import json
from pathlib import Path

import pandas as pd

from src.evaluation.generate_model_plots import (
    ATTACK_RESULT_FILE,
    PLOT_EXCLUDED_MODELS,
    collect_benign_rows,
    condition_from_row,
    friendly_model_name,
    pareto_frontier,
)
from src.evaluation.article_cohort import load_complete_article_cohort


RESULTS_DIR = Path("output/model_results")
OUT_DIR = Path("output/plots/article_plots")


def load_accepted_attack_rows() -> pd.DataFrame:
    rows = []
    for row in load_complete_article_cohort(RESULTS_DIR).attack_rows:
        rows.append({
            "model": row["_article_model"],
            "attack_type": row.get("attack_type"),
            "condition": condition_from_row(row),
            "prompt_index": row.get("prompt_index"),
            "source_dataset": row.get("source_dataset", ""),
            "source_row": str(row.get("source_row", "")),
            "attack_success": bool(row.get("attack_success")),
            "refusal_detected": bool(row.get("refusal_detected")),
            "latency_ms": row.get("latency_ms"),
            "completion_tokens": row.get("completion_tokens") or row.get("output_tokens") or 0,
        })
    return pd.DataFrame(rows)


def load_benign_model_metrics() -> pd.DataFrame:
    rows = []
    cohort = load_complete_article_cohort(RESULTS_DIR)
    benign = pd.DataFrame([
        {
            "model": row["_article_model"],
            "condition": condition_from_row(row),
            "refusal_detected": bool(row.get("refusal_detected")),
            "harmful_content_detected": bool(row.get("harmful_content_detected")),
        }
        for row in cohort.benign_rows
    ])
    for model in cohort.included_models:
        part = benign[(benign["model"] == model) & (benign["condition"] == "arcshield")]
        if part.empty:
            continue
        rows.append({
            "model": model,
            "benign_refusal_rate": part["refusal_detected"].mean(),
            "false_flag_rate": part["harmful_content_detected"].mean(),
            "benign_n": len(part),
        })
    return pd.DataFrame(rows)


def condition_metrics(attacks: pd.DataFrame) -> pd.DataFrame:
    def summarize(group: pd.DataFrame) -> pd.Series:
        refused = group[group["refusal_detected"]]
        answered = group[~group["refusal_detected"]]
        return pd.Series({
            "n": len(group),
            "asr": group["attack_success"].mean(),
            "refusal_rate": group["refusal_detected"].mean(),
            "latency_overall_ms": group["latency_ms"].mean(),
            "latency_refused_ms": refused["latency_ms"].mean(),
            "latency_answered_ms": answered["latency_ms"].mean(),
            "completion_tokens_mean": group["completion_tokens"].mean(),
            "completion_tokens_refused_mean": refused["completion_tokens"].mean(),
            "completion_tokens_answered_mean": answered["completion_tokens"].mean(),
        })

    return attacks.groupby(["model", "attack_type", "condition"]).apply(summarize, include_groups=False).reset_index()


def wide_cell_table(metrics: pd.DataFrame, benign: pd.DataFrame, pareto: pd.DataFrame) -> pd.DataFrame:
    values = [column for column in metrics.columns if column not in {"model", "attack_type", "condition"}]
    wide = metrics.pivot(index=["model", "attack_type"], columns="condition", values=values)
    wide.columns = [f"{metric}_{condition}" for metric, condition in wide.columns]
    wide = wide.reset_index()
    wide["asr_reduction_absolute"] = wide["asr_no_defense"] - wide["asr_arcshield"]
    wide["asr_reduction_relative"] = wide["asr_reduction_absolute"] / wide["asr_no_defense"].replace(0, pd.NA)
    wide = wide.merge(benign, on="model", how="left")
    memberships = pareto.pivot(index="model", columns="condition", values="is_pareto_frontier").reset_index()
    memberships = memberships.rename(columns={"no_defense": "pareto_no_defense", "arcshield": "pareto_arcshield"})
    wide = wide.merge(memberships, on="model", how="left")
    wide.insert(1, "model_label", wide["model"].map(friendly_model_name))
    return wide


def dominance_table(attacks: pd.DataFrame) -> pd.DataFrame:
    overall = attacks.groupby(["model", "condition"], as_index=False).agg(
        attack_success_rate=("attack_success", "mean"), latency_mean_ms=("latency_ms", "mean")
    )
    rows = []
    for condition, points in overall.groupby("condition"):
        frontier_models = set(pareto_frontier(points)["model"])
        for _, point in points.iterrows():
            dominates = points[
                (points["attack_success_rate"] >= point["attack_success_rate"])
                & (points["latency_mean_ms"] >= point["latency_mean_ms"])
                & ((points["attack_success_rate"] > point["attack_success_rate"]) | (points["latency_mean_ms"] > point["latency_mean_ms"]))
            ]["model"].tolist()
            dominators = points[
                (points["attack_success_rate"] <= point["attack_success_rate"])
                & (points["latency_mean_ms"] <= point["latency_mean_ms"])
                & ((points["attack_success_rate"] < point["attack_success_rate"]) | (points["latency_mean_ms"] < point["latency_mean_ms"]))
            ].sort_values(["attack_success_rate", "latency_mean_ms"])
            example = dominators.iloc[0]["model"] if not dominators.empty else ""
            rows.append({
                "condition": condition,
                "model": point["model"],
                "model_label": friendly_model_name(point["model"]),
                "is_pareto_frontier": point["model"] in frontier_models,
                "attack_success_rate": point["attack_success_rate"],
                "latency_mean_ms": point["latency_mean_ms"],
                "nondominated_objectives": f"ASR={point['attack_success_rate']:.6f}; latency_ms={point['latency_mean_ms']:.3f}" if point["model"] in frontier_models else "",
                "nondominated_reason": "No evaluated model has both ASR and latency less than or equal to this model, with at least one strict improvement." if point["model"] in frontier_models else "",
                "dominates_models": ";".join(friendly_model_name(model) for model in dominates),
                "dominated_by_example": friendly_model_name(example) if example else "",
            })
    return pd.DataFrame(rows)


def reversal_table(attacks: pd.DataFrame, cell_table: pd.DataFrame) -> pd.DataFrame:
    keys = ["model", "attack_type", "prompt_index", "source_dataset", "source_row"]
    paired = attacks.pivot_table(index=keys, columns="condition", values="attack_success", aggfunc="first").reset_index()
    paired = paired.dropna(subset=["no_defense", "arcshield"])
    paired["failed_to_success"] = (~paired["no_defense"].astype(bool)) & paired["arcshield"].astype(bool)
    paired["success_to_failed"] = paired["no_defense"].astype(bool) & (~paired["arcshield"].astype(bool))
    counts = paired.groupby(["model", "attack_type"], as_index=False).agg(
        paired_n=("prompt_index", "size"),
        failed_to_success_n=("failed_to_success", "sum"),
        success_to_failed_n=("success_to_failed", "sum"),
    )
    columns = ["model", "model_label", "attack_type", "asr_no_defense", "asr_arcshield", "asr_reduction_absolute"]
    increased = cell_table[cell_table["asr_arcshield"] > cell_table["asr_no_defense"]][columns]
    return increased.merge(counts, on=["model", "attack_type"], how="left").sort_values("asr_reduction_absolute")


def diagnostic_tables(cell_table: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    columns = [
        "model", "model_label", "attack_type", "n_arcshield", "asr_arcshield", "refusal_rate_arcshield",
        "completion_tokens_mean_arcshield", "completion_tokens_refused_mean_arcshield",
        "completion_tokens_answered_mean_arcshield", "false_flag_rate", "benign_refusal_rate",
        "latency_overall_ms_arcshield", "latency_refused_ms_arcshield", "latency_answered_ms_arcshield",
    ]
    weakest_cells = cell_table.nlargest(10, "asr_arcshield")[columns]
    numeric = [column for column in columns if column not in {"model", "model_label", "attack_type"}]
    model_diagnostics = cell_table.groupby(["model", "model_label"], as_index=False)[numeric].mean()
    model_diagnostics = model_diagnostics.nlargest(5, "asr_arcshield")
    return weakest_cells, model_diagnostics


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    attacks = load_accepted_attack_rows()
    benign = load_benign_model_metrics()
    metrics = condition_metrics(attacks)
    dominance = dominance_table(attacks)
    cells = wide_cell_table(metrics, benign, dominance)
    reversals = reversal_table(attacks, cells)
    weakest_cells, weakest_models = diagnostic_tables(cells)

    cells.to_csv(OUT_DIR / "publication_model_attack_detailed.csv", index=False)
    dominance.to_csv(OUT_DIR / "publication_pareto_dominance.csv", index=False)
    reversals.to_csv(OUT_DIR / "publication_arcshield_asr_increases.csv", index=False)
    weakest_cells.to_csv(OUT_DIR / "publication_weakest_defended_cells.csv", index=False)
    weakest_models.to_csv(OUT_DIR / "publication_weakest_defended_models.csv", index=False)

    print("DOMINANCE")
    print(dominance.to_json(orient="records"))
    print("REVERSALS")
    print(reversals.to_json(orient="records"))
    print("WEAKEST_CELLS")
    print(weakest_cells.to_json(orient="records"))
    print("WEAKEST_MODELS")
    print(weakest_models.to_json(orient="records"))


if __name__ == "__main__":
    main()
