from pathlib import Path

import pandas as pd

from src.evaluation.generate_model_plots import (
    PLOT_EXCLUDED_MODELS,
    collect_attack_rows,
    collect_benign_rows,
    friendly_model_name,
    pareto_frontier,
)
from src.evaluation.article_cohort import condition_from_row, load_complete_article_cohort


def weighted_mean(frame: pd.DataFrame, value: str, weight: str) -> float:
    valid = frame.dropna(subset=[value, weight])
    total = valid[weight].sum()
    return float((valid[value] * valid[weight]).sum() / total) if total else float("nan")


def main() -> None:
    results_dir = Path("output/model_results")
    out_dir = Path("output/plots/article_plots")
    out_dir.mkdir(parents=True, exist_ok=True)

    cohort = load_complete_article_cohort(results_dir)
    attacks = pd.DataFrame([
        {
            "model": row["_article_model"],
            "attack_type": row.get("attack_type"),
            "condition": condition_from_row(row),
            "attack_success": bool(row.get("attack_success")),
            "refusal_detected": bool(row.get("refusal_detected")),
            "latency_ms": row.get("latency_ms"),
        }
        for row in cohort.attack_rows
    ])
    benign = pd.DataFrame([
        {
            "model": row["_article_model"],
            "condition": condition_from_row(row),
            "refusal_detected": bool(row.get("refusal_detected")),
            "harmful_content_detected": bool(row.get("harmful_content_detected")),
        }
        for row in cohort.benign_rows
    ])
    attacks["deployment"] = attacks["model"].map(lambda value: "Local" if value.startswith("ollama_") else "Cloud")

    rows = []
    for model, group in attacks.groupby("model"):
        conditions = {}
        for condition in ["no_defense", "arcshield"]:
            part = group[group["condition"] == condition]
            conditions[condition] = {
                "asr": float(part["attack_success"].mean()),
                "refusal": float(part["refusal_detected"].mean()),
                "latency": float(part["latency_ms"].mean()),
            }
        model_benign = benign[benign["model"] == model]
        benign_arc = model_benign[model_benign["condition"] == "arcshield"]
        benign_source = benign_arc if not benign_arc.empty else model_benign
        reduction = conditions["no_defense"]["asr"] - conditions["arcshield"]["asr"]
        refused = group[group["refusal_detected"] == True]
        answered = group[group["refusal_detected"] == False]
        rows.append({
            "model": model,
            "model_label": friendly_model_name(model),
            "deployment": "Local" if model.startswith("ollama_") else "Cloud",
            "asr_no_defense": conditions["no_defense"]["asr"],
            "asr_arcshield": conditions["arcshield"]["asr"],
            "asr_reduction_absolute": reduction,
            "asr_reduction_relative": reduction / conditions["no_defense"]["asr"] if conditions["no_defense"]["asr"] else float("nan"),
            "refusal_no_defense": conditions["no_defense"]["refusal"],
            "refusal_arcshield": conditions["arcshield"]["refusal"],
            "benign_refusal_rate": float(benign_source["refusal_detected"].mean()),
            "false_flag_rate": float(benign_source["harmful_content_detected"].mean()),
            "latency_refused_ms": float(refused["latency_ms"].mean()),
            "latency_answered_ms": float(answered["latency_ms"].mean()),
            "latency_overall_ms": float(group["latency_ms"].mean()),
        })
    models = pd.DataFrame(rows)

    for condition in ["no_defense", "arcshield"]:
        condition_rows = attacks[attacks["condition"] == condition]
        points = condition_rows.groupby("model", as_index=False).agg(
            attack_success_rate=("attack_success", "mean"), latency_mean_ms=("latency_ms", "mean")
        )
        frontier_models = set(pareto_frontier(points)["model"])
        models[f"pareto_{condition}"] = models["model"].isin(frontier_models)
    models.to_csv(out_dir / "publication_model_summary.csv", index=False)

    category = attacks.groupby(["condition", "attack_type"], as_index=False).agg(asr=("attack_success", "mean"), n=("attack_success", "size"))
    category.to_csv(out_dir / "publication_attack_category_summary.csv", index=False)
    extremes = []
    for condition, group in category.groupby("condition"):
        for rank_type, ranked in [("highest", group.nlargest(3, "asr")), ("lowest", group.nsmallest(3, "asr"))]:
            for rank, (_, row) in enumerate(ranked.iterrows(), 1):
                extremes.append({"condition": condition, "rank_type": rank_type, "rank": rank, **row.to_dict()})
    pd.DataFrame(extremes).to_csv(out_dir / "publication_attack_category_extremes.csv", index=False)

    by_cell = attacks.groupby(["model", "attack_type", "condition"], as_index=False).agg(asr=("attack_success", "mean"), n=("attack_success", "size"))
    changes = by_cell.pivot(index=["model", "attack_type"], columns="condition", values="asr").reset_index()
    changes["asr_reduction"] = changes["no_defense"] - changes["arcshield"]
    changes["model_label"] = changes["model"].map(friendly_model_name)
    changes.nlargest(5, "asr_reduction").to_csv(out_dir / "publication_largest_asr_reductions.csv", index=False)
    changes[changes["asr_reduction"] < 0].sort_values("asr_reduction").to_csv(out_dir / "publication_negative_asr_reductions.csv", index=False)

    deployment_rows = []
    for (deployment, condition), group in attacks.groupby(["deployment", "condition"]):
        refused = group[group["refusal_detected"] == True]
        answered = group[group["refusal_detected"] == False]
        deployment_rows.append({
            "deployment": deployment,
            "condition": condition,
            "models": group["model"].nunique(),
            "n": len(group),
            "asr": float(group["attack_success"].mean()),
            "refusal_rate": float(group["refusal_detected"].mean()),
            "latency_overall_ms": float(group["latency_ms"].mean()),
            "latency_refused_ms": float(refused["latency_ms"].mean()),
            "latency_answered_ms": float(answered["latency_ms"].mean()),
        })
    deployment = pd.DataFrame(deployment_rows)
    deployment.to_csv(out_dir / "publication_local_cloud_summary.csv", index=False)

    print(models.to_json(orient="records"))
    print("CATEGORY_EXTREMES")
    print(pd.DataFrame(extremes).to_json(orient="records"))
    print("TOP_REDUCTIONS")
    print(changes.nlargest(5, "asr_reduction").to_json(orient="records"))
    print("NEGATIVE_REDUCTIONS")
    print(changes[changes["asr_reduction"] < 0].sort_values("asr_reduction").to_json(orient="records"))
    print("DEPLOYMENT")
    print(deployment.to_json(orient="records"))


if __name__ == "__main__":
    main()
