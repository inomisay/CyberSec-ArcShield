import json
import math
from pathlib import Path

import pandas as pd
from scipy.stats import binomtest

from src.evaluation.generate_model_plots import ATTACK_RESULT_FILE, PLOT_EXCLUDED_MODELS, condition_from_row, friendly_model_name
from src.evaluation.article_cohort import is_provider_error, load_complete_article_cohort


RESULTS_DIR = Path("output/model_results")
OUT_DIR = Path("output/plots/article_plots")
ERROR_PREFIXES = (
    "cloudflare error:", "gemini error:", "groq error:", "mistral error:",
    "openai error:", "ollama error:", "error:",
)


def load_rows() -> pd.DataFrame:
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
            "provider_error": is_provider_error(row),
        })
    return pd.DataFrame(rows)


def paired_rows(rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    keys = ["model", "attack_type", "prompt_index", "source_dataset", "source_row"]
    outcomes = rows.pivot_table(index=keys, columns="condition", values="attack_success", aggfunc="first")
    errors = rows.pivot_table(index=keys, columns="condition", values="provider_error", aggfunc="max")
    joined = outcomes.join(errors, lsuffix="_success", rsuffix="_error").reset_index()
    required = ["no_defense_success", "arcshield_success", "no_defense_error", "arcshield_error"]
    complete = joined.dropna(subset=required).copy()
    incomplete_n = len(joined) - len(complete)
    error_mask = complete["no_defense_error"].astype(bool) | complete["arcshield_error"].astype(bool)
    excluded_errors = complete[error_mask].copy()
    valid = complete[~error_mask].copy()
    return valid, excluded_errors, incomplete_n


def wilson(successes: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    p = successes / n
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return center - half, center + half


def summarize(group: pd.DataFrame) -> dict:
    no_defense = group["no_defense_success"].astype(bool)
    arcshield = group["arcshield_success"].astype(bool)
    n = len(group)
    no_success = int(no_defense.sum())
    arc_success = int(arcshield.sum())
    no_low, no_high = wilson(no_success, n)
    arc_low, arc_high = wilson(arc_success, n)
    # b: success without defense -> failure with ArcShield (improvement).
    # c: failure without defense -> success with ArcShield (regression).
    b = int((no_defense & ~arcshield).sum())
    c = int((~no_defense & arcshield).sum())
    discordant = b + c
    statistic = ((b - c) ** 2 / discordant) if discordant else 0.0
    p_value = float(binomtest(min(b, c), discordant, 0.5, alternative="two-sided").pvalue) if discordant else 1.0
    return {
        "paired_n": n,
        "no_defense_successes": no_success,
        "asr_no_defense": no_success / n if n else float("nan"),
        "asr_no_defense_ci95_low": no_low,
        "asr_no_defense_ci95_high": no_high,
        "arcshield_successes": arc_success,
        "asr_arcshield": arc_success / n if n else float("nan"),
        "asr_arcshield_ci95_low": arc_low,
        "asr_arcshield_ci95_high": arc_high,
        "absolute_asr_change_arc_minus_no_defense": (arc_success - no_success) / n if n else float("nan"),
        "relative_asr_reduction": (no_success - arc_success) / no_success if no_success else float("nan"),
        "mcnemar_b_no_success_arc_failure": b,
        "mcnemar_c_no_failure_arc_success": c,
        "mcnemar_discordant_n": discordant,
        "mcnemar_chi2_statistic_uncorrected": statistic,
        "mcnemar_exact_p_raw": p_value,
        "mcnemar_exact_p_display": "<4.94e-324" if p_value == 0.0 and discordant else f"{p_value:.12g}",
        "paired_risk_difference_arc_minus_no_defense": (arc_success - no_success) / n if n else float("nan"),
        "discordant_odds_ratio_arc_vs_no_defense": (c + 0.5) / (b + 0.5),
    }


def holm_adjust(frame: pd.DataFrame, p_column: str) -> pd.DataFrame:
    result = frame.copy()
    m = len(result)
    order = result[p_column].sort_values().index
    adjusted = pd.Series(index=result.index, dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        candidate = min(1.0, (m - rank) * float(result.loc[index, p_column]))
        running = max(running, candidate)
        adjusted.loc[index] = running
    result["mcnemar_p_holm"] = adjusted
    result["significant_raw_alpha_0_05"] = result[p_column] < 0.05
    result["significant_holm_alpha_0_05"] = result["mcnemar_p_holm"] < 0.05
    return result


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    raw = load_rows()
    paired, excluded_errors, excluded_incomplete_pairs = paired_rows(raw)
    excluded_error_pairs = len(excluded_errors)

    overall = pd.DataFrame([{**summarize(paired),
        "scope": "overall",
        "provider_error_pairs_excluded": excluded_error_pairs,
        "incomplete_pairs_excluded": excluded_incomplete_pairs,
        "ci_method": "Wilson score 95%",
        "mcnemar_method": "Exact two-sided binomial",
    }])
    overall["mcnemar_p_holm"] = overall["mcnemar_exact_p_raw"]
    overall["significant_raw_alpha_0_05"] = overall["mcnemar_exact_p_raw"] < 0.05
    overall["significant_holm_alpha_0_05"] = overall["mcnemar_p_holm"] < 0.05

    model_rows = []
    for model, group in paired.groupby("model"):
        model_rows.append({
            "model": model,
            "model_label": friendly_model_name(model),
            "provider_error_pairs_excluded": int((excluded_errors["model"] == model).sum()),
            **summarize(group),
        })
    models = holm_adjust(pd.DataFrame(model_rows), "mcnemar_exact_p_raw")
    models["provider_error_pairs_excluded_overall"] = excluded_error_pairs
    models["ci_method"] = "Wilson score 95%"
    models["mcnemar_method"] = "Exact two-sided binomial; Holm across models"

    category_rows = []
    for attack_type, group in paired.groupby("attack_type"):
        category_rows.append({
            "attack_type": attack_type,
            "provider_error_pairs_excluded": int((excluded_errors["attack_type"] == attack_type).sum()),
            **summarize(group),
        })
    categories = holm_adjust(pd.DataFrame(category_rows), "mcnemar_exact_p_raw")
    categories["provider_error_pairs_excluded_overall"] = excluded_error_pairs
    categories["ci_method"] = "Wilson score 95%"
    categories["mcnemar_method"] = "Exact two-sided binomial; Holm across categories"

    overall.to_csv(OUT_DIR / "statistical_validation_overall.csv", index=False)
    models.to_csv(OUT_DIR / "statistical_validation_models.csv", index=False)
    categories.to_csv(OUT_DIR / "statistical_validation_categories.csv", index=False)

    print("EXCLUSIONS", {"provider_error_pairs": excluded_error_pairs, "incomplete_pairs": excluded_incomplete_pairs, "paired_n": len(paired)})
    print("OVERALL")
    print(overall.to_json(orient="records"))
    print("MODELS")
    print(models.to_json(orient="records"))
    print("CATEGORIES")
    print(categories.to_json(orient="records"))


if __name__ == "__main__":
    main()
