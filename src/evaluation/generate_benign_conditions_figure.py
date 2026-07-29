"""Generate the condition-separated benign refusal and false-flag figure."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.evaluation.article_cohort import (
    condition_from_row,
    is_provider_error,
    load_complete_article_cohort,
)
from src.evaluation.generate_model_plots import friendly_model_name


RESULTS_DIR = Path("output/model_results")
OUT_DIR = Path("output/plots/article_plots")
CSV_PATH = OUT_DIR / "benign_refusal_false_flags_conditions.csv"
FIGURE_PATH = OUT_DIR / "benign_refusal_false_flags_conditions.png"

MODEL_ORDER = [
    "cloudflare_cf_deepseek_ai_deepseek_r1_distill_qwen_32b",
    "cloudflare_cf_google_gemma_4_26b_a4b_it",
    "cloudflare_cf_qwen_qwen3_30b_a3b_fp8",
    "google_gemini_flash_lite_latest",
    "groq_llama_3_1_8b_instant",
    "mistral_mistral_small_latest",
    "ollama_deepseek_r1_latest",
    "ollama_gemma4_latest",
    "ollama_llama3_1_8b",
    "ollama_mistral_latest",
    "ollama_qwen3_latest",
]
CONDITIONS = ["no_defense", "arcshield"]
BRR_COLOR = "#59A14F"
FFR_COLOR = "#F28E2B"


def build_summary() -> tuple[pd.DataFrame, pd.DataFrame]:
    cohort = load_complete_article_cohort(RESULTS_DIR)
    records: list[dict[str, object]] = []
    errors: list[dict[str, object]] = []

    for model in MODEL_ORDER:
        model_rows = [
            row for row in cohort.benign_rows if row.get("_article_model") == model
        ]
        for condition in CONDITIONS:
            condition_rows = [
                row for row in model_rows if condition_from_row(row) == condition
            ]
            error_rows = [row for row in condition_rows if is_provider_error(row)]
            valid_rows = [row for row in condition_rows if not is_provider_error(row)]
            refused = sum(bool(row.get("refusal_detected")) for row in valid_rows)
            false_flags = sum(
                bool(row.get("harmful_content_detected")) for row in valid_rows
            )
            valid_n = len(valid_rows)

            records.append(
                {
                    "model": friendly_model_name(model),
                    "deployment": "Local" if model.startswith("ollama_") else "Cloud",
                    "condition": condition,
                    "valid_benign_n": valid_n,
                    "benign_refusal_count": refused,
                    "benign_refusal_rate": refused / valid_n if valid_n else np.nan,
                    "false_flag_count": false_flags,
                    "false_flag_rate": false_flags / valid_n if valid_n else np.nan,
                }
            )
            if error_rows:
                errors.append(
                    {
                        "model": friendly_model_name(model),
                        "condition": condition,
                        "provider_error_n": len(error_rows),
                    }
                )

    summary = pd.DataFrame(records)
    documented_errors = pd.DataFrame(errors)

    expected_cells = len(MODEL_ORDER) * len(CONDITIONS)
    if len(summary) != expected_cells:
        raise RuntimeError(f"Expected {expected_cells} cells, found {len(summary)}")
    invalid_counts = summary[summary["valid_benign_n"] != 200]
    documented_keys = {
        (row["model"], row["condition"])
        for row in documented_errors.to_dict(orient="records")
    }
    undocumented = invalid_counts[
        ~invalid_counts.apply(
            lambda row: (row["model"], row["condition"]) in documented_keys, axis=1
        )
    ]
    if not undocumented.empty:
        raise RuntimeError(
            "Benign completeness validation failed:\n"
            + undocumented.to_string(index=False)
        )
    return summary, documented_errors


def add_percentage_labels(ax: plt.Axes, bars) -> None:
    for bar in bars:
        height = float(bar.get_height())
        ax.annotate(
            f"{height * 100:.1f}%",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="semibold",
            color="#263238",
        )


def plot(summary: pd.DataFrame) -> None:
    labels = [friendly_model_name(model) for model in MODEL_ORDER]
    x = np.arange(len(labels))
    width = 0.37
    fig, axes = plt.subplots(1, 2, figsize=(25, 10), sharey=True)

    for ax, condition, panel_label in zip(
        axes,
        CONDITIONS,
        ["(a) No Defense", "(b) ArcShield"],
    ):
        condition_data = (
            summary[summary["condition"] == condition]
            .set_index("model")
            .reindex(labels)
        )
        brr = ax.bar(
            x - width / 2,
            condition_data["benign_refusal_rate"],
            width,
            color=BRR_COLOR,
            edgecolor="white",
            linewidth=0.8,
            label="Benign Refusal Rate (BRR)",
        )
        ffr = ax.bar(
            x + width / 2,
            condition_data["false_flag_rate"],
            width,
            color=FFR_COLOR,
            edgecolor="white",
            linewidth=0.8,
            label="False-Flag Rate (FFR)",
        )
        add_percentage_labels(ax, brr)
        add_percentage_labels(ax, ffr)
        ax.set_ylim(0, 1)
        ax.set_yticks(np.linspace(0, 1, 6))
        ax.set_yticklabels([f"{value:.0%}" for value in np.linspace(0, 1, 6)])
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=38, ha="right", fontsize=9)
        ax.tick_params(axis="y", labelsize=10)
        ax.grid(axis="y", linestyle="--", alpha=0.35, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.text(
            0.015,
            0.965,
            panel_label,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=15,
            fontweight="bold",
            color="#264653",
        )

    axes[0].set_ylabel("Rate", fontsize=12, fontweight="semibold")
    fig.suptitle(
        "Benign Refusal and False-Flag Rates Across Defense Conditions",
        fontsize=20,
        fontweight="bold",
        color="#264653",
        y=0.985,
    )
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.015),
        ncol=2,
        frameon=True,
        fontsize=11,
    )
    fig.subplots_adjust(left=0.055, right=0.99, top=0.91, bottom=0.27, wspace=0.08)
    fig.savefig(FIGURE_PATH, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary, errors = build_summary()
    summary.to_csv(CSV_PATH, index=False)
    plot(summary)
    print(summary.to_string(index=False))
    if errors.empty:
        print("VALIDATION: all 22 model-condition cells contain 200 valid benign rows.")
    else:
        print("DOCUMENTED PROVIDER ERRORS:")
        print(errors.to_string(index=False))
    print(f"Wrote {CSV_PATH}")
    print(f"Wrote {FIGURE_PATH}")


if __name__ == "__main__":
    main()
