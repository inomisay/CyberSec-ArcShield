"""Publication token/context-overhead analysis for the strict article cohort.

All token values are runner-level whitespace-token approximations produced by
src.core.attacker.approximate_token_count; they are not provider-billed tokens.
"""

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
DETAIL_PATH = OUT_DIR / "publication_token_context_overhead.csv"
SUMMARY_PATH = OUT_DIR / "publication_token_context_overhead_summary.csv"
FIGURE_PATH = OUT_DIR / "token_context_overhead_by_model.png"

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
METRIC_COLUMNS = [
    "mean_original_prompt_tokens",
    "mean_system_prompt_tokens",
    "mean_protected_context_tokens",
    "mean_completion_tokens",
    "mean_total_tokens",
    "absolute_context_overhead_tokens",
    "relative_context_overhead_percent",
    "absolute_completion_token_change",
    "relative_completion_token_change_percent",
    "absolute_total_token_change",
    "relative_total_token_change_percent",
]


def safe_relative(change: float, baseline: float) -> float:
    return change / baseline * 100.0 if baseline else float("nan")


def build_detail() -> pd.DataFrame:
    cohort = load_complete_article_cohort(RESULTS_DIR)
    records: list[dict[str, object]] = []

    for model in MODEL_ORDER:
        model_rows = [
            row for row in cohort.attack_rows if row.get("_article_model") == model
        ]
        condition_metrics: dict[str, dict[str, float]] = {}
        for condition in CONDITIONS:
            selected = [
                row
                for row in model_rows
                if condition_from_row(row) == condition and not is_provider_error(row)
            ]
            if not selected:
                raise RuntimeError(f"No valid rows for {model} / {condition}")
            frame = pd.DataFrame(selected)
            required = {
                "prompt_tokens",
                "context_tokens",
                "completion_tokens",
                "context_overhead",
            }
            missing = required - set(frame.columns)
            if missing:
                raise RuntimeError(
                    f"{model} / {condition} is missing token fields: {sorted(missing)}"
                )
            for column in required:
                frame[column] = pd.to_numeric(frame[column], errors="coerce")
            if frame[list(required)].isna().any().any():
                raise RuntimeError(
                    f"{model} / {condition} contains unavailable token fields."
                )

            original = float(frame["prompt_tokens"].mean())
            protected = float(frame["context_tokens"].mean())
            completion = float(frame["completion_tokens"].mean())
            if not np.allclose(
                frame["context_overhead"],
                frame["context_tokens"] - frame["prompt_tokens"],
                rtol=0,
                atol=1e-9,
            ):
                raise RuntimeError(
                    f"{model} / {condition} has inconsistent stored context-overhead fields."
                )
            system = float(frame["context_overhead"].mean())
            total = protected + completion
            condition_metrics[condition] = {
                "valid_n": int(len(frame)),
                "mean_original_prompt_tokens": original,
                "mean_system_prompt_tokens": system,
                "mean_protected_context_tokens": protected,
                "mean_completion_tokens": completion,
                "mean_total_tokens": total,
                "absolute_context_overhead_tokens": system,
                "relative_context_overhead_percent": safe_relative(system, original),
            }

        baseline_completion = condition_metrics["no_defense"]["mean_completion_tokens"]
        baseline_total = condition_metrics["no_defense"]["mean_total_tokens"]
        for condition in CONDITIONS:
            metrics = condition_metrics[condition]
            completion_change = metrics["mean_completion_tokens"] - baseline_completion
            total_change = metrics["mean_total_tokens"] - baseline_total
            records.append(
                {
                    "model": friendly_model_name(model),
                    "deployment": "Local" if model.startswith("ollama_") else "Cloud",
                    "condition": condition,
                    **metrics,
                    "absolute_completion_token_change": completion_change,
                    "relative_completion_token_change_percent": safe_relative(
                        completion_change, baseline_completion
                    ),
                    "absolute_total_token_change": total_change,
                    "relative_total_token_change_percent": safe_relative(
                        total_change, baseline_total
                    ),
                }
            )

    detail = pd.DataFrame(records)
    if len(detail) != 22 or detail.duplicated(["model", "condition"]).any():
        raise RuntimeError("Expected exactly 22 unique model-condition rows.")
    return detail


def build_summary(detail: pd.DataFrame) -> pd.DataFrame:
    groups = [
        ("All models", detail),
        ("Local", detail[detail["deployment"] == "Local"]),
        ("Cloud", detail[detail["deployment"] == "Cloud"]),
    ]
    records: list[dict[str, object]] = []
    for group_name, group in groups:
        for condition in CONDITIONS:
            selected = group[group["condition"] == condition]
            records.append(
                {
                    "aggregation_group": group_name,
                    "configuration_count": int(selected["model"].nunique()),
                    "condition": condition,
                    "aggregation_method": "unweighted mean of model-level runner approximations",
                    "valid_n_total": int(selected["valid_n"].sum()),
                    **{
                        column: float(selected[column].mean())
                        for column in METRIC_COLUMNS
                    },
                }
            )
    return pd.DataFrame(records)


def add_horizontal_labels(ax: plt.Axes, bars, values: list[float]) -> None:
    x_max = ax.get_xlim()[1]
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_width() + x_max * 0.008,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.1f}",
            va="center",
            ha="left",
            fontsize=7.8,
            color="#263238",
        )


def plot(detail: pd.DataFrame) -> None:
    labels = [friendly_model_name(model) for model in MODEL_ORDER]
    no_defense = (
        detail[detail["condition"] == "no_defense"].set_index("model").reindex(labels)
    )
    arcshield = (
        detail[detail["condition"] == "arcshield"].set_index("model").reindex(labels)
    )
    y = np.arange(len(labels))
    height = 0.34

    fig, axes = plt.subplots(1, 2, figsize=(17, 9), sharey=True)

    original = no_defense["mean_original_prompt_tokens"].to_numpy()
    additional = arcshield["mean_system_prompt_tokens"].to_numpy()
    original_bars = axes[0].barh(
        y,
        original,
        color="#4E79A7",
        edgecolor="white",
        label="Original input context",
    )
    axes[0].barh(
        y,
        additional,
        left=original,
        color="#E15759",
        edgecolor="white",
        label="Additional ArcShield context",
    )
    input_totals = original + additional
    axes[0].set_xlim(0, float(input_totals.max()) * 1.18)
    add_horizontal_labels(axes[0], original_bars, original.tolist())
    for index, total in enumerate(input_totals):
        axes[0].text(
            total + axes[0].get_xlim()[1] * 0.008,
            index,
            f"+{additional[index]:.1f}",
            va="center",
            ha="left",
            fontsize=7.8,
            fontweight="bold",
            color="#9B2C2C",
        )

    no_completion = no_defense["mean_completion_tokens"].to_numpy()
    arc_completion = arcshield["mean_completion_tokens"].to_numpy()
    no_bars = axes[1].barh(
        y + height / 2,
        no_completion,
        height,
        color="#9CCBFF",
        edgecolor="white",
        label="Completion: No Defense",
    )
    arc_bars = axes[1].barh(
        y - height / 2,
        arc_completion,
        height,
        color="#F6A6C8",
        edgecolor="white",
        label="Completion: ArcShield",
    )
    axes[1].set_xlim(
        0, float(max(no_completion.max(), arc_completion.max())) * 1.18
    )
    add_horizontal_labels(axes[1], no_bars, no_completion.tolist())
    add_horizontal_labels(axes[1], arc_bars, arc_completion.tolist())

    axes[0].set_yticks(y)
    axes[0].set_yticklabels(labels, fontsize=9)
    axes[0].invert_yaxis()
    axes[0].set_title(
        "(a) Input Context and ArcShield Overhead",
        fontsize=14,
        fontweight="bold",
        color="#264653",
    )
    axes[1].set_title(
        "(b) Mean Completion Tokens",
        fontsize=14,
        fontweight="bold",
        color="#264653",
    )
    for ax in axes:
        ax.set_xlabel("Approximate tokens (whitespace-token proxy)", fontsize=10)
        ax.grid(axis="x", linestyle="--", alpha=0.32)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    handles = [
        plt.Rectangle((0, 0), 1, 1, color="#4E79A7", label="Original input context"),
        plt.Rectangle((0, 0), 1, 1, color="#E15759", label="Additional ArcShield context"),
        plt.Rectangle((0, 0), 1, 1, color="#9CCBFF", label="Completion: No Defense"),
        plt.Rectangle((0, 0), 1, 1, color="#F6A6C8", label="Completion: ArcShield"),
    ]
    fig.suptitle(
        "Token and Context Overhead by Model",
        fontsize=19,
        fontweight="bold",
        color="#264653",
        y=0.98,
    )
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.075),
        ncol=2,
        frameon=True,
        fontsize=9.5,
    )
    fig.text(
        0.5,
        0.025,
        "Token counts are runner-level whitespace-token approximations, not provider-billed tokenizer values. "
        "Additional context is the stored ArcShield system-prompt contribution.",
        ha="center",
        fontsize=9,
        color="#666666",
    )
    fig.subplots_adjust(left=0.19, right=0.98, top=0.91, bottom=0.18, wspace=0.18)
    fig.savefig(FIGURE_PATH, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    detail = build_detail()
    summary = build_summary(detail)
    detail.to_csv(DETAIL_PATH, index=False)
    summary.to_csv(SUMMARY_PATH, index=False)
    plot(detail)

    counts = detail.pivot(index="model", columns="condition", values="valid_n")
    print("VALID COUNTS")
    print(counts.to_string())
    print("\nSUMMARY")
    print(summary.to_string(index=False))
    print(f"\nWrote {DETAIL_PATH}")
    print(f"Wrote {SUMMARY_PATH}")
    print(f"Wrote {FIGURE_PATH}")


if __name__ == "__main__":
    main()
