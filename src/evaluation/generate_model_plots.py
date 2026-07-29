import argparse
import json
import re
import shutil
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd
import seaborn as sns


ATTACK_RESULT_FILE = "results.json"
BENIGN_RESULT_FILE = "benign_results.json"
CONDITION_PALETTE = {
    "no_defense": "#F6A6C8",
    "arcshield": "#9CCBFF",
}
MODEL_COLORS = ["#F6A6C8", "#9CCBFF", "#C7B9FF", "#A7E8C5", "#FFD6A5", "#B8E0D2", "#FFCAD4", "#BDE0FE"]
STRATEGIC_MODEL_COLORS = [
    "#4E79A7",
    "#F28E2B",
    "#59A14F",
    "#E15759",
    "#B07AA1",
    "#76B7B2",
    "#EDC948",
    "#FF9DA7",
    "#9C755F",
    "#86BCB6",
    "#7F7F7F",
    "#6B5B95",
    "#2F4B7C",
    "#D45087",
    "#00A6A6",
    "#A05195",
]
METRIC_PALETTE = {
    "attack_success_rate": "#F6A6C8",
    "refusal_rate": "#9CCBFF",
    "partial_compliance_rate": "#C7B9FF",
    "benign_refusal_rate": "#A7E8C5",
    "harmful_false_flag_rate": "#FFD6A5",
}
TITLE_COLOR = "#264653"
AXIS_COLOR = "#264653"


def percent_formatter() -> FuncFormatter:
    return FuncFormatter(lambda value, _: f"{value * 100:.0f}")


def display_condition(value: str) -> str:
    return {
        "no_defense": "No Defense (Baseline)",
        "arcshield": "With ArcShield Defense",
    }.get(value, value)


def beautify_axes(title: str, ylabel: str, ylim: tuple[float, float] | None = (0, 1), percent: bool = True) -> None:
    ax = plt.gca()
    ax.set_title(title, fontsize=24, fontweight="bold", color=TITLE_COLOR, pad=46)
    ax.set_xlabel("")
    ax.set_ylabel(ylabel, fontsize=15, color=AXIS_COLOR, labelpad=10)
    if ylim:
        ax.set_ylim(*ylim)
    if percent:
        ax.yaxis.set_major_formatter(percent_formatter())
    ax.grid(axis="y", linestyle="--", linewidth=1.1, alpha=0.5)
    ax.grid(axis="x", linestyle=":", linewidth=0.8, alpha=0.18)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#CFCFCF")
    ax.spines["bottom"].set_color("#CFCFCF")
    ax.tick_params(axis="x", labelsize=12, colors="#444444")
    ax.tick_params(axis="y", labelsize=12, colors="#444444")


def apply_arcshield_hatching() -> None:
    ax = plt.gca()
    arcshield_rgba = to_rgba(CONDITION_PALETTE["arcshield"])
    for patch in ax.patches:
        face = patch.get_facecolor()
        is_arcshield = all(abs(face[i] - arcshield_rgba[i]) <= 0.05 for i in range(3))
        if is_arcshield:
            patch.set_hatch("//")
            patch.set_edgecolor("#EAF4FF")
            patch.set_linewidth(1.2)
            patch.set_alpha(0.9)
        else:
            patch.set_alpha(0.95)


def annotate_bars(percent: bool = True, offset: float | None = None) -> None:
    ax = plt.gca()
    ymin, ymax = ax.get_ylim()
    span = ymax - ymin
    for patch in ax.patches:
        height = patch.get_height()
        if pd.isna(height):
            continue
        if percent and abs(height) < 0.005:
            continue
        if not percent and abs(height) < 0.05:
            continue
        label = f"{height * 100:.1f}%" if percent else f"{height:.1f}"
        x = patch.get_x() + patch.get_width() / 2
        y = height
        offset = (0.018 if percent else 0.03) if offset is None else offset
        va = "bottom"
        if height < 0:
            if offset is None:
                offset = -(0.035 if percent else span * 0.025)
            else:
                offset = -abs(offset)
            va = "top"
        ax.text(
            x,
            y + offset,
            label,
            ha="center",
            va=va,
            fontsize=10,
            fontweight="bold",
            color="#000000",
        )


def bottom_condition_legend(y_anchor: float = 0.035) -> None:
    ax = plt.gca()
    if ax.get_legend() is not None:
        ax.get_legend().remove()
    handles = [
        Patch(facecolor=CONDITION_PALETTE["no_defense"], label=display_condition("no_defense")),
        Patch(
            facecolor=CONDITION_PALETTE["arcshield"],
            hatch="//",
            edgecolor="#EAF4FF",
            label=display_condition("arcshield"),
        ),
    ]
    fig = plt.gcf()
    legend = fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, y_anchor),
        ncol=2,
        frameon=True,
        fancybox=True,
        shadow=True,
        fontsize=12,
    )
    legend.get_frame().set_edgecolor("#BFBFBF")
    legend.get_frame().set_linewidth(1.0)


def pretty_name(value: str) -> str:
    value = re.sub(r"_\d{8}_\d{6}$", "", value)
    return value.replace("_", " ").title()


# Temporarily leave these Cloudflare models out of generated plots.
PLOT_EXCLUDED_MODELS = {
    "cloudflare2_cf_google_gemma_3_12b_it",
    "cloudflare2_cf_moonshotai_kimi_k2_6",
}


def friendly_model_name(model_name: str) -> str:
    replacements = {
        "ollama_llama3_1_8b": "OLLAMA LLAMA3.1",
        "ollama_qwen3_latest": "OLLAMA QWEN3",
        "ollama_mistral_latest": "OLLAMA MISTRAL",
        "ollama_deepseek_r1_latest": "OLLAMA DEEPSEEK R1",
        "ollama_gemma4_latest": "OLLAMA GEMMA 4",
        "google_gemini_flash_lite_latest": "GEMINI",
        "groq_llama_3_1_8b_instant": "GROQ LLAMA3.1",
        "mistral_mistral_small_latest": "MISTRAL",
        "cloudflare_cf_qwen_qwen3_30b_a3b_fp8": "CLOUDFLARE QWEN",
        "cloudflare_cf_deepseek_ai_deepseek_r1_distill_qwen_32b": "CLOUDFLARE DEEPSEEK",
        "cloudflare_cf_google_gemma_4_26b_a4b_it": "CLOUDFLARE GEMMA 4",
    }
    return replacements.get(model_name, model_name.replace("_", " ").upper())


def compact_model_name(model_name: str) -> str:
    replacements = {
        "ollama_llama3_1_8b": "Ollama\nLlama3.1",
        "ollama_qwen3_latest": "Ollama\nQwen3",
        "ollama_mistral_latest": "Ollama\nMistral",
        "google_gemini_flash_lite_latest": "Gemini",
        "groq_llama_3_1_8b_instant": "Groq\nLlama3.1",
        "mistral_mistral_small_latest": "Mistral",
        "cloudflare_cf_qwen_qwen3_30b_a3b_fp8": "Cloudflare\nQwen",
        "cloudflare_cf_deepseek_ai_deepseek_r1_distill_qwen_32b": "Cloudflare\nDeepSeek",
        "cloudflare_cf_google_gemma_4_26b_a4b_it": "Cloudflare\nGemma 4",
    }
    return replacements.get(model_name, model_name.replace("_", "\n"))


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def condition_from_row(row: dict[str, Any]) -> str:
    return "arcshield" if bool(row.get("defense_enabled")) else "no_defense"


def collect_attack_rows(model_dir: Path) -> tuple[pd.DataFrame, list[str]]:
    rows: list[dict[str, Any]] = []
    skipped: list[str] = []

    for run_dir in sorted(model_dir.iterdir()):
        if not run_dir.is_dir() or run_dir.name.startswith("benign_"):
            continue

        # Condition files are authoritative. A combined file can be stale after
        # a resumed condition was repaired, even when it still contains 400 rows.
        condition_rows: list[dict[str, Any]] = []
        condition_counts: list[str] = []
        for condition in ("no_defense", "arcshield"):
            condition_path = run_dir / condition / ATTACK_RESULT_FILE
            condition_data = load_json(condition_path) if condition_path.exists() else None
            count = len(condition_data) if isinstance(condition_data, list) else 0
            condition_counts.append(f"{condition}={count}")
            if isinstance(condition_data, list):
                condition_rows.extend(condition_data)
        if all(part.endswith("=200") for part in condition_counts):
            data = condition_rows
        else:
            combined_path = run_dir / "combined" / ATTACK_RESULT_FILE
            combined_data = load_json(combined_path) if combined_path.exists() else None
            combined_count = len(combined_data) if isinstance(combined_data, list) else 0
            skipped.append(
                f"{run_dir.name}: incomplete rows "
                f"(combined={combined_count}, {', '.join(condition_counts)})"
            )
            continue

        for row in data:
            attack_type = row.get("attack_type") or pretty_name(run_dir.name)
            rows.append(
                {
                    "run": run_dir.name,
                    "attack_type": attack_type,
                    "condition": condition_from_row(row),
                    "attack_success": bool(row.get("attack_success")),
                    "defense_success": bool(row.get("defense_success")),
                    "refusal_detected": bool(row.get("refusal_detected")),
                    "partial_compliance": bool(row.get("partial_compliance")),
                    "harmful_content_detected": bool(row.get("harmful_content_detected")),
                    "latency_ms": row.get("latency_ms"),
                    "final_label": row.get("final_label", ""),
                    "severity_score": row.get("severity_score", ""),
                }
            )

    return pd.DataFrame(rows), skipped


def collect_benign_rows(model_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    result_rows: list[dict[str, Any]] = []
    category_rows: list[dict[str, Any]] = []

    for run_dir in sorted(model_dir.iterdir()):
        if not run_dir.is_dir() or not run_dir.name.startswith("benign_"):
            continue

        report_path = run_dir / "combined" / "brr_report.json"

        condition_data: list[dict[str, Any]] = []
        for condition in ("no_defense", "arcshield"):
            path = run_dir / condition / BENIGN_RESULT_FILE
            data = load_json(path) if path.exists() else None
            if isinstance(data, list) and len(data) == 200:
                condition_data.extend(data)
        if len(condition_data) == 400:
            for row in condition_data:
                    result_rows.append(
                        {
                            "run": run_dir.name,
                            "condition": condition_from_row(row),
                            "refusal_detected": bool(row.get("refusal_detected")),
                            "harmful_content_detected": bool(row.get("harmful_content_detected")),
                            "benign_category": row.get("benign_category") or row.get("category") or "unknown",
                        }
                    )

        if report_path.exists():
            report = load_json(report_path)
            for row in report.get("per_benign_category", []):
                category_rows.append(
                    {
                        "run": run_dir.name,
                        "benign_category": row.get("benign_category", "unknown"),
                        "benign_refusal_rate": row.get("benign_refusal_rate", 0.0),
                        "harmful_false_flags": row.get("harmful_false_flags", 0),
                        "total_benign_prompts": row.get("total_benign_prompts", 0),
                    }
                )

    return pd.DataFrame(result_rows), pd.DataFrame(category_rows)


def save_barplot(
    data: pd.DataFrame,
    x: str,
    y: str,
    path: Path,
    title: str,
    hue: str | None = None,
    ylabel: str = "Rate",
    ylim: tuple[float, float] | None = (0, 1),
    legend_y_anchor: float = 0.035,
    bottom_adjust: float = 0.30,
    top_adjust: float = 0.82,
) -> None:
    plt.figure(figsize=(max(13, len(data[x].unique()) * 1.08), 8.2))
    palette = CONDITION_PALETTE if hue == "condition" else None
    if hue == "metric":
        palette = METRIC_PALETTE
    hue_order = ["no_defense", "arcshield"] if hue == "condition" else None
    sns.barplot(data=data, x=x, y=y, hue=hue, hue_order=hue_order, errorbar=None, palette=palette)
    beautify_axes(title, ylabel, ylim=ylim, percent=ylim == (0, 1))
    if hue == "condition":
        apply_arcshield_hatching()
        annotate_bars(percent=ylim == (0, 1))
        bottom_condition_legend(y_anchor=legend_y_anchor)
    elif ylim == (0, 1):
        annotate_bars(percent=True)
    plt.xticks(rotation=38, ha="right")
    plt.subplots_adjust(left=0.08, right=0.98, top=top_adjust, bottom=bottom_adjust)
    plt.savefig(path, dpi=180)
    plt.close()


def save_tradeoff_plot(
    model_dir: Path,
    out_dir: Path,
    attack_df: pd.DataFrame,
    benign_df: pd.DataFrame,
) -> None:
    if attack_df.empty or benign_df.empty:
        return

    attack_summary = (
        attack_df.groupby("condition", as_index=False)
        .agg(
            attack_success_rate=("attack_success", "mean"),
            refusal_rate=("refusal_detected", "mean"),
            partial_compliance_rate=("partial_compliance", "mean"),
        )
    )
    benign_summary = (
        benign_df.groupby("condition", as_index=False)
        .agg(
            benign_refusal_rate=("refusal_detected", "mean"),
            harmful_false_flag_rate=("harmful_content_detected", "mean"),
        )
    )

    tradeoff = attack_summary.merge(benign_summary, on="condition", how="outer").fillna(0.0)
    tradeoff.to_csv(out_dir / "security_benign_tradeoff.csv", index=False)

    melted = tradeoff.melt(
        id_vars=["condition"],
        value_vars=[
            "attack_success_rate",
            "partial_compliance_rate",
            "refusal_rate",
            "benign_refusal_rate",
            "harmful_false_flag_rate",
        ],
        var_name="metric",
        value_name="rate",
    )
    metric_labels = {
        "attack_success_rate": "Attack success",
        "partial_compliance_rate": "Partial compliance",
        "refusal_rate": "Attack refusal",
        "benign_refusal_rate": "Benign refusal",
        "harmful_false_flag_rate": "Benign false flag",
    }
    melted["metric_label"] = melted["metric"].map(metric_labels)

    plt.figure(figsize=(11, 7.8))
    sns.barplot(
        data=melted,
        x="metric_label",
        y="rate",
        hue="condition",
        errorbar=None,
        palette=CONDITION_PALETTE,
    )
    beautify_axes(f"{model_dir.name}: Why Benign Checks Matter", "Rate (%)", ylim=(0, 1), percent=True)
    apply_arcshield_hatching()
    annotate_bars(percent=True)
    bottom_condition_legend(y_anchor=0.035)
    plt.xticks(rotation=25, ha="right")
    plt.subplots_adjust(left=0.08, right=0.98, top=0.82, bottom=0.30)
    plt.savefig(out_dir / "security_vs_benign_tradeoff.png", dpi=180)
    plt.close()

    no_defense = tradeoff[tradeoff["condition"] == "no_defense"]
    arcshield = tradeoff[tradeoff["condition"] == "arcshield"]
    if not no_defense.empty and not arcshield.empty:
        delta = pd.DataFrame(
            [
                {
                    "metric": "Attack success reduction",
                    "value": float(no_defense["attack_success_rate"].iloc[0] - arcshield["attack_success_rate"].iloc[0]),
                },
                {
                    "metric": "Benign refusal change",
                    "value": float(arcshield["benign_refusal_rate"].iloc[0] - no_defense["benign_refusal_rate"].iloc[0]),
                },
                {
                    "metric": "Benign false-flag change",
                    "value": float(
                        arcshield["harmful_false_flag_rate"].iloc[0]
                        - no_defense["harmful_false_flag_rate"].iloc[0]
                    ),
                },
            ]
        )
        delta.to_csv(out_dir / "arcshield_security_usability_delta.csv", index=False)
        delta["metric_label"] = delta["metric"].replace(
            {
                "Attack success reduction": "Attack Success\nReduction",
                "Benign refusal change": "Benign Refusal\nChange",
                "Benign false-flag change": "Benign False-Flag\nChange",
            }
        )
        colors = [CONDITION_PALETTE["arcshield"] if value >= 0 else CONDITION_PALETTE["no_defense"] for value in delta["value"]]
        plt.figure(figsize=(10.5, 6.8))
        sns.barplot(data=delta, x="metric_label", y="value", hue="metric_label", palette=colors, errorbar=None, legend=False)
        ax = plt.gca()
        for patch, value in zip(ax.patches, delta["value"]):
            if float(value) >= 0:
                patch.set_hatch("//")
                patch.set_edgecolor("#EAF4FF")
                patch.set_linewidth(1.2)
        plt.axhline(0, color="#333333", linewidth=1)
        beautify_axes(f"{model_dir.name}: Security Gain vs Benign Cost", "Rate Change (%)", ylim=None, percent=True)
        annotate_bars(percent=True, offset=0.003)
        plt.xticks(rotation=0, ha="center")
        plt.subplots_adjust(left=0.10, right=0.98, top=0.76, bottom=0.20)
        plt.savefig(out_dir / "arcshield_security_usability_delta.png", dpi=180)
        plt.close()


def plot_attack_outputs(model_dir: Path, out_dir: Path, attack_df: pd.DataFrame) -> None:
    summary = (
        attack_df.groupby(["attack_type", "condition"], as_index=False)
        .agg(
            total=("attack_success", "size"),
            attack_success_rate=("attack_success", "mean"),
            refusal_rate=("refusal_detected", "mean"),
            partial_compliance_rate=("partial_compliance", "mean"),
            harmful_content_rate=("harmful_content_detected", "mean"),
            latency_mean_ms=("latency_ms", "mean"),
        )
        .sort_values(["attack_type", "condition"])
    )
    summary.to_csv(out_dir / "attack_summary_by_condition.csv", index=False)

    save_barplot(
        summary,
        x="attack_type",
        y="attack_success_rate",
        hue="condition",
        path=out_dir / "attack_success_rate_by_attack.png",
        title=f"{model_dir.name}: Attack Success Rate by Attack Type",
        ylabel="Attack success rate",
    )

    save_barplot(
        summary,
        x="attack_type",
        y="refusal_rate",
        hue="condition",
        path=out_dir / "refusal_rate_by_attack.png",
        title=f"{model_dir.name}: Refusal Rate by Attack Type",
        ylabel="Refusal rate",
    )

    save_barplot(
        summary,
        x="attack_type",
        y="partial_compliance_rate",
        hue="condition",
        path=out_dir / "partial_compliance_by_attack.png",
        title=f"{model_dir.name}: Partial Compliance by Attack Type",
        ylabel="Partial compliance rate",
    )

    latency = summary.dropna(subset=["latency_mean_ms"]).copy()
    if not latency.empty:
        latency["latency_mean_s"] = latency["latency_mean_ms"] / 1000.0
        save_barplot(
            latency,
            x="attack_type",
            y="latency_mean_s",
            hue="condition",
            path=out_dir / "latency_by_attack.png",
            title=f"{model_dir.name}: Mean Latency by Attack Type",
            ylabel="Mean Latency (s)",
            ylim=None,
            legend_y_anchor=0.035,
            bottom_adjust=0.34,
            top_adjust=0.82,
        )

    pivot = summary.pivot(index="attack_type", columns="condition", values="attack_success_rate").reset_index()
    if {"no_defense", "arcshield"}.issubset(pivot.columns):
        pivot["asr_reduction"] = pivot["no_defense"] - pivot["arcshield"]
        pivot.to_csv(out_dir / "arcshield_attack_success_reduction.csv", index=False)
        reduction = pivot.sort_values("asr_reduction", ascending=False).copy()
        reduction["color_key"] = reduction["asr_reduction"].apply(lambda value: "arcshield" if value >= 0 else "no_defense")

        plt.figure(figsize=(max(13, len(reduction["attack_type"].unique()) * 1.08), 8.2))
        sns.barplot(
            data=reduction,
            x="attack_type",
            y="asr_reduction",
            hue="color_key",
            palette=CONDITION_PALETTE,
            errorbar=None,
            dodge=False,
            legend=False,
        )
        ax = plt.gca()
        for patch, value in zip(ax.patches, reduction["asr_reduction"]):
            if float(value) >= 0:
                patch.set_hatch("//")
                patch.set_edgecolor("#EAF4FF")
                patch.set_linewidth(1.2)
        plt.axhline(0, color="#333333", linewidth=1)
        min_reduction = float(reduction["asr_reduction"].min())
        max_reduction = float(reduction["asr_reduction"].max())
        lower = min(-0.06, min_reduction - 0.06) if min_reduction < 0 else -0.03
        upper = max_reduction + 0.08
        beautify_axes(
            f"{model_dir.name}: ArcShield ASR Reduction",
            "No-defense ASR minus ArcShield ASR",
            ylim=(lower, upper),
            percent=True,
        )
        annotate_bars(percent=True, offset=0.006)
        plt.xticks(rotation=38, ha="right")
        plt.subplots_adjust(left=0.08, right=0.98, top=0.82, bottom=0.30)
        plt.savefig(out_dir / "arcshield_asr_reduction_by_attack.png", dpi=180)
        plt.close()


def plot_benign_outputs(model_dir: Path, out_dir: Path, benign_df: pd.DataFrame, category_df: pd.DataFrame) -> None:
    if not benign_df.empty:
        summary = (
            benign_df.groupby("condition", as_index=False)
            .agg(
                total=("refusal_detected", "size"),
                benign_refusal_rate=("refusal_detected", "mean"),
                harmful_false_flag_rate=("harmful_content_detected", "mean"),
            )
            .sort_values("condition")
        )
        summary.to_csv(out_dir / "benign_summary_by_condition.csv", index=False)

        melted = summary.melt(
            id_vars=["condition"],
            value_vars=["benign_refusal_rate", "harmful_false_flag_rate"],
            var_name="metric",
            value_name="rate",
        )
        metric_labels = {
            "benign_refusal_rate": "Benign refusal",
            "harmful_false_flag_rate": "Benign false flag",
        }
        melted["metric_label"] = melted["metric"].map(metric_labels)

        plt.figure(figsize=(10.5, 7.8))
        sns.barplot(
            data=melted,
            x="metric_label",
            y="rate",
            hue="condition",
            hue_order=["no_defense", "arcshield"],
            errorbar=None,
            palette=CONDITION_PALETTE,
        )
        beautify_axes(f"{model_dir.name}: Benign Refusal and False-Flag Rates", "Rate", ylim=(0, 1), percent=True)
        apply_arcshield_hatching()
        annotate_bars(percent=True)
        bottom_condition_legend(y_anchor=0.035)
        plt.xticks(rotation=15, ha="right")
        plt.subplots_adjust(left=0.08, right=0.98, top=0.82, bottom=0.30)
        plt.savefig(out_dir / "benign_refusal_and_false_flags.png", dpi=180)
        plt.close()

    if not benign_df.empty:
        cat = (
            benign_df.groupby(["benign_category", "condition"], as_index=False)
            .agg(
                harmful_false_flag_rate=("harmful_content_detected", "mean"),
                total=("harmful_content_detected", "size"),
            )
            .sort_values("harmful_false_flag_rate", ascending=False)
        )
        cat.to_csv(out_dir / "benign_summary_by_category.csv", index=False)

        plt.figure(figsize=(max(13, len(cat["benign_category"].unique()) * 1.08), 8.2))
        sns.barplot(
            data=cat,
            x="benign_category",
            y="harmful_false_flag_rate",
            hue="condition",
            hue_order=["no_defense", "arcshield"],
            errorbar=None,
            palette=CONDITION_PALETTE,
        )
        beautify_axes(f"{model_dir.name}: Benign Harmful False Flags by Category", "False-flag rate", ylim=(0, 1), percent=True)
        apply_arcshield_hatching()
        annotate_bars(percent=True)
        bottom_condition_legend(y_anchor=0.035)
        plt.xticks(rotation=38, ha="right")
        plt.subplots_adjust(left=0.08, right=0.98, top=0.82, bottom=0.30)
        plt.savefig(out_dir / "benign_false_flags_by_category.png", dpi=180)
        plt.close()
    elif not category_df.empty:
        cat = category_df.copy()
        cat["harmful_false_flag_rate"] = cat.apply(
            lambda r: (r["harmful_false_flags"] / r["total_benign_prompts"]) if r["total_benign_prompts"] else 0.0,
            axis=1,
        )
        cat.to_csv(out_dir / "benign_summary_by_category.csv", index=False)
        save_barplot(
            cat.sort_values("harmful_false_flag_rate", ascending=False),
            x="benign_category",
            y="harmful_false_flag_rate",
            path=out_dir / "benign_false_flags_by_category.png",
            title=f"{model_dir.name}: Benign Harmful False Flags by Category",
            ylabel="False-flag rate",
        )


def generate_for_model(model_dir: Path, output_root: Path) -> None:
    out_dir = output_root / model_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)

    attack_df, skipped = collect_attack_rows(model_dir)
    benign_df, benign_category_df = collect_benign_rows(model_dir)

    if not attack_df.empty:
        attack_df.to_csv(out_dir / "attack_rows_used.csv", index=False)
        plot_attack_outputs(model_dir, out_dir, attack_df)

    plot_benign_outputs(model_dir, out_dir, benign_df, benign_category_df)
    save_tradeoff_plot(model_dir, out_dir, attack_df, benign_df)

    manifest = {
        "model": model_dir.name,
        "attack_rows_used": int(len(attack_df)),
        "benign_rows_used": int(len(benign_df)),
        "skipped_attack_runs": skipped,
    }
    (out_dir / "plot_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def collect_model_attack_summaries(results_dir: Path) -> pd.DataFrame:
    summaries: list[pd.DataFrame] = []
    for model_dir in sorted(results_dir.iterdir()):
        if (
            not model_dir.is_dir()
            or model_dir.name.startswith("github_")
            or model_dir.name in PLOT_EXCLUDED_MODELS
        ):
            continue
        attack_df, _ = collect_attack_rows(model_dir)
        if attack_df.empty:
            continue
        summary = (
            attack_df.groupby(["attack_type", "condition"], as_index=False)
            .agg(
                total=("attack_success", "size"),
                attack_success_rate=("attack_success", "mean"),
                refusal_rate=("refusal_detected", "mean"),
                partial_compliance_rate=("partial_compliance", "mean"),
                latency_mean_ms=("latency_ms", "mean"),
            )
        )
        outcome_columns = {
            "refused": attack_df[attack_df["refusal_detected"] == True],
            "answered": attack_df[attack_df["refusal_detected"] == False],
        }
        for outcome, outcome_df in outcome_columns.items():
            outcome_summary = (
                outcome_df.groupby(["attack_type", "condition"], as_index=False)
                .agg(**{
                    f"latency_{outcome}_mean_ms": ("latency_ms", "mean"),
                    f"latency_{outcome}_n": ("latency_ms", "size"),
                })
            )
            summary = summary.merge(outcome_summary, on=["attack_type", "condition"], how="left")
            summary[f"latency_{outcome}_n"] = summary[f"latency_{outcome}_n"].fillna(0).astype(int)
        summary["model"] = model_dir.name
        summaries.append(summary)
    if not summaries:
        return pd.DataFrame()
    return pd.concat(summaries, ignore_index=True)


def save_model_comparison_bars(data: pd.DataFrame, out_dir: Path, condition: str, metric: str) -> None:
    subset = data[data["condition"] == condition].copy()
    if subset.empty:
        return

    filename = f"{metric}_{condition}_by_model_and_attack.png"
    plt.figure(figsize=(max(14, len(subset["attack_type"].unique()) * 1.10), 9.0))
    models = sorted(subset["model"].unique())
    palette = {model: MODEL_COLORS[index % len(MODEL_COLORS)] for index, model in enumerate(models)}
    sns.barplot(data=subset, x="attack_type", y=metric, hue="model", errorbar=None, palette=palette)
    beautify_axes(f"All Models: {pretty_name(metric)} ({display_condition(condition)})", f"{pretty_name(metric)} (%)", ylim=(0, 1), percent=True)
    plt.xticks(rotation=35, ha="right")
    handles, labels = plt.gca().get_legend_handles_labels()
    if plt.gca().get_legend() is not None:
        plt.gca().get_legend().remove()
    plt.gcf().legend(
        handles,
        [friendly_model_name(label) for label in labels],
        title="Model",
        loc="lower center",
        bbox_to_anchor=(0.5, 0.035),
        ncol=min(3, len(labels)),
        frameon=True,
        fancybox=True,
        shadow=True,
        fontsize=10,
    )
    plt.subplots_adjust(left=0.08, right=0.98, top=0.80, bottom=0.34)
    plt.savefig(out_dir / filename, dpi=180)
    plt.close()


def save_model_comparison_heatmap(data: pd.DataFrame, out_dir: Path, condition: str, metric: str) -> None:
    subset = data[data["condition"] == condition].copy()
    if subset.empty:
        return
    pivot = subset.pivot_table(index="model", columns="attack_type", values=metric, aggfunc="mean")
    plt.figure(figsize=(max(12, len(pivot.columns) * 0.75), max(5, len(pivot.index) * 0.45)))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="Spectral_r", vmin=0, vmax=1, linewidths=0.5)
    plt.title(f"All Models: {pretty_name(metric)} Heatmap ({condition})")
    plt.xlabel("")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(out_dir / f"{metric}_{condition}_heatmap.png", dpi=180)
    plt.close()


def strategic_profile(asr: float, latency: float, asr_cutoff: float, latency_cutoff: float) -> str:
    if asr <= asr_cutoff and latency <= latency_cutoff:
        return "OPTIMAL"
    if asr > asr_cutoff and latency <= latency_cutoff:
        return "VULNERABLE"
    if asr <= asr_cutoff and latency > latency_cutoff:
        return "SECURE / ROBUST"
    return "HI-RISK"


def pareto_frontier(points: pd.DataFrame) -> pd.DataFrame:
    candidates = points.dropna(subset=["latency_mean_ms", "attack_success_rate"]).copy()
    frontier_indices: list[Any] = []
    for index, point in candidates.iterrows():
        dominated = (
            (candidates["latency_mean_ms"] <= point["latency_mean_ms"])
            & (candidates["attack_success_rate"] <= point["attack_success_rate"])
            & (
                (candidates["latency_mean_ms"] < point["latency_mean_ms"])
                | (candidates["attack_success_rate"] < point["attack_success_rate"])
            )
        ).any()
        if not dominated:
            frontier_indices.append(index)
    return candidates.loc[frontier_indices].sort_values(["latency_mean_ms", "attack_success_rate"])


def save_strategic_pareto(
    summary: pd.DataFrame,
    out_dir: Path,
    condition: str,
    benign_summary: pd.DataFrame | None = None,
    axis_limits: tuple[float, float, float] | None = None,
) -> None:
    subset = summary[summary["condition"] == condition].copy()
    if subset.empty or "latency_mean_ms" not in subset.columns:
        return

    model_summary = (
        subset.groupby("model", as_index=False)
        .agg(
            attack_success_rate=("attack_success_rate", "mean"),
            latency_mean_ms=("latency_mean_ms", "mean"),
            total=("total", "sum"),
        )
        .dropna(subset=["latency_mean_ms"])
    )
    if model_summary.empty:
        return

    if benign_summary is not None and not benign_summary.empty:
        benign_subset = benign_summary[benign_summary["condition"] == condition][["model", "benign_refusal_rate"]].copy()
        model_summary = model_summary.merge(benign_subset, on="model", how="left")
    else:
        model_summary["benign_refusal_rate"] = pd.NA

    model_summary["model_label"] = model_summary["model"].map(friendly_model_name)
    asr_cutoff = float(model_summary["attack_success_rate"].median())
    latency_cutoff = float(model_summary["latency_mean_ms"].median())
    model_summary["profile"] = model_summary.apply(
        lambda row: strategic_profile(
            float(row["attack_success_rate"]),
            float(row["latency_mean_ms"]),
            asr_cutoff,
            latency_cutoff,
        ),
        axis=1,
    )
    frontier = pareto_frontier(model_summary)
    frontier_models = set(frontier["model"])
    model_summary["is_pareto_frontier"] = model_summary["model"].isin(frontier_models)
    model_summary.to_csv(out_dir / f"strategic_pareto_{condition}.csv", index=False)

    colors = {
        model: STRATEGIC_MODEL_COLORS[index % len(STRATEGIC_MODEL_COLORS)]
        for index, model in enumerate(sorted(model_summary["model"].unique()))
    }
    if axis_limits is None:
        x_min = max(1.0, float(model_summary["latency_mean_ms"].min()) * 0.65)
        x_max = float(model_summary["latency_mean_ms"].max()) * 1.55
        y_max = 1.0
    else:
        x_min, x_max, y_max = axis_limits

    fig, ax = plt.subplots(figsize=(18.4, 10.6))
    ax.set_xscale("log")

    ax.axvspan(x_min, latency_cutoff, ymin=0, ymax=asr_cutoff / y_max, color="#E8F5EC", alpha=0.75)
    ax.axvspan(latency_cutoff, x_max, ymin=0, ymax=asr_cutoff / y_max, color="#EEF3FA", alpha=0.75)
    ax.axvspan(x_min, latency_cutoff, ymin=asr_cutoff / y_max, ymax=1, color="#FCEEEF", alpha=0.55)
    ax.axvspan(latency_cutoff, x_max, ymin=asr_cutoff / y_max, ymax=1, color="#FFF4E8", alpha=0.55)
    ax.axhline(asr_cutoff, color="#88B99A", linestyle="--", linewidth=1.1)
    ax.axvline(latency_cutoff, color="#E4D1B9", linestyle="-", linewidth=0.9, alpha=0.65)

    for _, row in model_summary.iterrows():
        x = float(row["latency_mean_ms"])
        y = float(row["attack_success_rate"])
        is_frontier = bool(row["is_pareto_frontier"])
        size = (300 + min(float(row["total"]), 6000) / 6000 * 420) if is_frontier else 210
        usability_label = ""
        if pd.notna(row.get("benign_refusal_rate")):
            usability_label = f" | BRR {float(row['benign_refusal_rate']) * 100:.1f}%"
        ax.scatter(
            x,
            y,
            s=size,
            facecolor=colors[row["model"]] if is_frontier else "none",
            edgecolor="white" if is_frontier else colors[row["model"]],
            linewidth=1.8 if is_frontier else 2.2,
            alpha=0.98,
            zorder=4 if is_frontier else 2,
            label=(
                f"{row['model_label']} ({row['profile']}; {'FRONTIER' if is_frontier else 'DOMINATED'})\n"
                f"{x:.0f}ms | {y * 100:.1f}% ASR{usability_label}"
            ),
        )

    if len(frontier) > 1:
        ax.step(
            frontier["latency_mean_ms"],
            frontier["attack_success_rate"],
            where="post",
            linestyle="--",
            linewidth=2.5,
            color="#5F7480",
            alpha=0.95,
            zorder=3,
        )
        mid = frontier.iloc[len(frontier) // 2]
        ax.text(
            float(mid["latency_mean_ms"]),
            float(mid["attack_success_rate"]) + y_max * 0.05,
            "PARETO FRONTIER",
            fontsize=10,
            fontweight="bold",
            color="#5F7480",
        )

    ax.text(x_min * 1.05, y_max * 0.94, "VULNERABLE / EFFICIENT", color="#E8A8A8", fontsize=12, fontweight="bold", alpha=0.65)
    ax.text(latency_cutoff * 1.12, y_max * 0.94, "HI-RISK / REDUNDANT", color="#D6A28D", fontsize=12, fontweight="bold", alpha=0.65)
    ax.text(x_min * 1.05, y_max * 0.05, "STRATEGIC OPTIMUM", color="#91C7A3", fontsize=12, fontweight="bold", alpha=0.65)
    ax.text(latency_cutoff * 1.12, y_max * 0.05, "SECURE / ROBUST", color="#A6B8E8", fontsize=12, fontweight="bold", alpha=0.65)

    beautify_axes(
        f"Security vs. Efficiency\nStrategic Pareto - {display_condition(condition)}",
        "Security Breach Rate (ASR %)",
        ylim=(0, y_max),
        percent=True,
    )
    ax.set_title(
        f"Security vs. Efficiency\nStrategic Pareto - {display_condition(condition)}",
        fontsize=21,
        fontweight="bold",
        color=TITLE_COLOR,
        pad=26,
    )
    ax.set_xlim(x_min, x_max)
    ax.set_xlabel("Inference Latency (ms, Log Scale)", fontsize=15, color=AXIS_COLOR, labelpad=12)
    ax.legend(
        title="Model Strategic Profiles",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        frameon=True,
        fancybox=True,
        shadow=True,
        fontsize=10,
        title_fontsize=11,
        labelspacing=1.2,
        borderpad=1.2,
    )
    fig.tight_layout(rect=(0.03, 0.04, 0.76, 0.94))
    fig.savefig(out_dir / f"strategic_pareto_{condition}.png", dpi=220, bbox_inches="tight")
    plt.close()


def article_condition_label(condition: str) -> str:
    return {
        "no_defense": "No Defense",
        "arcshield": "ArcShield",
    }.get(condition, condition)


def save_article_ranked_asr(summary: pd.DataFrame, out_dir: Path) -> None:
    overall = (
        summary.groupby(["model", "condition"], as_index=False)
        .agg(attack_success_rate=("attack_success_rate", "mean"), total=("total", "sum"))
    )
    if overall.empty:
        return
    wide = overall.pivot_table(index="model", columns="condition", values="attack_success_rate", aggfunc="mean").reset_index()
    if not {"no_defense", "arcshield"}.issubset(wide.columns):
        overall["model_label"] = overall["model"].map(friendly_model_name)
        wide = overall.rename(columns={"attack_success_rate": "arcshield"})
        wide["no_defense"] = wide["arcshield"]
    wide["model_label"] = wide["model"].map(friendly_model_name)
    wide["asr_reduction"] = wide["no_defense"] - wide["arcshield"]
    wide = wide.sort_values(["arcshield", "asr_reduction"], ascending=[True, False])
    wide.to_csv(out_dir / "overall_asr_ranked.csv", index=False)

    fig_height = max(7.2, len(wide) * 0.46)
    fig, ax = plt.subplots(figsize=(11.5, fig_height), dpi=240)
    y_positions = list(range(len(wide)))
    x_max = min(1.0, max(0.28, float(wide[["no_defense", "arcshield"]].max().max()) + 0.12))
    reduction_x = x_max * 0.985
    for idx in y_positions:
        if idx % 2 == 0:
            ax.axhspan(idx - 0.46, idx + 0.46, color="#F7FAFC", zorder=0)
    for idx, (_, row) in enumerate(wide.iterrows()):
        ax.annotate(
            "",
            xy=(row["arcshield"], idx),
            xytext=(row["no_defense"], idx),
            arrowprops={
                "arrowstyle": "->",
                "color": "#B9C6CC",
                "lw": 2.3,
                "shrinkA": 7,
                "shrinkB": 7,
                "mutation_scale": 12,
            },
            zorder=1,
        )
    ax.scatter(wide["no_defense"], y_positions, s=120, marker="o", color="#E76F91", edgecolor="white", linewidth=1.35, label="No Defense", zorder=3)
    ax.scatter(wide["arcshield"], y_positions, s=120, marker="o", color="#2A9DF4", edgecolor="white", linewidth=1.35, label="ArcShield", zorder=4)
    for idx, (_, row) in enumerate(wide.iterrows()):
        ax.text(row["no_defense"] + 0.006, idx - 0.16, f"{row['no_defense'] * 100:.1f}%", va="center", fontsize=8.5, color="#7A3B52")
        ax.text(row["arcshield"] - 0.006, idx + 0.16, f"{row['arcshield'] * 100:.1f}%", va="center", ha="right", fontsize=8.5, color="#1D6699")
        ax.text(
            reduction_x,
            idx,
            f"-{row['asr_reduction'] * 100:.1f} pts",
            va="center",
            ha="right",
            fontsize=9.4,
            fontweight="bold",
            color="#2D6A4F" if row["asr_reduction"] >= 0 else "#B94747",
        )

    ax.text(reduction_x, -0.82, "ASR reduction", ha="right", va="bottom", fontsize=10.2, fontweight="bold", color=AXIS_COLOR)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(wide["model_label"], fontsize=10)
    ax.invert_yaxis()
    ax.set_xlim(0, x_max)
    ax.xaxis.set_major_formatter(percent_formatter())
    ax.set_xlabel("Attack Success Rate (ASR, %)", fontsize=12, color=AXIS_COLOR, labelpad=10)
    ax.set_ylabel("")
    ax.set_title("Overall ASR Comparison Across Models", fontsize=17, fontweight="bold", color=TITLE_COLOR, pad=10)
    ax.grid(axis="x", linestyle="--", alpha=0.34)
    ax.grid(axis="y", visible=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.text(0.02, 0.035, "●", transform=ax.transAxes, fontsize=12, color="#E76F91", va="center", fontweight="bold")
    ax.text(0.045, 0.035, "No Defense", transform=ax.transAxes, fontsize=9.8, color="#333333", va="center")
    ax.text(0.14, 0.035, "●", transform=ax.transAxes, fontsize=12, color="#2A9DF4", va="center", fontweight="bold")
    ax.text(0.165, 0.035, "ArcShield", transform=ax.transAxes, fontsize=9.8, color="#333333", va="center")
    for key_text in ax.texts[-4:]:
        key_text.remove()
    marker_key = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#E76F91", markeredgecolor="white", markersize=9, label="No Defense"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#2A9DF4", markeredgecolor="white", markersize=9, label="ArcShield"),
    ]
    ax.legend(
        handles=marker_key,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
        ncol=2,
        frameon=True,
        fancybox=True,
        framealpha=0.96,
        borderpad=0.6,
        handletextpad=0.55,
        columnspacing=1.5,
        fontsize=9.7,
    )
    fig.tight_layout(rect=(0, 0.11, 1, 0.98))
    fig.savefig(out_dir / "overall_asr_ranked.png", dpi=240, bbox_inches="tight", pad_inches=0.18)
    plt.close()


def save_article_asr_by_attack_heatmaps(summary: pd.DataFrame, out_dir: Path) -> None:
    for condition in ["no_defense", "arcshield"]:
        subset = summary[summary["condition"] == condition].copy()
        if subset.empty:
            continue
        subset["model_label"] = subset["model"].map(friendly_model_name)
        pivot = subset.pivot_table(index="attack_type", columns="model_label", values="attack_success_rate", aggfunc="mean")
        if pivot.empty:
            continue
        pivot.to_csv(out_dir / f"asr_by_attack_category_{condition}.csv")
        plt.figure(figsize=(max(11, len(pivot.columns) * 1.35), max(8, len(pivot.index) * 0.42)))
        sns.heatmap(
            pivot,
            annot=True,
            fmt=".0%",
            cmap="YlOrRd",
            vmin=0,
            vmax=1,
            linewidths=0.5,
            cbar_kws={"label": "ASR"},
        )
        plt.title(f"ASR by Attack Category ({display_condition(condition)})", fontsize=17, fontweight="bold", color=TITLE_COLOR, pad=16)
        plt.xlabel("")
        plt.ylabel("")
        plt.xticks(rotation=30, ha="right")
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(out_dir / f"asr_by_attack_category_{condition}.png", dpi=240)
        plt.close()


def save_article_refusal_by_attack_heatmaps(summary: pd.DataFrame, out_dir: Path) -> None:
    for condition in ["no_defense", "arcshield"]:
        subset = summary[summary["condition"] == condition].copy()
        if subset.empty:
            continue
        subset["model_label"] = subset["model"].map(friendly_model_name)
        pivot = subset.pivot_table(index="attack_type", columns="model_label", values="refusal_rate", aggfunc="mean")
        if pivot.empty:
            continue
        pivot.to_csv(out_dir / f"refusal_rate_{condition}_by_model_and_attack.csv")
        plt.figure(figsize=(max(11, len(pivot.columns) * 1.35), max(8, len(pivot.index) * 0.42)))
        sns.heatmap(
            pivot,
            annot=True,
            fmt=".0%",
            cmap="YlGnBu",
            vmin=0,
            vmax=1,
            linewidths=0.5,
            cbar_kws={"label": "Refusal rate"},
        )
        plt.title(f"Refusal Rate by Attack Category ({display_condition(condition)})", fontsize=17, fontweight="bold", color=TITLE_COLOR, pad=16)
        plt.xlabel("")
        plt.ylabel("")
        plt.xticks(rotation=30, ha="right")
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(out_dir / f"refusal_rate_{condition}_by_model_and_attack.png", dpi=240)
        plt.close()


def save_article_reduction_heatmap(summary: pd.DataFrame, out_dir: Path) -> None:
    pivot = summary.pivot_table(
        index=["model", "attack_type"],
        columns="condition",
        values="attack_success_rate",
        aggfunc="mean",
    ).reset_index()
    if not {"no_defense", "arcshield"}.issubset(pivot.columns):
        return
    pivot["asr_reduction"] = pivot["no_defense"] - pivot["arcshield"]
    pivot["model_label"] = pivot["model"].map(friendly_model_name)
    heat = pivot.pivot_table(index="attack_type", columns="model_label", values="asr_reduction", aggfunc="mean")
    if heat.empty:
        return
    color_limit = max(0.01, float(heat.abs().max().max()))
    pivot.to_csv(out_dir / "arcshield_asr_reduction.csv", index=False)
    plt.figure(figsize=(max(11, len(heat.columns) * 1.35), max(8, len(heat.index) * 0.42)))
    sns.heatmap(
        heat,
        annot=True,
        fmt=".0%",
        cmap="RdYlGn",
        center=0,
        vmin=-color_limit,
        vmax=color_limit,
        linewidths=0.5,
        cbar_kws={"label": "ASR reduction (negative = defense performed worse)"},
    )
    plt.title("ArcShield ASR Reduction Heatmap", fontsize=17, fontweight="bold", color=TITLE_COLOR, pad=16)
    plt.xlabel("")
    plt.ylabel("")
    plt.xticks(rotation=30, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(out_dir / "arcshield_asr_reduction_heatmap.png", dpi=240)
    plt.close()


def collect_model_benign_summaries(results_dir: Path) -> pd.DataFrame:
    summaries: list[pd.DataFrame] = []
    for model_dir in sorted(results_dir.iterdir()):
        if (
            not model_dir.is_dir()
            or model_dir.name.startswith("github_")
            or model_dir.name in PLOT_EXCLUDED_MODELS
        ):
            continue
        benign_df, _ = collect_benign_rows(model_dir)
        if benign_df.empty:
            continue
        summary = (
            benign_df.groupby("condition", as_index=False)
            .agg(
                benign_refusal_rate=("refusal_detected", "mean"),
                harmful_false_flag_rate=("harmful_content_detected", "mean"),
                total=("refusal_detected", "size"),
            )
        )
        summary["model"] = model_dir.name
        summaries.append(summary)
    if not summaries:
        return pd.DataFrame()
    return pd.concat(summaries, ignore_index=True)


def save_article_benign_rates(benign_summary: pd.DataFrame, out_dir: Path) -> None:
    if benign_summary.empty:
        return
    data = benign_summary.copy()
    data["model_label"] = data["model"].map(friendly_model_name)
    data["condition_label"] = data["condition"].map(article_condition_label)
    melted = data.melt(
        id_vars=["model", "model_label", "condition", "condition_label"],
        value_vars=["benign_refusal_rate", "harmful_false_flag_rate"],
        var_name="metric",
        value_name="rate",
    )
    melted["metric_label"] = melted["metric"].replace(
        {
            "benign_refusal_rate": "Benign refusal",
            "harmful_false_flag_rate": "False flag",
        }
    )
    melted.to_csv(out_dir / "benign_refusal_false_flags.csv", index=False)
    plt.figure(figsize=(max(12, len(data["model"].unique()) * 1.2), 7.5))
    sns.barplot(data=melted, x="model_label", y="rate", hue="metric_label", errorbar=None, palette=["#A7E8C5", "#FFD6A5"])
    beautify_axes("Benign Refusal and False-Flag Rates", "Rate", ylim=(0, 1), percent=True)
    annotate_bars(percent=True)
    plt.xticks(rotation=30, ha="right")
    plt.legend(title="", loc="upper right")
    plt.tight_layout()
    plt.savefig(out_dir / "benign_refusal_false_flags.png", dpi=240)
    plt.close()


def save_article_security_usability(summary: pd.DataFrame, benign_summary: pd.DataFrame, out_dir: Path) -> None:
    if summary.empty or benign_summary.empty:
        return
    attack = (
        summary.groupby(["model", "condition"], as_index=False)
        .agg(attack_success_rate=("attack_success_rate", "mean"), latency_mean_ms=("latency_mean_ms", "mean"))
    )
    data = attack.merge(benign_summary, on=["model", "condition"], how="inner")
    if data.empty:
        return
    data["model_label"] = data["model"].map(friendly_model_name)
    data["condition_label"] = data["condition"].map(article_condition_label)
    data.to_csv(out_dir / "security_usability_tradeoff.csv", index=False)

    plot_data = data[data["condition"] == "arcshield"].copy()
    if plot_data.empty:
        plot_data = data.copy()
    plot_data["deployment"] = plot_data["model"].map(deployment_type)
    fig, ax = plt.subplots(figsize=(10.8, 7.2), dpi=240)
    palette = {"Local": "#74B9FF", "Cloud": "#8FD19E"}
    sns.scatterplot(
        data=plot_data,
        x="benign_refusal_rate",
        y="attack_success_rate",
        hue="deployment",
        s=150,
        palette=palette,
        edgecolor="white",
        linewidth=1.2,
        ax=ax,
    )
    x_pad = max(0.012, float(plot_data["benign_refusal_rate"].max()) * 0.08)
    y_pad = max(0.015, float(plot_data["attack_success_rate"].max()) * 0.08)
    label_offsets = {
        "GEMINI": (24, -16),
        "MISTRAL": (18, -14),
        "OLLAMA LLAMA3.1": (18, -4),
        "OLLAMA QWEN3": (18, 10),
        "CLOUDFLARE QWEN": (34, -18),
        "GROQ LLAMA3.1": (18, 14),
        "OLLAMA MISTRAL": (34, 12),
        "OLLAMA DEEPSEEK R1": (34, -2),
    }
    for idx, (_, row) in enumerate(plot_data.sort_values("attack_success_rate").iterrows()):
        offset = label_offsets.get(row["model_label"], (18, 10 if idx % 2 == 0 else -12))
        label = (
            f"{row['model_label']}\n"
            f"ASR {float(row['attack_success_rate']) * 100:.1f}%, "
            f"BRR {float(row['benign_refusal_rate']) * 100:.1f}%"
        )
        ax.annotate(
            label,
            xy=(float(row["benign_refusal_rate"]), float(row["attack_success_rate"])),
            xytext=offset,
            textcoords="offset points",
            fontsize=8.0,
            color="#333333",
            arrowprops={"arrowstyle": "-", "color": "#CDD6DD", "lw": 0.6, "alpha": 0.8},
        )
    ax.axhline(plot_data["attack_success_rate"].median(), color="#B8C2CC", linestyle="--", linewidth=1.0, alpha=0.8)
    ax.axvline(plot_data["benign_refusal_rate"].median(), color="#B8C2CC", linestyle="--", linewidth=1.0, alpha=0.8)
    ax.xaxis.set_major_formatter(percent_formatter())
    ax.yaxis.set_major_formatter(percent_formatter())
    ax.set_xlabel("Benign Refusal Rate (%)", fontsize=13, color=AXIS_COLOR)
    ax.set_ylabel("Attack Success Rate (ASR, %)", fontsize=13, color=AXIS_COLOR)
    ax.set_title("Security-Usability Trade-Off under ArcShield", fontsize=16, fontweight="bold", color=TITLE_COLOR, pad=14)
    ax.set_xlim(left=-x_pad * 0.35, right=float(plot_data["benign_refusal_rate"].max()) + x_pad)
    ax.set_ylim(bottom=0, top=float(plot_data["attack_success_rate"].max()) + y_pad)
    ax.grid(True, linestyle="--", alpha=0.28)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(title="", loc="upper right", frameon=True, fontsize=9.5)
    fig.text(0.015, 0.02, "Lower-left is better: fewer successful attacks and fewer benign refusals.", fontsize=8.8, color="#666666")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(out_dir / "security_usability_tradeoff.png", dpi=240)
    plt.close()


def save_article_latency_by_attack_legacy(summary: pd.DataFrame, out_dir: Path) -> None:
    subset = summary.dropna(subset=["latency_mean_ms"]).copy()
    if subset.empty:
        return
    latency = (
        subset.groupby(["attack_type", "condition"], as_index=False)
        .agg(latency_mean_ms=("latency_mean_ms", "mean"))
    )
    latency["latency_mean_s"] = latency["latency_mean_ms"] / 1000.0
    latency["condition_label"] = latency["condition"].map(article_condition_label)
    latency.to_csv(out_dir / "mean_latency_by_attack_type.csv", index=False)
    wide = latency.pivot_table(index="attack_type", columns="condition", values="latency_mean_s", aggfunc="mean").reset_index()
    if not {"no_defense", "arcshield"}.issubset(wide.columns):
        return
    wide["delta"] = wide["no_defense"] - wide["arcshield"]
    wide = wide.sort_values("arcshield", ascending=True)
    fig, ax = plt.subplots(figsize=(11.5, max(7.2, len(wide) * 0.46)), dpi=240)
    y_positions = list(range(len(wide)))
    for idx in y_positions:
        if idx % 2 == 0:
            ax.axhspan(idx - 0.46, idx + 0.46, color="#F7FAFC", zorder=0)
    for idx, (_, row) in enumerate(wide.iterrows()):
        ax.annotate("", xy=(row["arcshield"], idx), xytext=(row["no_defense"], idx), arrowprops={"arrowstyle": "->", "color": "#B9C6CC", "lw": 2.3, "shrinkA": 7, "shrinkB": 7, "mutation_scale": 12}, zorder=1)
    ax.scatter(wide["no_defense"], y_positions, s=120, color="#E76F91", edgecolor="white", linewidth=1.35, label="No Defense", zorder=3)
    ax.scatter(wide["arcshield"], y_positions, s=120, color="#2A9DF4", edgecolor="white", linewidth=1.35, label="ArcShield", zorder=4)
    for idx, (_, row) in enumerate(wide.iterrows()):
        ax.text(row["arcshield"] - 0.35, idx, f"{row['arcshield']:.1f}", ha="right", va="center", fontsize=8.2, color="#555555")
        ax.text(row["no_defense"] + 0.35, idx, f"{row['no_defense']:.1f}", va="center", fontsize=8.2, color="#555555")
    ax.set_yticks(y_positions)
    ax.set_yticklabels(wide["attack_type"], fontsize=9.5)
    ax.invert_yaxis()
    ax.set_xlabel("Mean Latency (seconds)", fontsize=13, color=AXIS_COLOR, labelpad=10)
    ax.set_ylabel("")
    ax.set_title("Mean Latency by Attack Type", fontsize=16, fontweight="bold", color=TITLE_COLOR, pad=12)
    ax.set_xlim(left=0, right=float(wide[["no_defense", "arcshield"]].max().max()) + 4)
    ax.grid(axis="x", linestyle="--", alpha=0.34)
    ax.grid(axis="y", visible=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.text(0.02, 0.035, "●", transform=ax.transAxes, fontsize=12, color="#F29BC1", va="center", fontweight="bold")
    ax.text(0.045, 0.035, "No Defense", transform=ax.transAxes, fontsize=9.8, color="#333333", va="center")
    ax.text(0.14, 0.035, "●", transform=ax.transAxes, fontsize=12, color="#74B9FF", va="center", fontweight="bold")
    ax.text(0.165, 0.035, "ArcShield", transform=ax.transAxes, fontsize=9.8, color="#333333", va="center")
    for key_text in ax.texts[-4:]:
        key_text.remove()
    marker_key = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#F29BC1", markeredgecolor="white", markersize=9, label="No Defense"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#74B9FF", markeredgecolor="white", markersize=9, label="ArcShield"),
    ]
    ax.legend(
        handles=marker_key,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.14),
        ncol=2,
        frameon=True,
        fancybox=True,
        framealpha=0.96,
        borderpad=0.6,
        handletextpad=0.55,
        columnspacing=1.5,
        fontsize=9.7,
    )
    fig.tight_layout(rect=(0, 0.11, 1, 0.98))
    fig.savefig(out_dir / "mean_latency_by_attack_type.png", dpi=240, bbox_inches="tight", pad_inches=0.18)
    plt.close()


def save_article_latency_by_attack(summary: pd.DataFrame, out_dir: Path) -> None:
    rows: list[dict[str, Any]] = []
    for outcome in ["refused", "answered"]:
        mean_col = f"latency_{outcome}_mean_ms"
        n_col = f"latency_{outcome}_n"
        for (attack_type, condition), group in summary.groupby(["attack_type", "condition"]):
            valid = group[group[n_col] > 0].dropna(subset=[mean_col])
            n = int(valid[n_col].sum())
            if n == 0:
                continue
            mean_ms = float((valid[mean_col] * valid[n_col]).sum() / n)
            rows.append({"attack_type": attack_type, "condition": condition, "outcome": outcome, "latency_mean_s": mean_ms / 1000.0, "n": n})
    latency = pd.DataFrame(rows)
    if latency.empty:
        return
    latency.to_csv(out_dir / "mean_latency_by_attack_type.csv", index=False)
    x_max = float(latency["latency_mean_s"].max()) * 1.12
    attack_order = sorted(latency["attack_type"].unique())
    fig, axes = plt.subplots(1, 2, figsize=(16.0, max(7.2, len(attack_order) * 0.46)), dpi=240, sharex=True, sharey=True)
    for ax, outcome in zip(axes, ["refused", "answered"]):
        data = latency[latency["outcome"] == outcome]
        wide = data.pivot_table(index="attack_type", columns="condition", values="latency_mean_s", aggfunc="mean").reindex(attack_order)
        y_positions = list(range(len(wide)))
        for idx in y_positions:
            if idx % 2 == 0:
                ax.axhspan(idx - 0.46, idx + 0.46, color="#F7FAFC", zorder=0)
        if {"no_defense", "arcshield"}.issubset(wide.columns):
            for idx, (_, row) in enumerate(wide.iterrows()):
                if pd.notna(row["no_defense"]) and pd.notna(row["arcshield"]):
                    ax.plot([row["no_defense"], row["arcshield"]], [idx, idx], color="#C8D0D6", linewidth=2.0, zorder=1)
            ax.scatter(wide["no_defense"], y_positions, s=85, color="#E76F91", edgecolor="white", linewidth=1.2, zorder=3)
            ax.scatter(wide["arcshield"], y_positions, s=85, color="#2A9DF4", edgecolor="white", linewidth=1.2, zorder=4)
        ax.set_title(f"{outcome.title()} responses", fontsize=15, fontweight="bold", color=TITLE_COLOR, pad=12)
        ax.set_yticks(y_positions)
        ax.set_yticklabels(wide.index, fontsize=8.8)
        ax.invert_yaxis()
        ax.set_xlim(0, x_max)
        ax.set_xlabel("Mean latency (seconds)", fontsize=11, color=AXIS_COLOR)
        ax.grid(axis="x", linestyle="--", alpha=0.34)
        ax.grid(axis="y", visible=False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#E76F91", markeredgecolor="white", markersize=8, label="No Defense"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#2A9DF4", markeredgecolor="white", markersize=8, label="ArcShield"),
    ]
    fig.suptitle("Latency by Response Outcome and Attack Type", fontsize=18, fontweight="bold", color=TITLE_COLOR, y=0.98)
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.06), ncol=2, frameon=True, fancybox=True)
    fig.text(0.5, 0.018, "Refused and answered responses are separated because early refusals often terminate generation sooner.", ha="center", fontsize=9.3, color="#666666")
    fig.tight_layout(rect=(0, 0.12, 1, 0.95), w_pad=2.5)
    fig.savefig(out_dir / "mean_latency_by_attack_type.png", dpi=240, bbox_inches="tight", pad_inches=0.18)
    plt.close()


def save_article_pareto(summary: pd.DataFrame, benign_summary: pd.DataFrame, out_dir: Path) -> None:
    dominance_path = out_dir / "publication_pareto_dominance.csv"
    if not dominance_path.exists():
        raise FileNotFoundError(
            "Authoritative Pareto table is required before plotting: "
            f"{dominance_path}"
        )
    dominance = pd.read_csv(dominance_path)
    required = {
        "condition",
        "model",
        "model_label",
        "is_pareto_frontier",
        "attack_success_rate",
        "latency_mean_ms",
        "dominated_by_example",
    }
    missing = required - set(dominance.columns)
    if missing:
        raise ValueError(f"Pareto table is missing columns: {sorted(missing)}")

    model_summary = dominance[list(required)].copy()
    model_summary["is_pareto_frontier"] = (
        model_summary["is_pareto_frontier"]
        .astype(str)
        .str.strip()
        .str.lower()
        .eq("true")
    )
    model_summary["attack_success_rate"] = pd.to_numeric(
        model_summary["attack_success_rate"], errors="raise"
    )
    model_summary["latency_mean_ms"] = pd.to_numeric(
        model_summary["latency_mean_ms"], errors="raise"
    )

    strict_summary = (
        summary.dropna(subset=["latency_mean_ms"])
        .groupby(["model", "condition"], as_index=False)
        .agg(
            attack_success_rate=("attack_success_rate", "mean"),
            latency_mean_ms=("latency_mean_ms", "mean"),
        )
    )
    merged_check = model_summary.merge(
        strict_summary,
        on=["model", "condition"],
        how="outer",
        suffixes=("_authoritative", "_strict"),
        indicator=True,
    )
    if len(model_summary) != 22 or set(model_summary["condition"]) != {"no_defense", "arcshield"}:
        raise ValueError("Expected exactly 22 authoritative model-condition rows.")
    condition_counts = model_summary.groupby("condition")["model"].nunique()
    if not (condition_counts == 11).all() or model_summary.duplicated(["model", "condition"]).any():
        raise ValueError("Each condition must contain each of the eleven models exactly once.")
    if not (merged_check["_merge"] == "both").all():
        raise ValueError("Authoritative Pareto rows do not match the strict-cohort model cells.")
    for metric in ["attack_success_rate", "latency_mean_ms"]:
        if not np.allclose(
            merged_check[f"{metric}_authoritative"],
            merged_check[f"{metric}_strict"],
            rtol=0,
            atol=1e-9,
        ):
            raise ValueError(f"Authoritative and strict-cohort {metric} values differ.")

    model_summary["deployment"] = model_summary["model"].map(
        lambda value: "Local" if str(value).startswith("ollama_") else "Cloud"
    )
    plotted = pd.DataFrame(
        {
            "model": model_summary["model_label"],
            "deployment": model_summary["deployment"],
            "condition": model_summary["condition"],
            "asr_percent": model_summary["attack_success_rate"] * 100.0,
            "mean_latency_ms": model_summary["latency_mean_ms"],
            "pareto_nondominated": model_summary["is_pareto_frontier"],
            "dominated_by": model_summary["dominated_by_example"],
        }
    )
    plotted.to_csv(
        out_dir / "security_efficiency_pareto_frontier_data.csv", index=False
    )

    models = sorted(model_summary["model"].unique())
    colors = {model: STRATEGIC_MODEL_COLORS[index % len(STRATEGIC_MODEL_COLORS)] for index, model in enumerate(models)}
    x_min = max(1.0, float(model_summary["latency_mean_ms"].min()) * 0.72)
    x_max = float(model_summary["latency_mean_ms"].max()) * 1.42
    y_max = float(model_summary["attack_success_rate"].max()) * 1.18

    fig, axes = plt.subplots(1, 2, figsize=(14.8, 8.2), dpi=240, sharex=True, sharey=True)
    for ax, condition, panel_title in zip(
        axes,
        ["no_defense", "arcshield"],
        ["(a) No Defense", "(b) ArcShield"],
    ):
        data = model_summary[model_summary["condition"] == condition].copy()
        frontier = data[data["is_pareto_frontier"]].sort_values("latency_mean_ms")
        dominated = data[~data["is_pareto_frontier"]]
        ax.set_xscale("log")
        if len(frontier) > 1:
            ax.plot(
                frontier["latency_mean_ms"],
                frontier["attack_success_rate"],
                color="#264653",
                linewidth=2.0,
                linestyle="-",
                zorder=2,
            )
        for _, row in dominated.iterrows():
            x = float(row["latency_mean_ms"])
            y = float(row["attack_success_rate"])
            ax.scatter(
                x,
                y,
                s=92,
                color=colors[row["model"]],
                edgecolor="#FFFFFF",
                linewidth=1.0,
                alpha=0.62,
                zorder=3,
            )
        for label_index, (_, row) in enumerate(frontier.iterrows()):
            x = float(row["latency_mean_ms"])
            y = float(row["attack_success_rate"])
            ax.scatter(
                x,
                y,
                s=190,
                color=colors[row["model"]],
                edgecolor="#1F2933",
                linewidth=2.0,
                zorder=4,
            )
            ax.annotate(
                friendly_model_name(row["model"]),
                xy=(x, y),
                xytext=(7, 9 if label_index % 2 == 0 else -15),
                textcoords="offset points",
                fontsize=8.2,
                fontweight="semibold",
                color="#263238",
                ha="left",
                va="bottom" if label_index % 2 == 0 else "top",
            )
        ax.set_title(panel_title, fontsize=15, fontweight="bold", color=TITLE_COLOR, pad=12)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(0, y_max)
        ax.set_xlabel("Mean inference latency (ms, log scale)", fontsize=11, color=AXIS_COLOR)
        ax.yaxis.set_major_formatter(percent_formatter())
        ax.grid(True, linestyle="--", alpha=0.28)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].set_ylabel("Attack Success Rate (ASR, %)", fontsize=11, color=AXIS_COLOR)
    fig.suptitle(
        "Security–Efficiency Pareto Frontiers by Defense Condition",
        fontsize=18,
        fontweight="bold",
        color=TITLE_COLOR,
        y=0.985,
    )
    model_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=colors[model],
            markeredgecolor="white",
            markersize=8,
            label=friendly_model_name(model),
        )
        for model in models
    ]
    status_handles = [
        Line2D(
            [0], [0], marker="o", linestyle="none", markerfacecolor="#FFFFFF",
            markeredgecolor="#1F2933", markeredgewidth=2.0, markersize=10,
            label="Pareto-nondominated configuration",
        ),
        Line2D(
            [0], [0], marker="o", linestyle="none", markerfacecolor="#9CA3AF",
            markeredgecolor="#FFFFFF", alpha=0.62, markersize=8,
            label="Dominated configuration",
        ),
        Line2D(
            [0], [0], color="#264653", linewidth=2.0,
            label="Pareto-frontier connection",
        ),
    ]
    status_legend = fig.legend(
        handles=status_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.925),
        ncol=3,
        frameon=True,
        fancybox=True,
        fontsize=8.7,
    )
    fig.add_artist(status_legend)
    fig.legend(
        handles=model_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.065),
        ncol=4,
        frameon=True,
        fancybox=True,
        fontsize=8.2,
        columnspacing=1.2,
    )
    fig.text(
        0.5,
        0.018,
        "Lower-left is preferable: lower ASR and lower mean inference latency.",
        ha="center",
        fontsize=9.5,
        color="#666666",
    )
    fig.tight_layout(rect=(0.02, 0.21, 1, 0.88), w_pad=2.6)
    fig.savefig(out_dir / "security_efficiency_pareto_frontier.png", dpi=220, bbox_inches="tight", pad_inches=0.2)
    plt.close()


def save_article_strategic_paretos(summary: pd.DataFrame, benign_summary: pd.DataFrame, out_dir: Path) -> None:
    latency = summary["latency_mean_ms"].dropna()
    if latency.empty:
        return
    shared_limits = (
        max(1.0, float(latency.min()) * 0.65),
        float(latency.max()) * 1.55,
        1.0,
    )
    for condition in ["no_defense", "arcshield"]:
        save_strategic_pareto(summary, out_dir, condition, benign_summary, shared_limits)


def deployment_type(model: str) -> str:
    if model.startswith("ollama_"):
        return "Local"
    return "Cloud"


def save_article_local_cloud(summary: pd.DataFrame, out_dir: Path) -> None:
    dominance_path = out_dir / "publication_pareto_dominance.csv"
    if not dominance_path.exists():
        raise FileNotFoundError(
            "Strict-cohort model metrics are required before plotting: "
            f"{dominance_path}"
        )
    data = pd.read_csv(dominance_path)[
        ["model", "model_label", "condition", "attack_success_rate", "latency_mean_ms"]
    ].copy()
    if len(data) != 22 or data.duplicated(["model", "condition"]).any():
        raise ValueError("Expected one strict-cohort row for each of 11 models in both conditions.")
    data["deployment"] = data["model"].map(deployment_type)
    data["condition_label"] = data["condition"].map(article_condition_label)
    deployment_counts = data.groupby(["condition", "deployment"])["model"].nunique()
    for condition in ["no_defense", "arcshield"]:
        if deployment_counts.get((condition, "Local"), 0) != 5:
            raise ValueError(f"{condition} does not contain five local configurations.")
        if deployment_counts.get((condition, "Cloud"), 0) != 6:
            raise ValueError(f"{condition} does not contain six cloud configurations.")
    data.to_csv(out_dir / "local_vs_cloud_deployment.csv", index=False)
    deploy = (
        data.groupby(["deployment", "condition_label"], as_index=False)
        .agg(
            model_count=("model", "nunique"),
            attack_success_rate=("attack_success_rate", "mean"),
            attack_success_rate_sd=("attack_success_rate", "std"),
            latency_mean_ms=("latency_mean_ms", "mean"),
            latency_mean_ms_sd=("latency_mean_ms", "std"),
        )
    )
    deploy.to_csv(out_dir / "local_vs_cloud_deployment_summary.csv", index=False)

    deployment_order = ["Cloud", "Local"]
    condition_order = ["ArcShield", "No Defense"]
    condition_colors = {
        "ArcShield": CONDITION_PALETTE["no_defense"],
        "No Defense": CONDITION_PALETTE["arcshield"],
    }
    width = 0.36
    x = np.arange(len(deployment_order))
    offsets = {"ArcShield": -width / 2, "No Defense": width / 2}
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 7.6))

    for condition_label in condition_order:
        values = (
            deploy[deploy["condition_label"] == condition_label]
            .set_index("deployment")
            .reindex(deployment_order)
        )
        bars = axes[0].bar(
            x + offsets[condition_label],
            values["attack_success_rate"],
            width,
            color=condition_colors[condition_label],
            edgecolor="white",
            linewidth=1.0,
            alpha=0.86,
            label=condition_label,
        )
        for bar in bars:
            height = float(bar.get_height())
            axes[0].annotate(
                f"{height * 100:.1f}%",
                (bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 5),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9.2,
                fontweight="bold",
                color="#263238",
            )
        latency_bars = axes[1].bar(
            x + offsets[condition_label],
            values["latency_mean_ms"],
            width,
            color=condition_colors[condition_label],
            edgecolor="white",
            linewidth=1.0,
            alpha=0.86,
            label=condition_label,
        )
        for bar in latency_bars:
            height = float(bar.get_height())
            axes[1].annotate(
                f"{height:,.0f} ms",
                (bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 5),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9.2,
                fontweight="bold",
                color="#263238",
            )

        condition_key = "arcshield" if condition_label == "ArcShield" else "no_defense"
        for deployment_index, deployment in enumerate(deployment_order):
            points = data[
                (data["condition"] == condition_key)
                & (data["deployment"] == deployment)
            ].sort_values("model")
            # Keep each model cluster visibly inside its corresponding bar.
            jitter = np.linspace(-0.055, 0.055, len(points))
            point_x = x[deployment_index] + offsets[condition_label] + jitter
            axes[0].scatter(
                point_x,
                points["attack_success_rate"],
                s=34,
                facecolor=condition_colors[condition_label],
                edgecolor="#263238",
                linewidth=1.0,
                alpha=1.0,
                zorder=4,
            )
            axes[1].scatter(
                point_x,
                points["latency_mean_ms"],
                s=34,
                facecolor=condition_colors[condition_label],
                edgecolor="#263238",
                linewidth=1.0,
                alpha=1.0,
                zorder=4,
            )

    axes[0].yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value * 100:.0f}%"))
    axes[0].set_title("ASR", fontsize=14, fontweight="bold", color=TITLE_COLOR)
    axes[0].set_ylabel("Attack Success Rate (ASR)")
    axes[1].set_title("Latency", fontsize=14, fontweight="bold", color=TITLE_COLOR)
    axes[1].set_ylabel("Mean Latency (ms)")
    axes[1].yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:,.0f}"))
    max_asr = float(data["attack_success_rate"].max())
    max_latency = float(data["latency_mean_ms"].max())
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(deployment_order, fontsize=11)
        ax.set_xlabel("")
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].set_ylim(0, max_asr * 1.18)
    axes[1].set_ylim(0, max_latency * 1.17)

    legend_handles = [
        Patch(facecolor=condition_colors[label], edgecolor="white", label=label)
        for label in condition_order
    ]
    legend_handles.extend([
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=condition_colors["ArcShield"],
            markeredgecolor="#263238",
            markersize=6,
            label="ArcShield model",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=condition_colors["No Defense"],
            markeredgecolor="#263238",
            markersize=6,
            label="No Defense model",
        ),
    ])
    fig.suptitle(
        "Local vs Cloud Deployment Comparison",
        fontsize=18,
        fontweight="bold",
        color=TITLE_COLOR,
        y=0.98,
    )
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.105),
        ncol=4,
        frameon=True,
        fontsize=10,
    )
    fig.text(
        0.5,
        0.035,
        "Bars are unweighted model-level means across five local and six cloud configurations; "
        "points show individual model values.",
        ha="center",
        fontsize=9.5,
        color="#666666",
    )
    plt.subplots_adjust(bottom=0.24, top=0.86, wspace=0.25)
    plt.savefig(
        out_dir / "local_vs_cloud_deployment_comparison.png",
        dpi=240,
        bbox_inches="tight",
        pad_inches=0.16,
    )
    plt.close()


def generate_article_plots(results_dir: Path, output_root: Path) -> None:
    out_dir = output_root / "article_plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = collect_model_attack_summaries(results_dir)
    benign_summary = collect_model_benign_summaries(results_dir)

    manifest = {
        "status": "ok" if not summary.empty else "no complete attack rows found",
        "attack_summary_rows": int(len(summary)),
        "benign_summary_rows": int(len(benign_summary)),
        "figures": [
            "overall_asr_ranked.png",
            "asr_by_attack_category_no_defense.png",
            "asr_by_attack_category_arcshield.png",
            "refusal_rate_no_defense_by_model_and_attack.png",
            "refusal_rate_arcshield_by_model_and_attack.png",
            "arcshield_asr_reduction_heatmap.png",
            "benign_refusal_false_flags.png",
            "mean_latency_by_attack_type.png",
            "security_efficiency_pareto_frontier.png",
            "strategic_pareto_no_defense.png",
            "strategic_pareto_arcshield.png",
            "local_vs_cloud_deployment_comparison.png",
        ],
    }
    if summary.empty:
        (out_dir / "article_plot_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return

    summary.to_csv(out_dir / "article_attack_summary_source.csv", index=False)
    summary[["model", "attack_type", "condition", "total"]].rename(columns={"total": "n"}).to_csv(
        out_dir / "cell_sample_sizes.csv", index=False
    )
    if not benign_summary.empty:
        benign_summary.to_csv(out_dir / "article_benign_summary_source.csv", index=False)

    save_article_ranked_asr(summary, out_dir)
    save_article_asr_by_attack_heatmaps(summary, out_dir)
    save_article_refusal_by_attack_heatmaps(summary, out_dir)
    save_article_reduction_heatmap(summary, out_dir)
    save_article_benign_rates(benign_summary, out_dir)
    save_article_latency_by_attack(summary, out_dir)
    save_article_pareto(summary, benign_summary, out_dir)
    save_article_strategic_paretos(summary, benign_summary, out_dir)
    save_article_local_cloud(summary, out_dir)
    (out_dir / "article_plot_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def generate_all_model_comparison(results_dir: Path, output_root: Path) -> None:
    summary = collect_model_attack_summaries(results_dir)
    out_dir = output_root / "all_models_comparison"
    out_dir.mkdir(parents=True, exist_ok=True)

    if summary.empty:
        (out_dir / "plot_manifest.json").write_text(
            json.dumps({"status": "no complete attack rows found"}, indent=2),
            encoding="utf-8",
        )
        return

    summary.to_csv(out_dir / "all_models_attack_summary.csv", index=False)

    overall = (
        summary.groupby(["model", "condition"], as_index=False)
        .agg(attack_success_rate=("attack_success_rate", "mean"))
        .sort_values(["model", "condition"])
    )
    overall["model_label"] = overall["model"].map(compact_model_name)
    overall.to_csv(out_dir / "overall_asr_by_model.csv", index=False)
    plt.figure(figsize=(max(13, len(overall["model"].unique()) * 1.65), 8.8))
    sns.barplot(
        data=overall,
        x="model_label",
        y="attack_success_rate",
        hue="condition",
        errorbar=None,
        palette=CONDITION_PALETTE,
    )
    beautify_axes("Attack Success Rate (ASR) Comparison", "ASR (%)", ylim=(0, 1), percent=True)
    apply_arcshield_hatching()
    annotate_bars(percent=True)
    bottom_condition_legend(y_anchor=0.035)
    plt.xticks(rotation=0, ha="center", fontsize=11, fontweight="bold")
    plt.subplots_adjust(left=0.08, right=0.98, top=0.80, bottom=0.28)
    plt.savefig(out_dir / "overall_asr_comparison.png", dpi=180)
    plt.close()

    for condition in ["no_defense", "arcshield"]:
        save_model_comparison_bars(summary, out_dir, condition, "attack_success_rate")
        save_model_comparison_bars(summary, out_dir, condition, "refusal_rate")
        save_model_comparison_heatmap(summary, out_dir, condition, "attack_success_rate")
        save_strategic_pareto(summary, out_dir, condition)

    pivot = summary.pivot_table(
        index=["model", "attack_type"],
        columns="condition",
        values="attack_success_rate",
        aggfunc="mean",
    ).reset_index()
    if {"no_defense", "arcshield"}.issubset(pivot.columns):
        pivot["arcshield_asr_reduction"] = pivot["no_defense"] - pivot["arcshield"]
        pivot.to_csv(out_dir / "all_models_arcshield_asr_reduction.csv", index=False)
        heat = pivot.pivot_table(index="model", columns="attack_type", values="arcshield_asr_reduction", aggfunc="mean")
        plt.figure(figsize=(max(12, len(heat.columns) * 0.75), max(5, len(heat.index) * 0.45)))
        sns.heatmap(heat, annot=True, fmt=".2f", cmap="RdYlGn", center=0, linewidths=0.5)
        plt.title("All Models: ArcShield Attack Success Reduction")
        plt.xlabel("")
        plt.ylabel("")
        plt.tight_layout()
        plt.savefig(out_dir / "all_models_arcshield_asr_reduction_heatmap.png", dpi=180)
        plt.close()


def pair_display_name(model_name: str) -> str:
    names = {
        "ollama_llama3_1_8b": "Local Llama3.1",
        "groq_llama_3_1_8b_instant": "Groq Llama3.1",
        "ollama_qwen3_latest": "Local Qwen3",
        "cloudflare_cf_qwen_qwen3_30b_a3b_fp8": "Cloudflare Qwen",
        "ollama_deepseek_r1_latest": "Local DeepSeek R1",
        "cloudflare_cf_deepseek_ai_deepseek_r1_distill_qwen_32b": "Cloudflare DeepSeek R1 Distill Qwen 32B",
        "ollama_gemma4_latest": "Local Gemma 4",
        "cloudflare_cf_google_gemma_4_26b_a4b_it": "Cloudflare Gemma 4",
        "ollama_mistral_latest": "Local Mistral",
        "mistral_mistral_small_latest": "Cloud Mistral",
    }
    return names.get(model_name, model_name.replace("cloudflare_cf_", "Cloud ").replace("ollama_", "Local "))


def pair_folder_name(model_name: str) -> str:
    return model_name.replace("cloudflare2_cf_", "cloud_").replace("cloudflare_cf_", "cloud_").replace("ollama_", "local_")


def generate_pair_comparison(results_dir: Path, output_root: Path, local_model: str, cloud_model: str) -> None:
    local_dir = results_dir / local_model
    cloud_dir = results_dir / cloud_model
    out_dir = output_root / f"{pair_folder_name(local_model)}_vs_{pair_folder_name(cloud_model)}"
    out_dir.mkdir(parents=True, exist_ok=True)

    local_df, local_skipped = collect_attack_rows(local_dir) if local_dir.exists() else (pd.DataFrame(), ["missing model folder"])
    cloud_df, cloud_skipped = collect_attack_rows(cloud_dir) if cloud_dir.exists() else (pd.DataFrame(), ["missing model folder"])

    manifest = {
        "local_model": local_model,
        "cloud_model": cloud_model,
        "local_attack_rows_used": int(len(local_df)),
        "cloud_attack_rows_used": int(len(cloud_df)),
        "local_skipped_attack_runs": local_skipped,
        "cloud_skipped_attack_runs": cloud_skipped,
    }
    (out_dir / "plot_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    if local_df.empty or cloud_df.empty:
        return

    frames = []
    for label, df in [("local", local_df), ("cloud", cloud_df)]:
        summary = (
            df.groupby(["attack_type", "condition"], as_index=False)
            .agg(
                attack_success_rate=("attack_success", "mean"),
                refusal_rate=("refusal_detected", "mean"),
                partial_compliance_rate=("partial_compliance", "mean"),
            )
        )
        summary["source"] = label
        frames.append(summary)
    paired = pd.concat(frames, ignore_index=True)
    paired.to_csv(out_dir / "local_vs_cloud_attack_summary.csv", index=False)

    for metric in ["attack_success_rate", "refusal_rate", "partial_compliance_rate"]:
        for condition in ["no_defense", "arcshield"]:
            subset = paired[paired["condition"] == condition]
            if subset.empty:
                continue
            plt.figure(figsize=(max(13, len(subset["attack_type"].unique()) * 1.05), 8.2))
            sns.barplot(data=subset, x="attack_type", y=metric, hue="source", errorbar=None, palette=["#F6A6C8", "#9CCBFF"])
            title = (
                f"{pair_display_name(local_model)} vs {pair_display_name(cloud_model)}\n"
                f"{pretty_name(metric)} ({display_condition(condition)})"
            )
            beautify_axes(
                title,
                f"{pretty_name(metric)} (%)",
                ylim=(0, 1),
                percent=True,
            )
            annotate_bars(percent=True)
            plt.xticks(rotation=35, ha="right")
            fig = plt.gcf()
            fig.legend(loc="lower center", bbox_to_anchor=(0.5, 0.035), ncol=2, frameon=True, fancybox=True, shadow=True)
            if plt.gca().get_legend() is not None:
                plt.gca().get_legend().remove()
            plt.subplots_adjust(left=0.08, right=0.98, top=0.76, bottom=0.30)
            plt.savefig(out_dir / f"{metric}_{condition}_local_vs_cloud.png", dpi=180)
            plt.close()


def generate_pair_comparisons(results_dir: Path, output_root: Path) -> None:
    pairs = [
        ("ollama_qwen3_latest", "cloudflare_cf_qwen_qwen3_30b_a3b_fp8"),
        ("ollama_deepseek_r1_latest", "cloudflare_cf_deepseek_ai_deepseek_r1_distill_qwen_32b"),
        ("ollama_gemma4_latest", "cloudflare_cf_google_gemma_4_26b_a4b_it"),
        ("ollama_mistral_latest", "mistral_mistral_small_latest"),
        ("ollama_llama3_1_8b", "groq_llama_3_1_8b_instant"),
    ]
    for local_model, cloud_model in pairs:
        generate_pair_comparison(results_dir, output_root, local_model, cloud_model)
        print(f"[plots] wrote {output_root / f'{pair_folder_name(local_model)}_vs_{pair_folder_name(cloud_model)}'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate model-level plots from ArcShield evaluation outputs.")
    parser.add_argument("--results-dir", default="output/model_results", help="Directory containing model result folders.")
    parser.add_argument("--output-dir", default="output/plots", help="Directory to write plots into.")
    parser.add_argument("--model", action="append", help="Specific model folder name to plot. Can be repeated.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_dirs = [
        p for p in sorted(results_dir.iterdir())
        if (
            p.is_dir()
            and not p.name.startswith("github_")
            and p.name not in PLOT_EXCLUDED_MODELS
        )
    ]
    if args.model:
        wanted = set(args.model)
        model_dirs = [p for p in model_dirs if p.name in wanted]

    sns.set_theme(style="whitegrid", context="notebook", palette="bright")
    for model_dir in model_dirs:
        generate_for_model(model_dir, output_dir)
        print(f"[plots] wrote {output_dir / model_dir.name}")

    generate_pair_comparisons(results_dir, output_dir)
    generate_all_model_comparison(results_dir, output_dir)
    print(f"[plots] wrote {output_dir / 'all_models_comparison'}")
    generate_article_plots(results_dir, output_dir)
    print(f"[plots] wrote {output_dir / 'article_plots'}")


if __name__ == "__main__":
    main()
