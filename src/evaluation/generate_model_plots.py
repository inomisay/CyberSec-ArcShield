import argparse
import json
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter
import pandas as pd
import seaborn as sns


ATTACK_RESULT_FILE = "results.json"
BENIGN_RESULT_FILE = "benign_results.json"
CONDITION_PALETTE = {
    "no_defense": "#F6A6C8",
    "arcshield": "#9CCBFF",
}
MODEL_COLORS = ["#F6A6C8", "#9CCBFF", "#C7B9FF", "#A7E8C5", "#FFD6A5", "#B8E0D2", "#FFCAD4", "#BDE0FE"]
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


def friendly_model_name(model_name: str) -> str:
    replacements = {
        "ollama_llama3_1_8b": "OLLAMA LLAMA3.1",
        "ollama_qwen3_latest": "OLLAMA QWEN3",
        "ollama_mistral_latest": "OLLAMA MISTRAL",
        "google_gemini_flash_lite_latest": "GEMINI",
        "groq_llama_3_1_8b_instant": "GROQ LLAMA3.1",
        "mistral_mistral_small_latest": "MISTRAL",
        "cloudflare_cf_qwen_qwen3_30b_a3b_fp8": "CLOUDFLARE QWEN",
        "cloudflare_cf_deepseek_ai_deepseek_r1_distill_qwen_32b": "CLOUDFLARE DEEPSEEK",
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

        combined_path = run_dir / "combined" / ATTACK_RESULT_FILE
        if not combined_path.exists():
            skipped.append(f"{run_dir.name}: missing combined/{ATTACK_RESULT_FILE}")
            continue

        data = load_json(combined_path)
        if not isinstance(data, list):
            skipped.append(f"{run_dir.name}: combined result is not a list")
            continue
        if len(data) < 400:
            skipped.append(f"{run_dir.name}: incomplete combined rows ({len(data)}/400)")
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

        combined_path = run_dir / "combined" / BENIGN_RESULT_FILE
        report_path = run_dir / "combined" / "brr_report.json"

        if combined_path.exists():
            data = load_json(combined_path)
            if isinstance(data, list) and len(data) >= 400:
                for row in data:
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
        if not model_dir.is_dir() or model_dir.name.startswith("github_"):
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
    ordered = points.sort_values(["latency_mean_ms", "attack_success_rate"], ascending=[True, True]).copy()
    frontier_rows = []
    best_asr = float("inf")
    for _, row in ordered.iterrows():
        asr = float(row["attack_success_rate"])
        if asr < best_asr:
            frontier_rows.append(row)
            best_asr = asr
    return pd.DataFrame(frontier_rows)


def save_strategic_pareto(summary: pd.DataFrame, out_dir: Path, condition: str) -> None:
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
    model_summary.to_csv(out_dir / f"strategic_pareto_{condition}.csv", index=False)

    colors = {
        model: MODEL_COLORS[index % len(MODEL_COLORS)]
        for index, model in enumerate(sorted(model_summary["model"].unique()))
    }
    x_min = max(1.0, float(model_summary["latency_mean_ms"].min()) * 0.65)
    x_max = float(model_summary["latency_mean_ms"].max()) * 1.45
    y_max = max(1.0, float(model_summary["attack_success_rate"].max()) * 1.25)

    plt.figure(figsize=(14, 8.5))
    ax = plt.gca()
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
        size = 240 + min(float(row["total"]), 6000) / 6000 * 700
        ax.scatter(
            x,
            y,
            s=size,
            color=colors[row["model"]],
            edgecolor="white",
            linewidth=1.8,
            alpha=0.95,
            label=(
                f"{row['model_label']} ({row['profile']})\n"
                f"{x:.0f}ms | {y * 100:.1f}% ASR"
            ),
        )

    frontier = pareto_frontier(model_summary)
    if len(frontier) > 1:
        ax.plot(
            frontier["latency_mean_ms"],
            frontier["attack_success_rate"],
            linestyle="--",
            linewidth=2.5,
            color="#5F7480",
            alpha=0.95,
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
        f"Security vs. Efficiency (Strategic Pareto) - {display_condition(condition)}",
        "Security Breach Rate (ASR %)",
        ylim=(0, y_max),
        percent=True,
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
    plt.tight_layout()
    plt.savefig(out_dir / f"strategic_pareto_{condition}.png", dpi=180)
    plt.close()


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
        "ollama_mistral_latest": "Local Mistral",
        "mistral_mistral_small_latest": "Cloud Mistral",
    }
    return names.get(model_name, model_name.replace("cloudflare_cf_", "Cloud ").replace("ollama_", "Local "))


def pair_folder_name(model_name: str) -> str:
    return model_name.replace("cloudflare_cf_", "cloud_").replace("ollama_", "local_")


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
        if p.is_dir() and not p.name.startswith("github_")
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


if __name__ == "__main__":
    main()
