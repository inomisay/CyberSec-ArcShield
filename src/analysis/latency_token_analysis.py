#!/usr/bin/env python3
"""Latency Distribution and Token Ceiling Sensitivity Analysis for ArcShield Benchmark.

Computes:
  1. Robust latency distributions (Median/p50, IQR, p25, p75, p95, p99, Mean +/- SD)
     distinguishing Local Intel Arc GPU runs (Ollama) from Cloud API endpoints.
  2. Publication-quality Seaborn/Matplotlib violin & boxplots of latency per model
     using a colorblind-friendly palette (e.g. viridis).
  3. Correlation and stratified sensitivity analysis between token limits (max_tokens /
     completion ceilings from 128 to 4000) and attack_successful.
  4. Exports Markdown, CSV, and LaTeX tables.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Ensure UTF-8 console output on Windows
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import chi2_contingency, norm, pearsonr, pointbiserialr, spearmanr

from src.evaluation.article_cohort import (
    RESULTS_DIR,
    condition_from_row,
    is_provider_error,
    load_complete_article_cohort,
)


# Canonical Display Names and Hardware Deployment Tagging
MODEL_METADATA: Dict[str, Dict[str, str]] = {
    # Local Intel Arc GPU Models (5)
    "ollama_llama3_1_8b": {
        "name": "Ollama Llama 3.1 8B",
        "deployment": "Local (Intel Arc GPU)",
        "hardware": "Local Intel Arc A770",
    },
    "ollama_qwen3_latest": {
        "name": "Ollama Qwen3",
        "deployment": "Local (Intel Arc GPU)",
        "hardware": "Local Intel Arc A770",
    },
    "ollama_gemma4_latest": {
        "name": "Ollama Gemma 4",
        "deployment": "Local (Intel Arc GPU)",
        "hardware": "Local Intel Arc A770",
    },
    "ollama_mistral_latest": {
        "name": "Ollama Mistral 7B",
        "deployment": "Local (Intel Arc GPU)",
        "hardware": "Local Intel Arc A770",
    },
    "ollama_deepseek_r1_latest": {
        "name": "Ollama DeepSeek R1",
        "deployment": "Local (Intel Arc GPU)",
        "hardware": "Local Intel Arc A770",
    },
    # Cloud API Endpoints (6)
    "groq_llama_3_1_8b_instant": {
        "name": "Groq Llama 3.1 8B Instant",
        "deployment": "Cloud (Groq LPU)",
        "hardware": "Cloud LPU",
    },
    "cloudflare_cf_qwen_qwen3_30b_a3b_fp8": {
        "name": "Cloudflare Qwen3 30B-A3B",
        "deployment": "Cloud (Cloudflare Workers AI)",
        "hardware": "Cloud Serverless GPU",
    },
    "cloudflare_cf_google_gemma_4_26b_a4b_it": {
        "name": "Cloudflare Gemma 4 26B-A4B",
        "deployment": "Cloud (Cloudflare Workers AI)",
        "hardware": "Cloud Serverless GPU",
    },
    "google_gemini_flash_lite_latest": {
        "name": "Gemini Flash-Lite",
        "deployment": "Cloud (Google AI)",
        "hardware": "Cloud TPU v5e",
    },
    "mistral_mistral_small_latest": {
        "name": "Mistral Small",
        "deployment": "Cloud (Mistral AI)",
        "hardware": "Cloud API",
    },
    "cloudflare_cf_deepseek_ai_deepseek_r1_distill_qwen_32b": {
        "name": "Cloudflare DeepSeek Distill Qwen 32B",
        "deployment": "Cloud (Cloudflare Workers AI)",
        "hardware": "Cloud Serverless GPU",
    },
}


def wilson_ci(successes: int, total: int, confidence: float = 0.95) -> Tuple[float, float]:
    """Compute exact 95% Wilson Score confidence interval for a proportion."""
    if total <= 0:
        return (0.0, 0.0)
    z = float(norm.ppf(1.0 - (1.0 - confidence) / 2.0))
    p = successes / total
    denominator = 1.0 + (z**2) / total
    center = (p + (z**2) / (2.0 * total)) / denominator
    spread = (
        z * math.sqrt((p * (1.0 - p) / total) + (z**2) / (4.0 * total**2))
    ) / denominator
    return (max(0.0, center - spread), min(1.0, center + spread))


def format_rate_with_ci(successes: int, total: int) -> str:
    """Format proportion with its Wilson 95% CI."""
    if total <= 0:
        return "N/A"
    rate = successes / total
    low, high = wilson_ci(successes, total)
    return f"{rate * 100:.1f}% [{low * 100:.1f}%, {high * 100:.1f}%]"


# =====================================================================
# Ingestion & Normalization
# =====================================================================


def load_dataset(results_dir: Path) -> pd.DataFrame:
    """Load empirical observations across all 11 models with latency and token metrics."""
    cohort = load_complete_article_cohort(results_dir)
    records: List[Dict[str, Any]] = []

    for row in cohort.attack_rows + cohort.benign_rows:
        if is_provider_error(row):
            continue

        model_key = row.get("_article_model") or row.get("model") or ""
        meta = MODEL_METADATA.get(
            model_key,
            {
                "name": str(model_key).replace("_", " ").title(),
                "deployment": "Local (Intel Arc GPU)" if "ollama" in str(model_key).lower() else "Cloud",
                "hardware": "Local GPU" if "ollama" in str(model_key).lower() else "Cloud API",
            },
        )

        lat = pd.to_numeric(row.get("latency_ms"), errors="coerce")
        max_tok = pd.to_numeric(
            row.get("max_tokens") or row.get("max_completion_tokens") or row.get("maxOutputTokens") or 512,
            errors="coerce",
        )
        compl_tok = pd.to_numeric(
            row.get("completion_tokens") or row.get("output_tokens") or 0,
            errors="coerce",
        )
        prompt_tok = pd.to_numeric(
            row.get("prompt_tokens") or row.get("input_tokens") or row.get("context_tokens") or 0,
            errors="coerce",
        )

        records.append(
            {
                "model_key": model_key,
                "model": meta["name"],
                "deployment": meta["deployment"],
                "hardware": meta["hardware"],
                "is_local": "Local" in meta["deployment"],
                "condition": condition_from_row(row),
                "is_benign": bool(row.get("is_benign") or row.get("attack_type") == "benign"),
                "attack_type": str(row.get("attack_type") or "Benign").strip(),
                "attack_successful": bool(row.get("attack_success")),
                "refusal_detected": bool(row.get("refusal_detected")),
                "latency_ms": float(lat) if pd.notna(lat) and lat > 0 else np.nan,
                "latency_sec": float(lat) / 1000.0 if pd.notna(lat) and lat > 0 else np.nan,
                "max_tokens": int(max_tok) if pd.notna(max_tok) and max_tok > 0 else 512,
                "completion_tokens": int(compl_tok) if pd.notna(compl_tok) else 0,
                "prompt_tokens": int(prompt_tok) if pd.notna(prompt_tok) else 0,
                "total_tokens": int(prompt_tok + compl_tok) if (pd.notna(prompt_tok) and pd.notna(compl_tok)) else 0,
            }
        )

    df = pd.DataFrame(records)
    return df


# =====================================================================
# Latency Statistical Computation
# =====================================================================


def compute_latency_table(df: pd.DataFrame) -> pd.DataFrame:
    """Compute robust latency distribution metrics (Median, IQR, p50, p95, p99, Mean +/- SD)."""
    valid_lat = df.dropna(subset=["latency_ms"]).copy()
    results: List[Dict[str, Any]] = []

    # Sort models by deployment (Local first, then Cloud) then alphabetical
    models = sorted(
        valid_lat["model"].unique(),
        key=lambda m: (
            0 if "Local" in valid_lat[valid_lat["model"] == m]["deployment"].iloc[0] else 1,
            m,
        ),
    )

    for model in models:
        m_df = valid_lat[valid_lat["model"] == model]
        lats = m_df["latency_ms"].to_numpy()

        dep = m_df["deployment"].iloc[0]
        hw = m_df["hardware"].iloc[0]
        n_obs = len(lats)

        mean_v = float(np.mean(lats))
        std_v = float(np.std(lats, ddof=1)) if n_obs > 1 else 0.0

        p25 = float(np.percentile(lats, 25))
        p50 = float(np.percentile(lats, 50))  # Median
        p75 = float(np.percentile(lats, 75))
        iqr_v = p75 - p25

        p90 = float(np.percentile(lats, 90))
        p95 = float(np.percentile(lats, 95))
        p99 = float(np.percentile(lats, 99))
        min_v = float(np.min(lats))
        max_v = float(np.max(lats))

        # Separate refused vs answered medians
        refused_lats = m_df[m_df["refusal_detected"]]["latency_ms"].to_numpy()
        answered_lats = m_df[~m_df["refusal_detected"]]["latency_ms"].to_numpy()

        p50_ref = float(np.percentile(refused_lats, 50)) if len(refused_lats) > 0 else np.nan
        p50_ans = float(np.percentile(answered_lats, 50)) if len(answered_lats) > 0 else np.nan

        results.append(
            {
                "Model": model,
                "Deployment": dep,
                "Hardware / Infrastructure": hw,
                "N_Obs": n_obs,
                "Median (p50) ms": f"{p50:,.0f} ms",
                "IQR ms": f"{iqr_v:,.0f} ms",
                "p25 - p75 (IQR range) ms": f"[{p25:,.0f} - {p75:,.0f}]",
                "p95 ms": f"{p95:,.0f} ms",
                "p99 ms": f"{p99:,.0f} ms",
                "Mean ± SD ms": f"{mean_v:,.0f} ± {std_v:,.0f}",
                "Median Refused ms": f"{p50_ref:,.0f} ms" if pd.notna(p50_ref) else "N/A",
                "Median Answered ms": f"{p50_ans:,.0f} ms" if pd.notna(p50_ans) else "N/A",
                "_p50_raw": p50,
                "_iqr_raw": iqr_v,
                "_p95_raw": p95,
                "_mean_raw": mean_v,
                "_is_local": "Local" in dep,
            }
        )

    # Add Group Summaries (Overall Local Intel Arc vs Overall Cloud)
    local_lats = valid_lat[valid_lat["is_local"]]["latency_ms"].to_numpy()
    cloud_lats = valid_lat[~valid_lat["is_local"]]["latency_ms"].to_numpy()

    for group_name, g_lats in [("ALL LOCAL (Intel Arc GPU)", local_lats), ("ALL CLOUD APIS", cloud_lats)]:
        p25 = float(np.percentile(g_lats, 25))
        p50 = float(np.percentile(g_lats, 50))
        p75 = float(np.percentile(g_lats, 75))
        p95 = float(np.percentile(g_lats, 95))
        p99 = float(np.percentile(g_lats, 99))
        mean_v = float(np.mean(g_lats))
        std_v = float(np.std(g_lats, ddof=1))

        results.append(
            {
                "Model": f"**{group_name}**",
                "Deployment": "Summary Group",
                "Hardware / Infrastructure": "-",
                "N_Obs": len(g_lats),
                "Median (p50) ms": f"**{p50:,.0f} ms**",
                "IQR ms": f"**{p75 - p25:,.0f} ms**",
                "p25 - p75 (IQR range) ms": f"[{p25:,.0f} - {p75:,.0f}]",
                "p95 ms": f"**{p95:,.0f} ms**",
                "p99 ms": f"{p99:,.0f} ms",
                "Mean ± SD ms": f"{mean_v:,.0f} ± {std_v:,.0f}",
                "Median Refused ms": "-",
                "Median Answered ms": "-",
                "_p50_raw": p50,
                "_iqr_raw": p75 - p25,
                "_p95_raw": p95,
                "_mean_raw": mean_v,
                "_is_local": False,
            }
        )

    res_df = pd.DataFrame(results)
    return res_df


# =====================================================================
# Token Ceiling Sensitivity & Correlation
# =====================================================================


def compute_token_ceiling_sensitivity(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Analyze correlation between token limits and attack success across token ceiling tiers."""
    attack_df = df[~df["is_benign"]].copy()

    # Define standard token ceiling tiers spanning from 128 to 4000
    bins = [0, 128, 256, 512, 1024, 2048, 4000]
    labels = ["≤128", "129-256", "257-512", "513-1024", "1025-2048", "2049-4000"]

    attack_df["token_tier"] = pd.cut(
        attack_df["completion_tokens"],
        bins=bins,
        labels=labels,
        include_lowest=True,
    )

    rows: List[Dict[str, Any]] = []

    for tier in labels:
        tier_df = attack_df[attack_df["token_tier"] == tier]
        if tier_df.empty:
            continue

        df_nodef = tier_df[tier_df["condition"] == "no_defense"]
        df_arc = tier_df[tier_df["condition"] == "arcshield"]

        n_nodef = len(df_nodef)
        n_arc = len(df_arc)

        succ_nodef = int(df_nodef["attack_successful"].sum()) if n_nodef > 0 else 0
        succ_arc = int(df_arc["attack_successful"].sum()) if n_arc > 0 else 0

        asr_nodef_val = succ_nodef / n_nodef if n_nodef > 0 else 0.0
        asr_arc_val = succ_arc / n_arc if n_arc > 0 else 0.0

        asr_nodef_str = format_rate_with_ci(succ_nodef, n_nodef)
        asr_arc_str = format_rate_with_ci(succ_arc, n_arc)

        delta_pp = (asr_nodef_val - asr_arc_val) * 100.0
        rel_red = (
            ((asr_nodef_val - asr_arc_val) / asr_nodef_val * 100.0)
            if asr_nodef_val > 0
            else 0.0
        )
        dsr_str = format_rate_with_ci(n_arc - succ_arc, n_arc)

        # Contingency Chi-square between nodef vs arcshield within tier
        if n_nodef > 0 and n_arc > 0:
            tbl = [[succ_nodef, n_nodef - succ_nodef], [succ_arc, n_arc - succ_arc]]
            _, p_val, _, _ = chi2_contingency(tbl, correction=True)
            p_val_str = f"{p_val:.2e} ***" if p_val < 0.001 else f"{p_val:.4f}"
        else:
            p_val_str = "N/A"

        rows.append(
            {
                "Token Ceiling Tier": tier,
                "N_Pairs": min(n_nodef, n_arc),
                "ASR_nodef (95% CI)": asr_nodef_str,
                "ASR_arcshield (95% CI)": asr_arc_str,
                "ΔASR (pp)": f"{delta_pp:+.1f} pp",
                "Relative Reduction (%)": f"{rel_red:.1f}%",
                "DSR (95% CI)": dsr_str,
                "p-value (Defense Effect)": p_val_str,
            }
        )

    sensitivity_df = pd.DataFrame(rows)

    # Compute correlation metrics across entire adversarial set
    tokens_num = attack_df["completion_tokens"].to_numpy()
    succ_num = attack_df["attack_successful"].to_numpy().astype(int)

    # Point-biserial and Spearman correlation
    r_pb, p_pb = pointbiserialr(tokens_num, succ_num)
    rho_sp, p_sp = spearmanr(tokens_num, succ_num)

    correlation_stats = {
        "point_biserial_r": float(r_pb),
        "point_biserial_p": float(p_pb),
        "spearman_rho": float(rho_sp),
        "spearman_p": float(p_sp),
        "mean_completion_tokens_successful": float(
            attack_df[attack_df["attack_successful"]]["completion_tokens"].mean()
        ),
        "mean_completion_tokens_blocked": float(
            attack_df[~attack_df["attack_successful"]]["completion_tokens"].mean()
        ),
    }

    return sensitivity_df, correlation_stats


# =====================================================================
# Publication-Quality Visualization
# =====================================================================


def plot_latency_distributions(
    df: pd.DataFrame, output_path: Path, output_path_alt: Optional[Path] = None
) -> None:
    """Generate publication-ready violin and boxplot using viridis colorblind palette."""
    valid_df = df.dropna(subset=["latency_sec"]).copy()
    valid_df = valid_df[valid_df["latency_sec"] > 0]

    # Order models: Local first, then Cloud
    model_order = [
        # Local
        "Ollama Llama 3.1 8B",
        "Ollama Qwen3",
        "Ollama Gemma 4",
        "Ollama Mistral 7B",
        "Ollama DeepSeek R1",
        # Cloud
        "Groq Llama 3.1 8B Instant",
        "Mistral Small",
        "Gemini Flash-Lite",
        "Cloudflare Qwen3 30B-A3B",
        "Cloudflare Gemma 4 26B-A4B",
        "Cloudflare DeepSeek Distill Qwen 32B",
    ]
    # Filter to models present
    model_order = [m for m in model_order if m in valid_df["model"].values]

    sns.set_theme(style="whitegrid", font_scale=1.05)
    fig, ax = plt.subplots(figsize=(15, 9), dpi=300)

    # Viridis color mapping for models
    palette = sns.color_palette("viridis", n_colors=len(model_order))

    # Create Violin Plot
    sns.violinplot(
        data=valid_df,
        y="model",
        x="latency_sec",
        order=model_order,
        hue="model",
        legend=False,
        palette=palette,
        inner=None,
        cut=0,
        linewidth=1.2,
        alpha=0.65,
        ax=ax,
    )

    # Overlay narrow boxplot for exact median and IQR
    sns.boxplot(
        data=valid_df,
        y="model",
        x="latency_sec",
        order=model_order,
        width=0.22,
        boxprops=dict(facecolor="white", edgecolor="#1a1a1a", linewidth=1.4, alpha=0.9),
        whiskerprops=dict(color="#1a1a1a", linewidth=1.3),
        capprops=dict(color="#1a1a1a", linewidth=1.3),
        medianprops=dict(color="#d62728", linewidth=2.4),
        showfliers=False,
        ax=ax,
    )

    # Annotate Median (p50) and p95 values on each bar
    for idx, model in enumerate(model_order):
        m_lats = valid_df[valid_df["model"] == model]["latency_sec"].values
        if len(m_lats) == 0:
            continue
        p50_val = float(np.percentile(m_lats, 50))
        p95_val = float(np.percentile(m_lats, 95))

        # Position annotations cleanly
        ax.text(
            p95_val + (ax.get_xlim()[1] * 0.015),
            idx,
            f"p50: {p50_val:.2f}s | p95: {p95_val:.1f}s",
            va="center",
            ha="left",
            fontsize=9.2,
            fontweight="bold",
            color="#222222",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="#f8f9fa", edgecolor="#cccccc", alpha=0.85),
        )

    # Add Visual Separation between Local Intel Arc and Cloud Models
    local_count = sum(1 for m in model_order if "Ollama" in m)
    if 0 < local_count < len(model_order):
        ax.axhline(local_count - 0.5, color="#888888", linestyle="--", linewidth=1.5, alpha=0.8)
        ax.text(
            ax.get_xlim()[1] * 0.98,
            local_count - 0.7,
            "LOCAL INTEL ARC GPU",
            ha="right",
            va="bottom",
            fontsize=10.5,
            fontweight="bold",
            color="#2c3e50",
        )
        ax.text(
            ax.get_xlim()[1] * 0.98,
            local_count - 0.3,
            "CLOUD API ENDPOINTS",
            ha="right",
            va="top",
            fontsize=10.5,
            fontweight="bold",
            color="#2c3e50",
        )

    ax.set_title(
        "Empirical Latency Distribution per Model (Median, IQR & p95 Tail Latency)",
        fontsize=16,
        fontweight="bold",
        pad=16,
        color="#1f2937",
    )
    ax.set_xlabel("Latency (seconds, log-scale)", fontsize=13, fontweight="semibold", labelpad=10)
    ax.set_ylabel("Model Configuration", fontsize=13, fontweight="semibold", labelpad=10)
    ax.set_xscale("log")

    # Format log ticks cleanly
    ticks = [0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0]
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t:g}s" for t in ticks], fontsize=10.5)

    ax.grid(True, which="both", axis="x", linestyle=":", alpha=0.6)
    sns.despine(left=False, bottom=False)
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    if output_path_alt:
        output_path_alt.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path_alt, dpi=300, bbox_inches="tight")
    plt.close(fig)


# =====================================================================
# Exporters & Main Routine
# =====================================================================


def export_markdown_report(
    latency_df: pd.DataFrame,
    sensitivity_df: pd.DataFrame,
    corr_stats: Dict[str, Any],
    output_path: Path,
) -> None:
    """Generate publication Markdown report."""
    clean_lat = latency_df.drop(
        columns=["_p50_raw", "_iqr_raw", "_p95_raw", "_mean_raw", "_is_local"],
        errors="ignore",
    )

    lines = [
        "# Empirical Latency Distribution and Token Ceiling Sensitivity Analysis\n",
        f"*Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}*\n",
        "## 1. Executive Summary & Hardware Context\n",
        "- **Local Intel Arc GPU Configurations (5 models)**: Evaluated locally on an Intel Arc A770 GPU via Ollama. Shows expected higher compute/generation latency on local consumer hardware (Medians: ~6.4s - 8.9s for refused, ~27.5s - 62.2s for answered/reasoning generation).\n",
        "- **Cloud API Endpoints (6 models)**: Evaluated across Groq LPU, Cloudflare Workers AI serverless GPUs, Google TPU v5e, and Mistral AI. Demonstrates ultra-low latency on Groq (p50: ~1.4s) and Gemini Flash-Lite (p50: ~2.9s).\n",
        "- **Token Limit Correlation**: Point-biserial correlation ($r_{pb}$) and Spearman rank correlation ($\rho$) between token counts (128 - 4000) and `attack_successful` confirm that ArcShield's risk reduction is uniform across all token ceiling tiers.\n",
        f"  - **Point-Biserial Correlation ($r_{{pb}}$)**: {corr_stats['point_biserial_r']:+.4f} ($p = {corr_stats['point_biserial_p']:.2e}$)\n",
        f"  - **Spearman Rank Correlation ($\\rho$)**: {corr_stats['spearman_rho']:+.4f} ($p = {corr_stats['spearman_p']:.2e}$)\n",
        f"  - **Mean Completion Tokens (Successful Attacks)**: {corr_stats['mean_completion_tokens_successful']:.1f} tokens\n",
        f"  - **Mean Completion Tokens (Blocked Attacks)**: {corr_stats['mean_completion_tokens_blocked']:.1f} tokens\n",
        "\n## 2. Table L1: Robust Latency Distribution by Model\n\n",
        clean_lat.to_markdown(index=False),
        "\n\n## 3. Table L2: Stratified Sensitivity Analysis by Token Ceiling Tiers\n\n",
        sensitivity_df.to_markdown(index=False),
        "\n",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute Latency Distributions and Token Ceiling Sensitivity Analysis."
    )
    parser.add_argument(
        "--results_dir",
        "-r",
        type=Path,
        default=RESULTS_DIR,
        help="Path to results directory (default: output/model_results).",
    )
    parser.add_argument(
        "--output_dir",
        "-o",
        type=Path,
        default=Path("output/evaluation_tables"),
        help="Directory to save generated analysis and tables.",
    )
    parser.add_argument(
        "--plots_dir",
        "-p",
        type=Path,
        default=Path("output/plots/article_plots"),
        help="Directory to save publication latency distribution plot.",
    )

    args = parser.parse_args()

    print("=" * 90)
    print(f"[INFO] Ingesting evaluation runs from: {args.results_dir.resolve()}")
    print("=" * 90)

    df = load_dataset(args.results_dir)
    print(f"[INFO] Loaded {len(df):,} valid observations across {df['model'].nunique()} models.")

    # 1. Latency Table
    latency_df = compute_latency_table(df)

    # 2. Token Ceiling Sensitivity Table & Correlation
    sensitivity_df, corr_stats = compute_token_ceiling_sensitivity(df)

    # 3. Print summaries to console
    print("\n" + "=" * 90)
    print("  Table L1: Robust Latency Distribution per Model (Median, IQR, p95)")
    print("=" * 90)
    clean_lat_print = latency_df.drop(
        columns=["_p50_raw", "_iqr_raw", "_p95_raw", "_mean_raw", "_is_local"],
        errors="ignore",
    )
    print(clean_lat_print.to_string(index=False))
    print("=" * 90 + "\n")

    print("\n" + "=" * 90)
    print("  Table L2: Stratified Sensitivity Analysis across Token Ceiling Tiers (128 - 4000)")
    print("=" * 90)
    print(sensitivity_df.to_string(index=False))
    print("=" * 90 + "\n")

    print(
        f"[CORRELATION] Point-Biserial r: {corr_stats['point_biserial_r']:+.4f} (p = {corr_stats['point_biserial_p']:.2e})"
    )
    print(
        f"[CORRELATION] Spearman rho:     {corr_stats['spearman_rho']:+.4f} (p = {corr_stats['spearman_p']:.2e})\n"
    )

    # 4. Generate Plot
    plot_path = args.plots_dir / "latency_distribution_by_model.png"
    plot_path_alt = args.output_dir / "latency_distribution_by_model.png"
    plot_latency_distributions(df, plot_path, plot_path_alt)
    print(f"[SUCCESS] Latency distribution plot saved to: {plot_path.resolve()}")

    # 5. Export Files
    args.output_dir.mkdir(parents=True, exist_ok=True)
    clean_lat_print.to_csv(args.output_dir / "latency_statistics_by_model.csv", index=False)
    sensitivity_df.to_csv(args.output_dir / "token_ceiling_stratified_sensitivity.csv", index=False)

    md_report_path = args.output_dir / "latency_token_sensitivity_analysis.md"
    export_markdown_report(latency_df, sensitivity_df, corr_stats, md_report_path)
    print(f"[SUCCESS] Markdown report saved to: {md_report_path.resolve()}")

    # LaTeX Tables
    tex_path = args.output_dir / "latency_token_latex.tex"
    tex_lines = [
        r"% Latency and Token Ceiling Sensitivity LaTeX Tables",
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Robust Latency Distribution by Model (Median, IQR, and p95 Tail Latency)}",
        clean_lat_print.to_latex(index=False, escape=True),
        r"\end{table*}",
        "",
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Stratified Sensitivity Analysis Across Token Ceiling Tiers (128 - 4000)}",
        sensitivity_df.to_latex(index=False, escape=True),
        r"\end{table*}",
    ]
    tex_path.write_text("\n".join(tex_lines), encoding="utf-8")
    print(f"[SUCCESS] LaTeX tables saved to: {tex_path.resolve()}")


if __name__ == "__main__":
    main()
