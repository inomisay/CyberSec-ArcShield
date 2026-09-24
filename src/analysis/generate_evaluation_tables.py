#!/usr/bin/env python3
"""Evaluation Results Table Generator for ArcShield Benchmark.

Loads the empirical benchmark evaluation dataset across all 11 model configurations
(74,800 observations: 70,400 adversarial across 16 attack categories + 4,400 benign
across 200 benign evaluation prompts) and computes publication-ready tables:

  - Table R1 (Model-level Attack Performance & Paired Statistical Significance)
  - Table R2 (Category-level Attack Breakdown across all 16 Attack Categories)
  - Table R3 (Benign Usability, Benign Refusal Rate & False-Flag Rate)
  - Table R4 (Efficiency, Latency Breakdown for Refused vs Answered & Token Overhead)

Outputs:
  - Formatted Markdown in output/evaluation_tables/evaluation_results_tables.md
  - Ready-to-copy LaTeX in output/evaluation_tables/tables_latex.tex
  - Clean CSVs for each table in output/evaluation_tables/
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Ensure UTF-8 console output on Windows
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import numpy as np
import pandas as pd
from scipy.stats import binomtest, norm

from src.evaluation.article_cohort import (
    RESULTS_DIR,
    condition_from_row,
    is_provider_error,
    load_complete_article_cohort,
)
from src.core.judge import ATTACK_TYPES


# Canonical Display Names for the 11 Article Models
MODEL_DISPLAY_NAMES: Dict[str, str] = {
    # Local Models (5)
    "ollama_llama3_1_8b": "Ollama Llama 3.1 8B",
    "ollama_qwen3_latest": "Ollama Qwen3",
    "ollama_gemma4_latest": "Ollama Gemma 4",
    "ollama_mistral_latest": "Ollama Mistral 7B",
    "ollama_deepseek_r1_latest": "Ollama DeepSeek R1",
    # Cloud Models (6)
    "groq_llama_3_1_8b_instant": "Groq Llama 3.1 8B Instant",
    "cloudflare_cf_qwen_qwen3_30b_a3b_fp8": "Cloudflare Qwen3 30B-A3B",
    "cloudflare_cf_google_gemma_4_26b_a4b_it": "Cloudflare Gemma 4 26B-A4B",
    "google_gemini_flash_lite_latest": "Gemini Flash-Lite",
    "mistral_mistral_small_latest": "Mistral Small",
    "cloudflare_cf_deepseek_ai_deepseek_r1_distill_qwen_32b": "Cloudflare DeepSeek Distill Qwen 32B",
}


def get_friendly_model_name(model_key: str) -> str:
    """Return publication-formatted model name."""
    if model_key in MODEL_DISPLAY_NAMES:
        return MODEL_DISPLAY_NAMES[model_key]
    cleaned = model_key.replace("cloudflare_cf_", "").replace("ollama_", "").replace("_", " ")
    return cleaned.title()


def get_deployment_type(model_key: str) -> str:
    """Return deployment classification (Local vs Cloud)."""
    return "Local" if model_key.startswith("ollama_") else "Cloud"


# =====================================================================
# Statistical Computations
# =====================================================================


def wilson_ci(
    successes: int, total: int, confidence: float = 0.95
) -> Tuple[float, float]:
    """Compute exact 95% Wilson Score confidence interval for a binomial proportion."""
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


def format_rate_with_ci(
    successes: int, total: int, percent: bool = True
) -> str:
    """Format proportion with its Wilson 95% CI."""
    if total <= 0:
        return "N/A"
    rate = successes / total
    low, high = wilson_ci(successes, total)
    if percent:
        return f"{rate * 100:.1f}% [{low * 100:.1f}%, {high * 100:.1f}%]"
    return f"{rate:.3f} [{low:.3f}, {high:.3f}]"


def mcnemar_exact_and_or(
    y_nodef: np.ndarray, y_arc: np.ndarray
) -> Tuple[int, int, float, float, Tuple[float, float]]:
    """Compute McNemar discordant pairs (b, c), exact binomial p-value, and Odds Ratio with 95% CI.

    b: nodef=unsafe (1) -> arcshield=safe (0)  [Improvement / Defense Success]
    c: nodef=safe (0)   -> arcshield=unsafe (1) [Regression]

    Discordant Odds Ratio: OR = b / c (with Haldane-Anscombe 0.5 correction if zero).
    Standard error for log-odds: sqrt(1/b + 1/c).
    """
    b = int(np.sum((y_nodef == 1) & (y_arc == 0)))
    c = int(np.sum((y_nodef == 0) & (y_arc == 1)))
    discordant = b + c

    if discordant == 0:
        return b, c, 1.0, 1.0, (1.0, 1.0)

    # Exact two-sided binomial test on discordant pairs under H0: p=0.5
    res = binomtest(min(b, c), discordant, 0.5, alternative="two-sided")
    p_val = float(res.pvalue)

    # Haldane-Anscombe continuity correction for odds ratio & standard error
    b_adj = b + 0.5 if (b == 0 or c == 0) else float(b)
    c_adj = c + 0.5 if (b == 0 or c == 0) else float(c)

    odds_ratio = b_adj / c_adj
    se_log_or = math.sqrt(1.0 / b_adj + 1.0 / c_adj)
    z = 1.959964
    ci_low = math.exp(math.log(odds_ratio) - z * se_log_or)
    ci_high = math.exp(math.log(odds_ratio) + z * se_log_or)

    return b, c, p_val, odds_ratio, (ci_low, ci_high)


def holm_bonferroni_adjust(p_values: Sequence[float]) -> List[float]:
    """Perform Holm-Bonferroni step-down p-value adjustment."""
    m = len(p_values)
    if m == 0:
        return []
    indexed_p = sorted(enumerate(p_values), key=lambda x: x[1])
    adjusted = [0.0] * m
    running_max = 0.0

    for rank, (original_idx, p) in enumerate(indexed_p):
        candidate = min(1.0, (m - rank) * p)
        running_max = max(running_max, candidate)
        adjusted[original_idx] = running_max

    return adjusted


def format_p_value(p: float) -> str:
    """Format p-value without underflowing or displaying 0.0 / <4.94e-324."""
    if p == 0.0 or p < 1e-300:
        return "p < 1e-300"
    if p < 0.001:
        return f"{p:.2e}"
    return f"{p:.4f}"


# =====================================================================
# Dataset Ingestion & Pairing
# =====================================================================


def load_empirical_cohort(results_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load empirical article cohort (adversarial and benign) across the 11 models.

    Returns:
      valid_paired_attacks: DataFrame containing paired attack outcomes.
      benign_df: DataFrame containing benign evaluation outcomes.
    """
    cohort = load_complete_article_cohort(results_dir)

    # Convert attack rows
    attack_records: List[Dict[str, Any]] = []
    for row in cohort.attack_rows:
        attack_records.append(
            {
                "model": row["_article_model"],
                "attack_type": row.get("attack_type"),
                "condition": condition_from_row(row),
                "prompt_index": row.get("prompt_index"),
                "source_dataset": row.get("source_dataset", ""),
                "source_row": str(row.get("source_row", "")),
                "attack_success": bool(row.get("attack_success")),
                "refusal_detected": bool(row.get("refusal_detected")),
                "harmful_content_detected": bool(row.get("harmful_content_detected")),
                "latency_ms": pd.to_numeric(row.get("latency_ms"), errors="coerce"),
                "prompt_tokens": pd.to_numeric(row.get("prompt_tokens"), errors="coerce"),
                "context_tokens": pd.to_numeric(
                    row.get("context_tokens") or row.get("prompt_tokens"),
                    errors="coerce",
                ),
                "completion_tokens": pd.to_numeric(
                    row.get("completion_tokens") or row.get("output_tokens"),
                    errors="coerce",
                ),
                "provider_error": is_provider_error(row),
            }
        )
    raw_attacks = pd.DataFrame(attack_records)

    # Perform strict paired alignment
    keys = ["model", "attack_type", "prompt_index", "source_dataset", "source_row"]
    outcomes = raw_attacks.pivot_table(
        index=keys, columns="condition", values="attack_success", aggfunc="first"
    )
    refusals = raw_attacks.pivot_table(
        index=keys, columns="condition", values="refusal_detected", aggfunc="first"
    )
    latencies = raw_attacks.pivot_table(
        index=keys, columns="condition", values="latency_ms", aggfunc="first"
    )
    contexts = raw_attacks.pivot_table(
        index=keys, columns="condition", values="context_tokens", aggfunc="first"
    )
    completions = raw_attacks.pivot_table(
        index=keys, columns="condition", values="completion_tokens", aggfunc="first"
    )
    errors = raw_attacks.pivot_table(
        index=keys, columns="condition", values="provider_error", aggfunc="max"
    )

    joined = outcomes.join(
        errors, lsuffix="_success", rsuffix="_error"
    ).join(
        refusals.add_suffix("_refusal")
    ).join(
        latencies.add_suffix("_latency")
    ).join(
        contexts.add_suffix("_context_tokens")
    ).join(
        completions.add_suffix("_completion_tokens")
    ).reset_index()

    required = ["no_defense_success", "arcshield_success", "no_defense_error", "arcshield_error"]
    complete = joined.dropna(subset=required).copy()
    error_mask = complete["no_defense_error"].astype(bool) | complete["arcshield_error"].astype(bool)
    valid_paired = complete[~error_mask].copy()

    # Convert benign rows
    benign_records: List[Dict[str, Any]] = []
    for row in cohort.benign_rows:
        if is_provider_error(row):
            continue
        benign_records.append(
            {
                "model": row["_article_model"],
                "condition": condition_from_row(row),
                "refusal_detected": bool(row.get("refusal_detected")),
                "harmful_content_detected": bool(row.get("harmful_content_detected")),
                "latency_ms": pd.to_numeric(row.get("latency_ms"), errors="coerce"),
                "prompt_tokens": pd.to_numeric(
                    row.get("prompt_tokens") or row.get("context_tokens"),
                    errors="coerce",
                ),
                "completion_tokens": pd.to_numeric(
                    row.get("completion_tokens") or row.get("output_tokens"),
                    errors="coerce",
                ),
            }
        )
    benign_df = pd.DataFrame(benign_records)

    return valid_paired, benign_df


# =====================================================================
# Table R1: Model-Level Attack Performance
# =====================================================================


def generate_table_r1(paired_df: pd.DataFrame) -> pd.DataFrame:
    """Generate Table R1: Model-level attack evaluation with paired statistical tests."""
    results: List[Dict[str, Any]] = []
    models = sorted(paired_df["model"].unique())

    for model_key in models:
        m_df = paired_df[paired_df["model"] == model_key]
        n_pairs = len(m_df)
        if n_pairs == 0:
            continue

        y_nodef = m_df["no_defense_success"].to_numpy().astype(int)
        y_arc = m_df["arcshield_success"].to_numpy().astype(int)

        succ_nodef = int(np.sum(y_nodef == 1))
        succ_arc = int(np.sum(y_arc == 1))

        asr_nodef_val = succ_nodef / n_pairs
        asr_arc_val = succ_arc / n_pairs

        asr_nodef_str = format_rate_with_ci(succ_nodef, n_pairs)
        asr_arc_str = format_rate_with_ci(succ_arc, n_pairs)

        abs_reduction = (asr_nodef_val - asr_arc_val) * 100.0
        rel_reduction = (
            ((asr_nodef_val - asr_arc_val) / asr_nodef_val * 100.0)
            if asr_nodef_val > 0
            else 0.0
        )
        dsr_str = format_rate_with_ci(n_pairs - succ_arc, n_pairs)

        b, c, p_raw, odds_ratio, (or_low, or_high) = mcnemar_exact_and_or(
            y_nodef, y_arc
        )
        or_str = f"{odds_ratio:.2f} [{or_low:.2f}, {or_high:.2f}]"

        results.append(
            {
                "Model": get_friendly_model_name(model_key),
                "Deployment": get_deployment_type(model_key),
                "N_Pairs": n_pairs,
                "ASR_nodef (95% CI)": asr_nodef_str,
                "ASR_arcshield (95% CI)": asr_arc_str,
                "ΔASR (pp)": f"{abs_reduction:+.1f} pp",
                "Relative Reduction (%)": f"{rel_reduction:.1f}%",
                "DSR (95% CI)": dsr_str,
                "McNemar (b, c)": f"({b}, {c})",
                "Discordant OR (95% CI)": or_str,
                "_raw_p": p_raw,
                "_abs_red": abs_reduction,
            }
        )

    r1_df = pd.DataFrame(results)
    if not r1_df.empty:
        # Sort by Deployment (Cloud, Local) then ΔASR descending
        p_adj = holm_bonferroni_adjust(r1_df["_raw_p"].tolist())
        r1_df["Holm-adj p-value"] = [format_p_value(p) for p in p_adj]
        r1_df = r1_df.drop(columns=["_raw_p", "_abs_red"], errors="ignore")

    return r1_df


# =====================================================================
# Table R2: Category-Level Attack Performance
# =====================================================================


def generate_table_r2(paired_df: pd.DataFrame) -> pd.DataFrame:
    """Generate Table R2: Category-level breakdown across all 16 attack categories."""
    results: List[Dict[str, Any]] = []
    categories = sorted(paired_df["attack_type"].unique())

    for cat in categories:
        c_df = paired_df[paired_df["attack_type"] == cat]
        n_pairs = len(c_df)
        if n_pairs == 0:
            continue

        y_nodef = c_df["no_defense_success"].to_numpy().astype(int)
        y_arc = c_df["arcshield_success"].to_numpy().astype(int)

        succ_nodef = int(np.sum(y_nodef == 1))
        succ_arc = int(np.sum(y_arc == 1))

        asr_nodef_val = succ_nodef / n_pairs
        asr_arc_val = succ_arc / n_pairs

        asr_nodef_str = format_rate_with_ci(succ_nodef, n_pairs)
        asr_arc_str = format_rate_with_ci(succ_arc, n_pairs)

        abs_reduction = (asr_nodef_val - asr_arc_val) * 100.0
        rel_reduction = (
            ((asr_nodef_val - asr_arc_val) / asr_nodef_val * 100.0)
            if asr_nodef_val > 0
            else 0.0
        )
        dsr_str = format_rate_with_ci(n_pairs - succ_arc, n_pairs)

        b, c, p_raw, odds_ratio, (or_low, or_high) = mcnemar_exact_and_or(
            y_nodef, y_arc
        )
        or_str = f"{odds_ratio:.2f} [{or_low:.2f}, {or_high:.2f}]"

        results.append(
            {
                "Attack Category": cat,
                "N_Pairs": n_pairs,
                "ASR_nodef (95% CI)": asr_nodef_str,
                "ASR_arcshield (95% CI)": asr_arc_str,
                "ΔASR (pp)": f"{abs_reduction:+.1f} pp",
                "Relative Reduction (%)": f"{rel_reduction:.1f}%",
                "DSR (95% CI)": dsr_str,
                "McNemar (b, c)": f"({b}, {c})",
                "Discordant OR (95% CI)": or_str,
                "_raw_p": p_raw,
            }
        )

    r2_df = pd.DataFrame(results)
    if not r2_df.empty:
        p_adj = holm_bonferroni_adjust(r2_df["_raw_p"].tolist())
        r2_df["Holm-adj p-value"] = [format_p_value(p) for p in p_adj]
        r2_df = r2_df.drop(columns=["_raw_p"], errors="ignore")

    return r2_df


# =====================================================================
# Table R3: Benign Usability & False Refusal Rates
# =====================================================================


def generate_table_r3(benign_df: pd.DataFrame) -> pd.DataFrame:
    """Generate Table R3: Benign usability metrics (BRR, FFR, and ΔFFR with Wilson CI)."""
    results: List[Dict[str, Any]] = []
    models = sorted(benign_df["model"].unique())

    for model_key in models:
        m_df = benign_df[benign_df["model"] == model_key]
        df_nodef = m_df[m_df["condition"] == "no_defense"]
        df_arc = m_df[m_df["condition"] == "arcshield"]

        n_nodef = len(df_nodef)
        n_arc = len(df_arc)

        # Benign Refusal Rate (BRR) = refusal_detected on benign prompts
        brr_succ_nodef = int(df_nodef["refusal_detected"].sum())
        brr_succ_arc = int(df_arc["refusal_detected"].sum())

        brr_nodef_str = format_rate_with_ci(brr_succ_nodef, n_nodef)
        brr_arc_str = format_rate_with_ci(brr_succ_arc, n_arc)

        # False-Flag Rate (FFR) = harmful_content_detected on benign prompts
        ffr_succ_nodef = int(df_nodef["harmful_content_detected"].sum())
        ffr_succ_arc = int(df_arc["harmful_content_detected"].sum())

        ffr_nodef_val = ffr_succ_nodef / n_nodef if n_nodef > 0 else 0.0
        ffr_arc_val = ffr_succ_arc / n_arc if n_arc > 0 else 0.0

        ffr_nodef_str = format_rate_with_ci(ffr_succ_nodef, n_nodef)
        ffr_arc_str = format_rate_with_ci(ffr_succ_arc, n_arc)

        delta_ffr_pp = (ffr_arc_val - ffr_nodef_val) * 100.0

        results.append(
            {
                "Model": get_friendly_model_name(model_key),
                "Deployment": get_deployment_type(model_key),
                "N_Benign": max(n_nodef, n_arc),
                "BRR_nodef (95% CI)": brr_nodef_str,
                "BRR_arcshield (95% CI)": brr_arc_str,
                "FFR_nodef (95% CI)": ffr_nodef_str,
                "FFR_arcshield (95% CI)": ffr_arc_str,
                "ΔFFR (pp)": f"{delta_ffr_pp:+.2f} pp",
            }
        )

    return pd.DataFrame(results)


# =====================================================================
# Table R4: Efficiency, Latency Breakdown & Overhead
# =====================================================================


def generate_table_r4(paired_df: pd.DataFrame) -> pd.DataFrame:
    """Generate Table R4: Latency breakdown (Refused vs Answered) & Token Overhead."""
    results: List[Dict[str, Any]] = []
    models = sorted(paired_df["model"].unique())

    for model_key in models:
        m_df = paired_df[paired_df["model"] == model_key]

        # Combine latencies from paired conditions
        refused_lats: List[float] = []
        answered_lats: List[float] = []

        for cond in ["no_defense", "arcshield"]:
            ref_mask = m_df[f"{cond}_refusal"].astype(bool)
            lats = m_df[f"{cond}_latency"].dropna()

            refused_lats.extend(m_df.loc[ref_mask, f"{cond}_latency"].dropna().tolist())
            answered_lats.extend(m_df.loc[~ref_mask, f"{cond}_latency"].dropna().tolist())

        def calc_summary(values: List[float]) -> str:
            if not values:
                return "N/A"
            arr = np.array(values)
            mean_v = float(np.mean(arr))
            median_v = float(np.median(arr))
            q25, q75 = np.percentile(arr, [25, 75])
            iqr = q75 - q25
            return f"{mean_v:.0f} ms / {median_v:.0f} ms (IQR: {iqr:.0f})"

        lat_refused = calc_summary(refused_lats)
        lat_answered = calc_summary(answered_lats)

        # Context Overhead (CO %) - Relative increase in input context
        nodef_ctx_mean = float(m_df["no_defense_context_tokens"].dropna().mean())
        arc_ctx_mean = float(m_df["arcshield_context_tokens"].dropna().mean())

        co_pct = (
            ((arc_ctx_mean - nodef_ctx_mean) / nodef_ctx_mean * 100.0)
            if nodef_ctx_mean > 0
            else 0.0
        )

        # Token Overhead (TO %) - Relative change in total (context + completion) tokens
        nodef_compl_mean = float(m_df["no_defense_completion_tokens"].dropna().mean())
        arc_compl_mean = float(m_df["arcshield_completion_tokens"].dropna().mean())

        nodef_total = nodef_ctx_mean + nodef_compl_mean
        arc_total = arc_ctx_mean + arc_compl_mean

        to_pct = (
            ((arc_total - nodef_total) / nodef_total * 100.0)
            if nodef_total > 0
            else 0.0
        )

        results.append(
            {
                "Model": get_friendly_model_name(model_key),
                "Deployment": get_deployment_type(model_key),
                "Latency Refused (Mean / Med [IQR])": lat_refused,
                "Latency Answered (Mean / Med [IQR])": lat_answered,
                "Context Overhead (CO %)": f"{co_pct:+.1f}%",
                "Token Overhead (TO %)": f"{to_pct:+.1f}%",
                "Mean Compl. Tokens (nodef -> arc)": f"{nodef_compl_mean:.1f} -> {arc_compl_mean:.1f}",
            }
        )

    return pd.DataFrame(results)


# =====================================================================
# Export Utilities (Console, Markdown, LaTeX, CSV)
# =====================================================================


def print_table(title: str, df: pd.DataFrame) -> None:
    """Print ASCII table to standard output."""
    print("\n" + "=" * 90)
    print(f"  {title}")
    print("=" * 90)
    if df.empty:
        print("  [No data available]")
    else:
        print(df.to_string(index=False))
    print("=" * 90 + "\n")


def export_markdown(tables: Dict[str, pd.DataFrame], output_path: Path) -> None:
    """Export all tables to a unified Markdown document."""
    lines = [
        "# ArcShield Empirical Evaluation Benchmark Results\n",
        f"*Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}*\n",
        f"*Dataset: 74,800 total empirical observations (70,400 adversarial across 16 attack categories + 4,400 benign)*\n",
    ]

    for title, table_df in tables.items():
        lines.append(f"\n## {title}\n")
        if table_df.empty:
            lines.append("_No data available_\n")
        else:
            lines.append(table_df.to_markdown(index=False))
            lines.append("\n")

    output_path.write_text("\n".join(lines), encoding="utf-8")


def export_combined_latex(tables: Dict[str, pd.DataFrame], output_path: Path) -> None:
    """Export all tables to a publication LaTeX file."""
    latex_snippets = [
        r"% ArcShield Empirical Benchmark Evaluation Tables",
        r"% Generated for publication reporting",
        r"\usepackage{booktabs}",
        r"\usepackage{makecell}",
        r"\usepackage{tabularx}",
        "",
    ]

    for title, df in tables.items():
        table_code = df.to_latex(index=False, escape=True)
        latex_snippets.append(f"% {'='*60}\n% {title}\n% {'='*60}")
        latex_snippets.append(r"\begin{table*}[t]")
        latex_snippets.append(r"\centering")
        latex_snippets.append(f"\\caption{{{title}}}")
        latex_snippets.append(table_code)
        latex_snippets.append(r"\end{table*}")
        latex_snippets.append("\n")

    output_path.write_text("\n".join(latex_snippets), encoding="utf-8")


# =====================================================================
# Main Execution Pipeline
# =====================================================================


def run_pipeline(
    results_dir: Path,
    output_dir: Path,
    export_formats: List[str],
) -> Dict[str, pd.DataFrame]:
    """Execute complete empirical evaluation table pipeline."""
    print("=" * 90)
    print(f"[INFO] Ingesting Empirical Evaluation Cohort from: {results_dir.resolve()}")
    print("=" * 90)

    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory does not exist: {results_dir}")

    paired_attacks, benign_df = load_empirical_cohort(results_dir)

    n_models = paired_attacks["model"].nunique()
    n_paired_attacks = len(paired_attacks)
    n_benign = len(benign_df)

    print(f"[VALIDATION] Models detected: {n_models} / 11")
    print(f"[VALIDATION] Valid paired adversarial observations: {n_paired_attacks:,} (expected: ~35,199)")
    print(f"[VALIDATION] Total benign observations loaded: {n_benign:,} (expected: 4,400)")

    # Assert model completeness
    assert n_models == 11, f"Expected 11 models, found {n_models}."
    assert n_paired_attacks >= 35000, f"Expected >= 35,000 paired attack rows, found {n_paired_attacks}."
    assert n_benign >= 4000, f"Expected >= 4,000 benign rows, found {n_benign}."

    print("[SUCCESS] Dataset integrity and pairing validation passed.\n")

    # Generate Tables
    t_r1 = generate_table_r1(paired_attacks)
    t_r2 = generate_table_r2(paired_attacks)
    t_r3 = generate_table_r3(benign_df)
    t_r4 = generate_table_r4(paired_attacks)

    tables = {
        "Table R1: Model-Level Attack Evaluation & Statistical Significance": t_r1,
        "Table R2: Attack Category-Level Breakdown (16 Attack Categories)": t_r2,
        "Table R3: Benign Usability, Refusal & False-Flag Rates": t_r3,
        "Table R4: Efficiency, Latency Breakdown & Overhead Analysis": t_r4,
    }

    # Print to console
    for title, df in tables.items():
        print_table(title, df)

    # Save to disk
    output_dir.mkdir(parents=True, exist_ok=True)

    if "csv" in export_formats:
        t_r1.to_csv(output_dir / "table_r1_model_level.csv", index=False)
        t_r2.to_csv(output_dir / "table_r2_category_level.csv", index=False)
        t_r3.to_csv(output_dir / "table_r3_benign_usability.csv", index=False)
        t_r4.to_csv(output_dir / "table_r4_efficiency_overhead.csv", index=False)
        print(f"[SUCCESS] CSV tables exported to: {output_dir.resolve()}")

    if "md" in export_formats:
        export_markdown(tables, output_dir / "evaluation_results_tables.md")
        print(f"[SUCCESS] Markdown report saved to: {(output_dir / 'evaluation_results_tables.md').resolve()}")

    if "latex" in export_formats:
        export_combined_latex(tables, output_dir / "tables_latex.tex")
        print(f"[SUCCESS] LaTeX tables saved to: {(output_dir / 'tables_latex.tex').resolve()}")

    return tables


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute and export publication Tables R1-R4 from empirical evaluation cohort."
    )
    parser.add_argument(
        "--results_dir",
        "--data_dir",
        "-r",
        "-d",
        type=Path,
        default=RESULTS_DIR,
        help="Directory containing the model evaluation runs (default: output/model_results).",
    )
    parser.add_argument(
        "--output_dir",
        "-o",
        type=Path,
        default=Path("output/evaluation_tables"),
        help="Directory to save generated evaluation tables.",
    )
    parser.add_argument(
        "--format",
        "-f",
        nargs="+",
        choices=["csv", "md", "latex", "all"],
        default=["all"],
        help="Export formats to generate (default: all).",
    )

    args = parser.parse_args()
    formats = ["csv", "md", "latex"] if "all" in args.format else args.format

    run_pipeline(
        results_dir=args.results_dir,
        output_dir=args.output_dir,
        export_formats=formats,
    )


if __name__ == "__main__":
    main()
