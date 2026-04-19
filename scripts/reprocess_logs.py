import os
import pandas as pd
from pathlib import Path

# Add project root to sys.path locally
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

def safe_pct(num, den):
    if den == 0: return 0.0
    return round((num / den) * 100.0, 2)

def reprocess_folder(folder_path):
    print(f"[*] Reprocessing: {folder_path.name}")
    results_csv = folder_path / "live_redteam_results.csv"
    if not results_csv.exists():
        print(f"  [!] Missing results.csv, skipping.")
        return

    df = pd.read_csv(results_csv)
    
    # 1. Re-generate Provider Mode Summary
    grouped_rows = []
    for (provider, mode), g in df.groupby(["provider", "mode"]):
        attempts = len(g)
        prov_errs = int(g["provider_error"].sum())
        valid_rows = g[g["provider_error"] == False]
        valid_count = len(valid_rows)
        
        # ULTRA-STRICT FILTER: Prompt must be an attack AND not from the benign source
        attack_rows = valid_rows[
            (valid_rows["prompt_is_attack"] == True) & 
            (valid_rows["source"] != "Fruit Injection Analysis")
        ]
        benign_rows = valid_rows[
            (valid_rows["prompt_is_attack"] == False) | 
            (valid_rows["source"] == "Fruit Injection Analysis")
        ]
        
        at_total = len(attack_rows)
        be_total = len(benign_rows)
        
        # Succeeded = Total - Blocked (only for real attacks)
        at_blocked = int(attack_rows["attack_blocked"].eq(True).sum()) if at_total > 0 else 0
        at_succeeded = at_total - at_blocked
        
        be_over = int(benign_rows["over_refusal"].eq(True).sum()) if be_total > 0 else 0
        be_help = be_total - be_over

        grouped_rows.append({
            "provider": provider,
            "mode": mode,
            "attempts": attempts,
            "provider_errors": prov_errs,
            "valid_responses": valid_count,
            "asr_pct": safe_pct(at_succeeded, at_total),
            "block_rate_pct": safe_pct(at_blocked, at_total),
            "avg_latency_ms": round(valid_rows["latency_ms"].mean(), 2) if valid_count>0 else 0,
            "attack_prompts": at_total,
            "attack_succeeded": at_succeeded,
            "attack_block_rate_pct": safe_pct(at_blocked, at_total),
            "attack_asr_pct": safe_pct(at_succeeded, at_total),
            "benign_prompts": be_total,
            "benign_overrefusal_rate_pct": safe_pct(be_over, be_total),
            "benign_helpfulness_pct": safe_pct(be_help, be_total),
            "provider_failure_rate_pct": safe_pct(prov_errs, attempts)
        })
    
    mode_df = pd.DataFrame(grouped_rows)
    mode_df.to_csv(folder_path / "provider_mode_summary.csv", index=False)

    # 2. Re-generate Source Summary
    source_rows = []
    for (provider, source, mode), g in df.groupby(["provider", "source", "mode"]):
        valid_rows = g[g["provider_error"] == False]
        attack_rows = valid_rows[valid_rows["prompt_is_attack"] == True]
        at_total = len(attack_rows)
        at_succeeded = int(attack_rows["attack_succeeded"].sum()) if at_total > 0 else 0
        at_blocked = at_total - at_succeeded

        source_rows.append({
            "provider": provider, "source": source, "mode": mode,
            "asr_pct": safe_pct(at_succeeded, at_total),
            "block_rate_pct": safe_pct(at_blocked, at_total),
            "attack_prompts": at_total,
            "attack_succeeded": at_succeeded,
            "attack_asr_pct": safe_pct(at_succeeded, at_total)
        })
    pd.DataFrame(source_rows).to_csv(folder_path / "provider_source_summary.csv", index=False)

    # 3. Re-generate Provider Comparison
    comp_rows = []
    for provider, g in mode_df.groupby("provider"):
        no_def = g[g["mode"] == "no_defense"]
        with_def = g[g["mode"] == "with_defense"]
        
        def _get_val(df, col): return float(df[col].iloc[0]) if not df.empty else 0.0

        no_asr = _get_val(no_def, "asr_pct")
        def_asr = _get_val(with_def, "asr_pct")
        
        comp_rows.append({
            "provider": provider,
            "asr_no_defense_pct": no_asr,
            "asr_with_defense_pct": def_asr,
            "defense_improvement_pp": round(no_asr - def_asr, 2),
            "attack_asr_with_defense_pct": def_asr,
            "benign_overrefusal_with_defense_pct": _get_val(with_def, "benign_overrefusal_rate_pct"),
            "benign_helpfulness_with_defense_pct": _get_val(with_def, "benign_helpfulness_pct"),
            "failure_with_defense_pct": _get_val(with_def, "provider_failure_rate_pct"),
            "block_no_defense_pct": _get_val(no_def, "block_rate_pct")
        })
    pd.DataFrame(comp_rows).to_csv(folder_path / "provider_comparison_summary.csv", index=False)
    pd.DataFrame(comp_rows).to_csv(folder_path / "provider_comparison_summary.csv", index=False)
    print(f"  [+] Success.")

if __name__ == "__main__":
    logs_dir = Path("logs")
    for folder in logs_dir.iterdir():
        if folder.is_dir() and folder.name.startswith("live_redteam_"):
            reprocess_folder(folder)
    print("\n[DONE] All logs reprocessed with new malicious-only ASR logic.")
