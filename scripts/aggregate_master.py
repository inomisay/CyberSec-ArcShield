import pandas as pd
from pathlib import Path

def aggregate_all_results():
    base_path = Path(r"d:\🎓📚\Dokuz Eylül Üniversitesi\8. Dönem\Network Security\Project\PromptEngInCyberSec")
    logs_dir = base_path / "logs"
    all_summaries = []
    
    # Identify all run folders
    for folder in logs_dir.iterdir():
        if not folder.is_dir() or not folder.name.startswith("live_redteam_"):
            continue
            
        summary_path = folder / "provider_comparison_summary.csv"
        results_path = folder / "live_redteam_results.csv"
        
        if summary_path.exists() and results_path.exists():
            df = pd.read_csv(summary_path)
            raw = pd.read_csv(results_path)
            # Verify attack count
            at_count = len(raw[(raw["prompt_is_attack"] == True) & (raw["source"] != "Fruit Injection Analysis")])
            provider_name = df["provider"].iloc[0]
            print(f"[*] Found {provider_name}: {at_count} true attack prompts analyzed.")
            
            df["run_folder"] = folder.name
            df["attack_sample_size"] = at_count
            all_summaries.append(df)
            
    if not all_summaries:
        print("No summary files found in any logs subfolders!")
        return

    # Combine all
    master_df = pd.concat(all_summaries, ignore_index=True)
    
    # Sort by run folder name (contains date) to get newest first
    master_df = master_df.sort_values("run_folder", ascending=False)
    
    # Drop duplicates for the same provider (keep most recent run)
    master_df = master_df.drop_duplicates(subset=["provider"], keep="first")
    
    # Save to the root logs folder for diagrams.py to find
    output_path = logs_dir / "provider_comparison_summary.csv"
    master_df.to_csv(output_path, index=False)
    
    # We also need the source-level summary for the Heatmap
    all_source_summaries = []
    for folder in logs_dir.iterdir():
        if not folder.is_dir() or not folder.name.startswith("live_redteam_"):
            continue
        source_path = folder / "provider_mode_summary.csv"
        if source_path.exists():
            all_source_summaries.append(pd.read_csv(source_path))
            
    if all_source_summaries:
        master_source_df = pd.concat(all_source_summaries, ignore_index=True)
        # Handle cases where older runs might be missing the 'source' or 'mode' columns
        if "source" not in master_source_df.columns:
            master_source_df["source"] = "General"
        if "mode" not in master_source_df.columns:
            master_source_df["mode"] = "with_defense"
            
        # Handle duplicates if same provider/source combo exists
        group_cols = [c for c in ["provider", "source", "mode"] if c in master_source_df.columns]
        master_source_df = master_source_df.groupby(group_cols).mean(numeric_only=True).reset_index()
        master_source_df.to_csv(logs_dir / "provider_mode_summary.csv", index=False)

    print(f"SUCCESS: Master summaries created with {len(master_df)} distinct models.")
    print(f"Location: {output_path}")

if __name__ == "__main__":
    aggregate_all_results()
