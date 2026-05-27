"""Export ArcShield curated dataset distribution and provenance metadata."""

from __future__ import annotations

import csv
import json
import argparse
from collections import Counter, defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = PROJECT_ROOT / "dataset" / "curated" / "attack_dataset_curated.csv"
DEFAULT_SUMMARY = PROJECT_ROOT / "dataset" / "curated" / "attack_dataset_curated_summary.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output" / "analysis" / "dataset_metadata"


SOURCE_LICENSE_NOTES = {
    "HF:nvidia/Aegis-AI-Content-Safety-Dataset-2.0": "See Hugging Face dataset card for license and usage terms.",
    "HF:xTRam1/safe-guard-prompt-injection": "See Hugging Face dataset card for license and usage terms.",
    "HF:JailbreakBench/JBB-Behaviors/judge_comparison": "See Hugging Face dataset card / JailbreakBench terms.",
    "HF:TrustAIRLab/in-the-wild-jailbreak-prompts/jailbreak_2023_12_25": "See Hugging Face dataset card for license and usage terms.",
    "HF:LibrAI/do-not-answer": "See Hugging Face dataset card for license and usage terms.",
    "HF:deepset/prompt-injections": "See Hugging Face dataset card for license and usage terms.",
    "strongreject": "See upstream strongreject dataset documentation.",
    "JailbreakLLMs": "See upstream JailbreakLLMs dataset documentation.",
    "QueryAttack": "See upstream QueryAttack dataset documentation.",
    "Foot-in-the-door-Jailbreak": "See upstream dataset documentation.",
    "TroGEN": "See upstream TroGEN dataset documentation.",
    "MAGIC": "See upstream MAGIC dataset documentation.",
    "educational-llm-guardrails-bench": "See upstream dataset documentation.",
    "MPA": "See upstream MPA dataset documentation.",
}


def _read_rows(dataset_path: str | Path = DEFAULT_DATASET) -> list[dict]:
    path = Path(dataset_path)
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["is_synthetic"] = str(row.get("is_synthetic", "")).lower() in {"true", "1", "yes"}
    return rows


def build_dataset_metadata(dataset_path: str | Path = DEFAULT_DATASET, summary_path: str | Path = DEFAULT_SUMMARY) -> dict:
    rows = _read_rows(dataset_path)
    summary = {}
    if Path(summary_path).exists():
        summary = json.loads(Path(summary_path).read_text(encoding="utf-8"))

    attack_counts = Counter(row["attack_type"] for row in rows)
    source_counts = Counter(row["source_dataset"] for row in rows)
    synthetic_counts = Counter(row["is_synthetic"] for row in rows)
    real_synthetic_by_attack = defaultdict(lambda: {"real": 0, "synthetic": 0})
    source_by_attack = defaultdict(Counter)
    augmentation_by_attack = defaultdict(Counter)

    for row in rows:
        attack_type = row["attack_type"]
        origin = "synthetic" if row["is_synthetic"] else "real"
        real_synthetic_by_attack[attack_type][origin] += 1
        source_by_attack[attack_type][row["source_dataset"]] += 1
        if row["is_synthetic"]:
            augmentation_by_attack[attack_type][row.get("augmentation_method") or "unknown"] += 1

    total = len(rows)
    return {
        "dataset_path": str(dataset_path),
        "summary_path": str(summary_path),
        "total_rows": total,
        "attack_type_count": len(attack_counts),
        "balanced_target_per_attack_type": summary.get("per_type_target", 200),
        "class_balancing_strategy": (
            "Deduplicate source prompts, classify into 16 attack types, then sample "
            "up to the per-type target. Sparse supported classes are topped off with "
            "template-based synthetic variants while preserving provenance fields."
        ),
        "synthetic_augmentation_strategy": (
            "Synthetic rows wrap real source prompts with attack-type-specific templates. "
            "Synthetic provenance is tracked by is_synthetic, augmentation_method, and augmentation_variant."
        ),
        "real_rows": synthetic_counts[False],
        "synthetic_rows": synthetic_counts[True],
        "real_ratio": synthetic_counts[False] / total if total else 0.0,
        "synthetic_ratio": synthetic_counts[True] / total if total else 0.0,
        "attack_type_counts": dict(sorted(attack_counts.items())),
        "source_counts": dict(sorted(source_counts.items())),
        "real_synthetic_by_attack_type": dict(sorted(real_synthetic_by_attack.items())),
        "source_by_attack_type": {key: dict(value) for key, value in sorted(source_by_attack.items())},
        "augmentation_by_attack_type": {key: dict(value) for key, value in sorted(augmentation_by_attack.items())},
        "source_license_notes": {source: SOURCE_LICENSE_NOTES.get(source, "Review upstream source license before redistribution.") for source in sorted(source_counts)},
    }


def export_dataset_metadata(output_dir: str | Path = DEFAULT_OUTPUT_DIR, dataset_path: str | Path = DEFAULT_DATASET) -> dict[str, str]:
    metadata = build_dataset_metadata(dataset_path)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    json_path = output / "dataset_metadata.json"
    attack_counts_path = output / "per_attack_type_counts.csv"
    source_counts_path = output / "source_metadata.csv"
    real_synthetic_path = output / "real_vs_synthetic_by_attack_type.csv"
    markdown_path = output / "dataset_metadata_tables.md"

    json_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    _write_two_column_csv(attack_counts_path, "attack_type", "rows", metadata["attack_type_counts"])
    _write_two_column_csv(source_counts_path, "source_dataset", "rows", metadata["source_counts"], license_notes=metadata["source_license_notes"])

    with real_synthetic_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["attack_type", "real", "synthetic", "total", "synthetic_ratio"])
        writer.writeheader()
        for attack_type, counts in metadata["real_synthetic_by_attack_type"].items():
            total = counts["real"] + counts["synthetic"]
            writer.writerow({
                "attack_type": attack_type,
                "real": counts["real"],
                "synthetic": counts["synthetic"],
                "total": total,
                "synthetic_ratio": counts["synthetic"] / total if total else 0.0,
            })

    markdown_path.write_text(_metadata_markdown(metadata), encoding="utf-8")
    return {
        "json": str(json_path),
        "per_attack_type_counts": str(attack_counts_path),
        "source_metadata": str(source_counts_path),
        "real_vs_synthetic": str(real_synthetic_path),
        "tables": str(markdown_path),
    }


def stratified_sample(rows: list[dict], sample_size: int, strata_fields: tuple[str, ...] = ("attack_type",), seed: int = 42) -> list[dict]:
    import random

    rng = random.Random(seed)
    buckets = defaultdict(list)
    for row in rows:
        buckets[tuple(row.get(field) for field in strata_fields)].append(row)
    keys = list(buckets)
    rng.shuffle(keys)
    selected = []
    while len(selected) < sample_size and keys:
        progressed = False
        for key in list(keys):
            bucket = buckets[key]
            if not bucket:
                keys.remove(key)
                continue
            selected.append(rng.choice(bucket))
            progressed = True
            if len(selected) >= sample_size:
                break
        if not progressed:
            break
    return selected[:sample_size]


def _write_two_column_csv(path: Path, key_name: str, value_name: str, values: dict, license_notes: dict | None = None) -> None:
    fields = [key_name, value_name] + (["license_note"] if license_notes else [])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for key, value in values.items():
            row = {key_name: key, value_name: value}
            if license_notes:
                row["license_note"] = license_notes.get(key, "")
            writer.writerow(row)


def _metadata_markdown(metadata: dict) -> str:
    lines = [
        "# ArcShield Dataset Metadata",
        "",
        f"- Total rows: {metadata['total_rows']}",
        f"- Attack types: {metadata['attack_type_count']}",
        f"- Balanced target per attack type: {metadata['balanced_target_per_attack_type']}",
        f"- Real rows: {metadata['real_rows']}",
        f"- Synthetic rows: {metadata['synthetic_rows']}",
        f"- Synthetic ratio: {metadata['synthetic_ratio']:.4f}",
        "",
        "## Class Balancing Strategy",
        "",
        metadata["class_balancing_strategy"],
        "",
        "## Synthetic Augmentation Strategy",
        "",
        metadata["synthetic_augmentation_strategy"],
        "",
        "## Per-Attack-Type Counts",
        "",
        "| Attack Type | Rows |",
        "| --- | ---: |",
    ]
    for attack_type, count in metadata["attack_type_counts"].items():
        lines.append(f"| {attack_type} | {count} |")
    lines.extend(["", "## Source Licensing Notes", "", "| Source | Rows | License Note |", "| --- | ---: | --- |"])
    for source, count in metadata["source_counts"].items():
        lines.append(f"| {source} | {count} | {metadata['source_license_notes'].get(source, '')} |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export ArcShield curated dataset metadata.")
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()
    paths = export_dataset_metadata(args.output_dir, args.dataset)
    for key, path in paths.items():
        print(f"{key}: {path}")


if __name__ == "__main__":
    main()
