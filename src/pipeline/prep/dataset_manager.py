"""Dataset manager for ArcShield benchmark data onboarding.

This script keeps a single registry of external and local dataset sources,
inspects local files, and reports schema gaps against benchmark requirements.

Usage examples:

# Human-readable summary to stdout
python src/pipeline/prep/dataset_manager.py summary

# Write markdown summary to an explicit path `output/dataset_inventory.md`
python src/pipeline/prep/dataset_manager.py summary --output output/dataset_inventory.md

# Or, provide only a filename and the script will namespace it under
# `output/<subdir>/` (default subdir is `dataset_manager`):
python src/pipeline/prep/dataset_manager.py summary --output dataset_inventory.md

# If you pass `--output output/<name>`, the script will also namespace the
# file into `output/<subdir>/<name>` (to avoid creating both
# `output/<name>` and `output/<subdir>/<name>`). To write exactly to
# `output/<name>` without namespacing, pass an absolute path or a path outside
# the repo output folder.

# Write full JSON summary (optionally probe Hugging Face schemas).
# Filename-only saves to `output/<subdir>/dataset_inventory.json` and is
# mirrored to `output/<subdir>/` by default (so both copies go to the same
# namespaced folder). Use `--also-save-to` to override the mirror target.
python src/pipeline/prep/dataset_manager.py json --output dataset_inventory.json --probe-remote

# For help
python src/pipeline/prep/dataset_manager.py -h

Notes:
- Local dataset sources live under the `dataset/` folder.
- `--output` accepts an explicit path (e.g. `output/...`) or a filename.
    - If you pass a filename only, the file will be written to
        `output/<subdir>/` (default `dataset_manager`).
- By default outputs are also mirrored to `dataset_info/`; change with
    `--also-save-to` if desired.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = PROJECT_ROOT / "dataset"


REQUIRED_CANONICAL_FIELDS = {
    "prompt": ["prompt", "text", "query", "input", "instruction", "original_query", "persuasive_prompt"],
    "is_attack": ["is_attack", "label", "intent", "attack", "harmful", "safety_label", "category"],
    "attack_vector": ["attack_vector", "technique", "attack_type", "jailbreak_type", "pattern"],
    "attack_mechanism": ["attack_mechanism", "mechanism", "method"],
    "linguistic_strategy": ["linguistic_strategy", "strategy", "language_strategy", "obfuscation_type"],
    "annotation_method": ["annotation_method", "labeling_method", "label_source"],
    "annotator_count": ["annotator_count", "num_annotators"],
    "inter_rater_reliability": ["inter_rater_reliability", "irr", "kappa", "fleiss_kappa"],
}


@dataclass(frozen=True)
class DatasetSource:
    name: str
    kind: str  # local | huggingface | kaggle
    path: Optional[str] = None
    hf_id: Optional[str] = None
    hf_config: Optional[str] = None
    kaggle_id: Optional[str] = None
    notes: Optional[str] = None


DATASET_REGISTRY: Sequence[DatasetSource] = [
    DatasetSource(
        name="in-the-wild-jailbreak-prompts",
        kind="huggingface",
        hf_id="TrustAIRLab/in-the-wild-jailbreak-prompts",
        hf_config="jailbreak_2023_12_25",
    ),
    DatasetSource(
        name="deepset prompt-injections",
        kind="huggingface",
        hf_id="deepset/prompt-injections",
    ),
    DatasetSource(
        name="JBB Behaviors",
        kind="huggingface",
        hf_id="JailbreakBench/JBB-Behaviors",
        hf_config="behaviors",
    ),
    DatasetSource(
        name="JBB judge_comparison",
        kind="huggingface",
        hf_id="JailbreakBench/JBB-Behaviors",
        hf_config="judge_comparison",
    ),
    DatasetSource(
        name="safe-guard prompt injection",
        kind="huggingface",
        hf_id="xTRam1/safe-guard-prompt-injection",
    ),
    DatasetSource(
        name="Aegis AI Content Safety 2.0",
        kind="huggingface",
        hf_id="nvidia/Aegis-AI-Content-Safety-Dataset-2.0",
        notes="Authentication required",
    ),
    DatasetSource(
        name="do-not-answer",
        kind="huggingface",
        hf_id="LibrAI/do-not-answer",
    ),
    DatasetSource(
        name="prompt-injection-malignant",
        kind="kaggle",
        kaggle_id="marycamilainfo/prompt-injection-malignant",
    ),
    DatasetSource(
        name="educational-llm-guardrails-bench",
        kind="local",
        path="dataset/educational-llm-guardrails-bench_Data",
    ),
    DatasetSource(
        name="Foot-in-the-door-Jailbreak",
        kind="local",
        path="dataset/Foot-in-the-door-Jailbreak_Data",
    ),
    DatasetSource(
        name="JailbreakLLMs",
        kind="local",
        path="dataset/JailbreakLLMs_Data",
    ),
    DatasetSource(
        name="LARGO",
        kind="local",
        path="dataset/LARGO_Data",
    ),
    DatasetSource(
        name="MAGIC",
        kind="local",
        path="dataset/MAGIC_Data",
    ),
    DatasetSource(
        name="MPA",
        kind="local",
        path="dataset/MPA_Data",
    ),
    DatasetSource(
        name="QueryAttack",
        kind="local",
        path="dataset/QueryAttack_Data",
    ),
    DatasetSource(
        name="StrongREJECT",
        kind="local",
        path="dataset/strongreject_Data",
    ),
    DatasetSource(
        name="TroGEN",
        kind="local",
        path="dataset/TroGEN_Data",
    ),
]


def _read_csv_headers(path: Path) -> List[str]:
    with path.open("r", encoding="utf-8", errors="ignore", newline="") as handle:
        reader = csv.reader(handle)
        row = next(reader, [])
    return [str(c).strip() for c in row if str(c).strip()]


def _read_jsonl_headers(path: Path) -> List[str]:
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        first = handle.readline().strip()
    if not first:
        return []
    try:
        payload = json.loads(first)
    except json.JSONDecodeError:
        return []
    if isinstance(payload, dict):
        return sorted(payload.keys())
    return []


def _read_json_headers(path: Path) -> List[str]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
    except json.JSONDecodeError:
        return []

    if isinstance(payload, list) and payload and isinstance(payload[0], dict):
        return sorted(payload[0].keys())
    if isinstance(payload, dict):
        if "records" in payload and isinstance(payload["records"], list) and payload["records"]:
            first = payload["records"][0]
            if isinstance(first, dict):
                return sorted(first.keys())
        return sorted(payload.keys())
    return []


def infer_headers(path: Path) -> List[str]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return _read_csv_headers(path)
    if suffix == ".jsonl":
        return _read_jsonl_headers(path)
    if suffix == ".json":
        return _read_json_headers(path)
    return []


def _canonical_field_coverage(columns: Set[str]) -> Dict[str, bool]:
    normalized = {c.strip().lower() for c in columns}
    coverage: Dict[str, bool] = {}
    for canonical, aliases in REQUIRED_CANONICAL_FIELDS.items():
        coverage[canonical] = any(alias.lower() in normalized for alias in aliases)
    return coverage


def scan_local_dataset(path: Path) -> Dict[str, object]:
    if not path.exists():
        return {
            "exists": False,
            "files": [],
            "all_columns": [],
            "coverage": {k: False for k in REQUIRED_CANONICAL_FIELDS},
        }

    files: List[Dict[str, object]] = []
    all_columns: Set[str] = set()
    for candidate in sorted(path.rglob("*")):
        if not candidate.is_file():
            continue
        if candidate.suffix.lower() not in {".csv", ".json", ".jsonl"}:
            continue

        headers = infer_headers(candidate)
        all_columns.update(headers)
        files.append(
            {
                "file": str(candidate.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                "columns": headers,
                "column_count": len(headers),
            }
        )

    return {
        "exists": True,
        "files": files,
        "all_columns": sorted(all_columns),
        "coverage": _canonical_field_coverage(all_columns),
    }


def probe_hf_dataset(source: DatasetSource) -> Dict[str, object]:
    try:
        from datasets import load_dataset  # type: ignore
    except Exception as exc:
        return {"ok": False, "reason": f"datasets library unavailable: {exc}"}

    try:
        if source.hf_config:
            ds = load_dataset(source.hf_id, source.hf_config)
        else:
            ds = load_dataset(source.hf_id)
    except Exception as exc:
        return {"ok": False, "reason": str(exc)}

    split_names = list(ds.keys())
    first_split = split_names[0] if split_names else None
    columns: List[str] = []
    if first_split:
        columns = list(ds[first_split].column_names)

    return {
        "ok": True,
        "splits": split_names,
        "first_split": first_split,
        "columns": columns,
        "coverage": _canonical_field_coverage(set(columns)),
    }


def build_summary(probe_remote: bool = False) -> Dict[str, object]:
    items: List[Dict[str, object]] = []
    for source in DATASET_REGISTRY:
        row = asdict(source)

        if source.kind == "local":
            assert source.path is not None
            local_path = PROJECT_ROOT / source.path
            row["scan"] = scan_local_dataset(local_path)

        if probe_remote and source.kind == "huggingface":
            row["probe"] = probe_hf_dataset(source)

        items.append(row)

    return {
        # Emit short, repo-relative identifiers instead of full absolute paths
        "project_root": PROJECT_ROOT.name,
        "data_root": str(DATA_ROOT.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "required_canonical_fields": REQUIRED_CANONICAL_FIELDS,
        "datasets": items,
    }


def render_markdown(summary: Dict[str, object]) -> str:
    lines: List[str] = []
    lines.append("# Dataset Inventory")
    lines.append("")
    lines.append("## Canonical Fields Needed By Benchmark")
    for field in REQUIRED_CANONICAL_FIELDS:
        lines.append(f"- {field}")

    lines.append("")
    lines.append("## Sources")

    for item in summary["datasets"]:  # type: ignore[index]
        source = dict(item)
        lines.append("")
        lines.append(f"### {source['name']}")
        lines.append(f"- kind: {source['kind']}")

        if source.get("hf_id"):
            config = source.get("hf_config")
            if config:
                lines.append(f"- huggingface: {source['hf_id']} ({config})")
            else:
                lines.append(f"- huggingface: {source['hf_id']}")

        if source.get("kaggle_id"):
            lines.append(f"- kaggle: {source['kaggle_id']}")

        if source.get("path"):
            lines.append(f"- local path: {source['path']}")

        if source.get("notes"):
            lines.append(f"- notes: {source['notes']}")

        scan = source.get("scan")
        if isinstance(scan, dict):
            lines.append(f"- local exists: {scan.get('exists')}")
            files = scan.get("files", [])
            lines.append(f"- data files detected: {len(files)}")
            coverage = scan.get("coverage", {})
            if isinstance(coverage, dict):
                covered = sorted([k for k, v in coverage.items() if v])
                missing = sorted([k for k, v in coverage.items() if not v])
                lines.append(f"- covered canonical fields: {', '.join(covered) if covered else 'none'}")
                lines.append(f"- missing canonical fields: {', '.join(missing) if missing else 'none'}")

        probe = source.get("probe")
        if isinstance(probe, dict):
            if probe.get("ok"):
                lines.append(f"- remote probe: ok ({probe.get('first_split')})")
                cols = probe.get("columns", [])
                lines.append(f"- remote columns: {', '.join(cols) if cols else 'none'}")
            else:
                lines.append(f"- remote probe: failed ({probe.get('reason')})")

    return "\n".join(lines).rstrip() + "\n"


def write_output(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Manage and inspect ArcShield datasets.")
    parser.add_argument("command", choices=["summary", "json"], help="Output mode")
    parser.add_argument("--output", help="Optional output file path")
    parser.add_argument(
        "--subdir",
        help="Optional output subdirectory under `output/` to namespace artifacts (default: dataset_manager)",
        default="dataset_manager",
    )
    parser.add_argument("--probe-remote", action="store_true", help="Attempt to probe Hugging Face schemas")
    parser.add_argument(
        "--also-save-to",
        help="Optional extra folder name under `output/` to copy outputs to in addition to the primary location. If omitted, the script mirrors into the same subdir as `--subdir`.",
        default=None,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = build_summary(probe_remote=args.probe_remote)

    if args.command == "json":
        rendered = json.dumps(summary, indent=2)
    else:
        rendered = render_markdown(summary)
    if args.output:
        out_path = Path(args.output)
        # If the user passed a filename-only value, namespace under output/<subdir>/
        if not out_path.parent or str(out_path.parent) == ".":
            target = PROJECT_ROOT / "output" / args.subdir / out_path.name
        else:
            # If user passed a path directly under `output/` (e.g. output/foo.md),
            # treat it as a request to namespace under the chosen subdir to avoid
            # creating both `output/foo.md` and `output/<subdir>/foo.md`.
            parts = out_path.parts
            if parts and parts[0] == "output" and len(parts) == 2:
                target = PROJECT_ROOT / "output" / args.subdir / out_path.name
            else:
                # Respect provided path (allow absolute or other relative paths)
                target = PROJECT_ROOT / args.output

        write_output(target, rendered)
        print(f"Wrote {str(target.relative_to(PROJECT_ROOT)).replace('\\', '/')}" )

        # Mirror into an additional, namespaced location under `output/`.
        try:
            mirror_folder = args.also_save_to if args.also_save_to else args.subdir
            secondary = PROJECT_ROOT / "output" / mirror_folder / target.name
            if secondary.resolve() != target.resolve():
                write_output(secondary, rendered)
                print(f"Also wrote {str(secondary.relative_to(PROJECT_ROOT)).replace('\\', '/')}" )
        except Exception:
            # Best-effort: do not fail the run if mirroring can't be done
            pass
    else:
        print(rendered)


if __name__ == "__main__":
    main()
