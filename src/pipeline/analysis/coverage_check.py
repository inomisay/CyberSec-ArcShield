"""Merged coverage scanner for ArcShield.

This script now replaces the old coverage runner, matrix exporter, and Hugging Face probe.
It scans local dataset files under `dataset/`, optionally probes referenced Hugging Face
sources, writes `coverage_out.json`, `hf_coverage_out.json` when available, and exports a
CSV matrix of local file coverage.

Run from the repository root:
    python src/pipeline/analysis/coverage_check.py
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import re
from pathlib import Path
from typing import Dict, Iterable, List, MutableMapping, Optional, Sequence, Set

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = PROJECT_ROOT / "dataset"
PREP_REGISTRY = PROJECT_ROOT / "src" / "pipeline" / "prep" / "dataset_manager.py"
OUT_DIR = Path(__file__).resolve().parent
COVERAGE_JSON = OUT_DIR / "coverage_out.json"
HF_COVERAGE_JSON = OUT_DIR / "hf_coverage_out.json"
COVERAGE_MATRIX_CSV = OUT_DIR / "coverage_matrix.csv"

coverage = [
    "Direct Exploitation",
    "Jailbreaks",
    "Prompt Injection",
    "Indirect Injection",
    "Role Playing",
    "Attention Shifting",
    "Logic Trap Attacks",
    "Hypothetical Manipulation",
    "Accidental Context Leakage",
    "Code Injection",
    "Dictionary Attack",
    "Fill-in-the-Blank Attack",
    "Multi-Language Attack",
    "Multi-Prompt Attack",
    "Obfuscation (Token Smuggling)",
    "Payload Splitting",
]

kw = {
    "Direct Exploitation": ["exploit", "ransomware", "malware", "hack", "hacking", "phishing", "ddos", "trojan", "virus", "botnet", "password cracking", "keylogger", "crypto mining", "steal", "attack", "bomb", "poison", "weapon"],
    "Jailbreaks": ["jailbreak", "bypass", "circumvent", "override", "disable moderator", "escape", "jailbreakbench", "jailbreak"],
    "Prompt Injection": ["prompt injection", "injection", "inject prompt", "instruct the model", "prompting to ignore", "ignore policy", "follow instructions despite", "prompt-injection"],
    "Indirect Injection": ["social engineering", "scam", "manipulate", "trick", "persuade", "indirect", "coerce", "bribe", "phishing", "scam"],
    "Role Playing": ["roleplay", "role play", "as an", "pretend", "pretending", "role-playing"],
    "Attention Shifting": ["distract", "attention", "shift attention", "misdirection", "divert"],
    "Logic Trap Attacks": ["logic trap", "paradox", "contradiction", "logic", "trap", "riddle", "loop"],
    "Hypothetical Manipulation": ["hypothetical", "what if", "imagine", "suppose", "hypothetical"],
    "Accidental Context Leakage": ["leak", "leakage", "expose", "accidental", "paste", "context leak", "paste private", "sensitive data"],
    "Code Injection": ["code", "script", "execute", "run code", "eval", "system command", "bash", "python script", "shell", "execute code", "compile", "sql injection", "query injection"],
    "Dictionary Attack": ["dictionary attack", "dictionary", "wordlist", "brute force", "bruteforce", "password list"],
    "Fill-in-the-Blank Attack": ["fill in the blank", "fill-in-the-blank", "complete the prompt", "blank", "cloze"],
    "Multi-Language Attack": ["translate", "spanish", "french", "german", "multi-language", "multi language", "language", "translated.csv"],
    "Multi-Prompt Attack": ["multi-prompt", "chain of prompts", "multiple prompts", "chain", "stepwise", "sequential prompts"],
    "Obfuscation (Token Smuggling)": ["obfuscate", "obfuscation", "token smuggling", "token", "encode", "encoded", "base64", "hex", "steganography"],
    "Payload Splitting": ["split", "payload", "split prompt", "split into parts", "partial instruction", "chunk"],
}


def empty_results() -> Dict[str, Dict[str, Set[str]]]:
    return {category: {"matched_files": set(), "examples": set()} for category in coverage}


def record_match(results: MutableMapping[str, MutableMapping[str, Set[str]]], category: str, file_id: str, snippet: str) -> None:
    results[category]["matched_files"].add(file_id)
    results[category]["examples"].add(snippet)


def scan_text_for_keywords(results: MutableMapping[str, MutableMapping[str, Set[str]]], file_id: str, text: str) -> None:
    lowered = text.lower()
    for category, keywords in kw.items():
        for keyword in keywords:
            if keyword in lowered:
                index = lowered.find(keyword)
                snippet = lowered[max(0, index - 60): index + len(keyword) + 60].replace("\n", " ")[:200]
                record_match(results, category, file_id, snippet)
                break


def read_csv_as_text(path: Path) -> str:
    df = pd.read_csv(path, dtype=str, keep_default_na=False, na_values=[])
    return "\n".join(df.astype(str).fillna("").apply(lambda row: " ".join(row.values), axis=1).tolist())


def scan_local_datasets() -> Dict[str, Dict[str, Set[str]]]:
    results = empty_results()
    excluded_old_data = os.path.normpath(str(DATA_ROOT / "oldData"))

    for dirpath, _, filenames in os.walk(DATA_ROOT):
        if excluded_old_data in os.path.normpath(dirpath):
            continue
        for filename in filenames:
            if not filename.lower().endswith((".csv", ".json", ".txt")):
                continue
            file_path = Path(dirpath) / filename
            try:
                file_id = file_path.relative_to(PROJECT_ROOT).as_posix()
            except Exception:
                file_id = file_path.as_posix()
            try:
                if file_path.suffix.lower() == ".csv":
                    text = read_csv_as_text(file_path)
                else:
                    text = file_path.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                try:
                    text = file_path.read_text(encoding="utf-8", errors="ignore")
                except Exception:
                    text = ""
            scan_text_for_keywords(results, file_id, text)
    return results


def discover_hf_ids() -> Dict[str, Optional[str]]:
    ids: Dict[str, Optional[str]] = {}

    if PREP_REGISTRY.exists():
        text = PREP_REGISTRY.read_text(encoding="utf-8", errors="ignore")
        for match in re.finditer(r"hf_id\s*=\s*\"([^\"]+)\"", text):
            ids[match.group(1)] = None
        for match in re.finditer(r"hf_id\s*=\s*\'([^\']+)\'", text):
            ids[match.group(1)] = None

    datasets_txt = DATA_ROOT / "Datasets.txt"
    if datasets_txt.exists():
        text = datasets_txt.read_text(encoding="utf-8", errors="ignore")
        for match in re.finditer(r"load_dataset\(\s*[\'\"]([^\'\"]+)[\'\"]\s*,\s*[\'\"]([^\'\"]+)[\'\"]", text):
            ids[match.group(1)] = match.group(2)
        for match in re.finditer(r"load_dataset\(\s*[\'\"]([^\'\"]+)[\'\"]\s*\)", text):
            ids.setdefault(match.group(1), None)
        for match in re.finditer(r"huggingface.co/datasets/([A-Za-z0-9_.\-]+/[A-Za-z0-9_.\-]+)", text):
            ids.setdefault(match.group(1), None)

    return ids


def scan_hf_source(hf_id: str, hf_config: Optional[str] = None) -> Optional[Dict[str, Dict[str, Set[str]]]]:
    try:
        from datasets import load_dataset
    except Exception as exc:
        print(f"datasets package not available: {exc}")
        print("Install with: pip install datasets")
        return None

    tag = f"huggingface:{hf_id}" + (f":{hf_config}" if hf_config else "")
    try:
        if hf_config:
            dataset = load_dataset(hf_id, hf_config, split="train[:200]")
        else:
            dataset = load_dataset(hf_id, split="train[:200]")
    except Exception:
        try:
            if hf_config:
                dataset = load_dataset(hf_id, hf_config)
            else:
                dataset = load_dataset(hf_id)
            rows = []
            for index, example in enumerate(dataset):
                if index > 199:
                    break
                if isinstance(example, dict):
                    rows.append(" ".join(str(value) for value in example.values() if value is not None))
                else:
                    rows.append(str(example))
            text = "\n".join(rows)
        except Exception as exc:
            print(f"failed to load {tag}: {exc}")
            return None
    else:
        rows = []
        for example in dataset:
            if isinstance(example, dict):
                rows.append(" ".join(str(value) for value in example.values() if value is not None))
            else:
                rows.append(str(example))
        text = "\n".join(rows)

    results = empty_results()
    scan_text_for_keywords(results, tag, text)
    return results


def scan_hf_datasets() -> Dict[str, Dict[str, Set[str]]]:
    results = empty_results()
    discovered = discover_hf_ids()
    if not discovered:
        return results

    print(f"found {len(discovered)} Hugging Face dataset references")
    for hf_id, hf_config in discovered.items():
        print(f"probing {hf_id} {hf_config or ''}".strip())
        scanned = scan_hf_source(hf_id, hf_config)
        if not scanned:
            continue
        merge_results(results, scanned)
    return results


def merge_results(target: MutableMapping[str, MutableMapping[str, Set[str]]], source: MutableMapping[str, MutableMapping[str, Set[str]]]) -> None:
    for category in coverage:
        target[category]["matched_files"].update(source[category]["matched_files"])
        target[category]["examples"].update(source[category]["examples"])


def finalize_results(raw_results: MutableMapping[str, MutableMapping[str, Set[str]]]) -> Dict[str, Dict[str, List[str] | int]]:
    return {
        category: {
            "count_files": len(details["matched_files"]),
            "files": sorted(details["matched_files"]),
            "examples": list(details["examples"])[:3],
        }
        for category, details in raw_results.items()
    }


def write_json(path: Path, payload: Dict[str, Dict[str, List[str] | int]]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_matrix(path: Path, payload: Dict[str, Dict[str, List[str] | int]]) -> None:
    file_map: Dict[str, Set[str]] = {}
    for category, details in payload.items():
        for file_path in details.get("files", []):
            if os.path.exists(file_path):
                file_map.setdefault(file_path, set()).add(category)

    categories = list(payload.keys())
    rows = sorted(file_map.keys())
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["file"] + categories)
        for file_path in rows:
            present = file_map.get(file_path, set())
            writer.writerow([file_path] + ["1" if category in present else "0" for category in categories])


def main() -> None:
    parser = argparse.ArgumentParser(description="Scan local and Hugging Face dataset sources for attack coverage.")
    parser.add_argument("--skip-hf", action="store_true", help="Only scan local dataset files")
    args = parser.parse_args()

    local_raw = scan_local_datasets()
    combined_raw = empty_results()
    merge_results(combined_raw, local_raw)

    hf_raw = empty_results()
    if not args.skip_hf:
        hf_raw = scan_hf_datasets()
        merge_results(combined_raw, hf_raw)

    local_payload = finalize_results(local_raw)
    combined_payload = finalize_results(combined_raw)
    hf_payload = finalize_results(hf_raw)

    write_json(COVERAGE_JSON, combined_payload)
    write_matrix(COVERAGE_MATRIX_CSV, combined_payload)
    if not args.skip_hf:
        write_json(HF_COVERAGE_JSON, hf_payload)

    print(f"wrote {COVERAGE_JSON.as_posix()}")
    print(f"wrote {COVERAGE_MATRIX_CSV.as_posix()}")
    if not args.skip_hf:
        print(f"wrote {HF_COVERAGE_JSON.as_posix()}")
    print(f"local files matched: {sum(item['count_files'] for item in local_payload.values())}")


if __name__ == "__main__":
    main()
