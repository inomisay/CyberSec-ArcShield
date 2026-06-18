"""Export model configuration information to CSV.

This script replaces the older JSON-based model detector. It reads the benchmark
configuration reference, probes local GPU and Ollama manifests, and writes a
flat CSV with the model metadata that is useful for reporting and comparisons.

Output columns include:
- model_variant
- source
- hosting_platform
- execution_engine
- quantization
- temperature
- top_p
- top_k
- max_tokens
- gpu
- endpoint
- source_file
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DIR = PROJECT_ROOT / "output"
MODEL_CONFIG_DIR = OUTPUT_DIR / "model_configurations"
LOGS_DIR = PROJECT_ROOT / "logs"
DEFAULT_SOURCE = MODEL_CONFIG_DIR / "model_configurations_current.txt"
LEGACY_SOURCES = [
    LOGS_DIR / "model_configurations_current.txt",
    OUTPUT_DIR / "model_configurations_current.txt",
]
DEFAULT_OUTPUT = MODEL_CONFIG_DIR / "model_configurations.csv"


def detect_gpu() -> dict:
    # Try nvidia-smi first.
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        lines = [line.strip() for line in out.splitlines() if line.strip()]
        gpus = []
        for line in lines:
            parts = [part.strip() for part in line.split(",")]
            if len(parts) >= 3:
                gpus.append({"name": parts[0], "memory_mb": parts[1], "driver": parts[2]})
        if gpus:
            return {"source": "nvidia-smi", "gpus": gpus}
    except Exception:
        pass

    # On Windows, try wmic.
    try:
        out = subprocess.check_output(
            ["wmic", "path", "win32_videocontroller", "get", "name,driverVersion,AdapterRAM", "/format:csv"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        lines = [line.strip() for line in out.splitlines() if line.strip() and not line.startswith("Node")]
        gpus = []
        for line in lines:
            parts = [part.strip() for part in line.split(",")]
            if len(parts) >= 4:
                name = parts[1]
                driver = parts[2]
                ram = parts[3]
                try:
                    memory_mb = int(ram) // (1024 * 1024)
                except Exception:
                    memory_mb = None
                gpus.append({"name": name, "memory_mb": memory_mb, "driver": driver})
        if gpus:
            return {"source": "wmic", "gpus": gpus}
    except Exception:
        pass

    # On Windows, try PowerShell CIM. Some shells block this, so failure is OK.
    try:
        out = subprocess.check_output(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                "Get-CimInstance Win32_VideoController | Select-Object Name,DriverVersion,AdapterRAM | ConvertTo-Json",
            ],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        payload = json.loads(out)
        devices = payload if isinstance(payload, list) else [payload]
        gpus = []
        for device in devices:
            if not isinstance(device, dict):
                continue
            name = device.get("Name")
            if not name:
                continue
            ram = device.get("AdapterRAM")
            try:
                memory_mb = int(ram) // (1024 * 1024)
            except Exception:
                memory_mb = None
            gpus.append({"name": name, "memory_mb": memory_mb, "driver": device.get("DriverVersion", "unknown")})
        if gpus:
            return {"source": "powershell-cim", "gpus": gpus}
    except Exception:
        pass

    return {"source": "unknown", "gpus": []}


def ollama_show_quantization(model_ref: str) -> str:
    try:
        out = subprocess.check_output(
            ["ollama", "show", model_ref],
            stderr=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except Exception:
        return "unknown"

    for line in out.splitlines():
        stripped = line.strip()
        if stripped.lower().startswith("quantization"):
            parts = stripped.split()
            if len(parts) >= 2:
                return parts[-1]
    return "unknown"


def read_ollama_manifests() -> Dict[str, Dict[str, str]]:
    home = Path.home()
    base = home / ".ollama" / "models" / "manifests" / "registry.ollama.ai" / "library"
    results: Dict[str, Dict[str, str]] = {}
    if not base.exists():
        return results

    for model_dir in base.iterdir():
        try:
            if not model_dir.is_dir():
                continue
            for tag_file in model_dir.iterdir():
                if not tag_file.is_file():
                    continue
                data = json.loads(tag_file.read_text(encoding="utf-8"))
                cfg = data.get("config", {})
                layers = data.get("layers", [])
                model_layers = [layer for layer in layers if layer.get("mediaType") == "application/vnd.ollama.image.model"]
                size_sum = sum(layer.get("size", 0) for layer in model_layers or layers)
                model_ref = f"{model_dir.name}:{tag_file.name}"
                quant = ollama_show_quantization(model_ref)
                info = {
                    "manifest_path": str(tag_file),
                    "config_digest": cfg.get("digest") if isinstance(cfg, dict) else None,
                    "layer_count": str(len(layers)),
                    "layer_size_bytes": str(size_sum),
                    "size_gb": f"{size_sum / 1_000_000_000:.1f} GB" if size_sum else "unknown",
                    "quantization_hint": quant,
                }
                results[model_ref] = info
                if tag_file.name == "latest":
                    results[model_dir.name] = info
        except Exception:
            continue

    return results


def summarize_gpu(payload: dict) -> str:
    gpus = payload.get("gpus") or []
    if not gpus:
        return "unknown"
    parts = []
    for gpu in gpus:
        name = gpu.get("name", "unknown")
        driver = gpu.get("driver", "unknown")
        memory_mb = gpu.get("memory_mb", "unknown")
        parts.append(f"{name} (driver={driver}, memory_mb={memory_mb})")
    return f"{payload.get('source', 'unknown')}: " + "; ".join(parts)


def gpu_from_source_text(source_path: Path) -> dict:
    text = source_path.read_text(encoding="utf-8", errors="ignore")
    match = re.search(
        r"- Local GPU \(detected via (?P<source>[^)]+)\):\s*"
        r"- Name: (?P<name>[^\r\n]+)\s*"
        r"- DriverVersion: (?P<driver>[^\r\n]+)\s*"
        r"- Memory \(approx\): (?P<memory>[^\r\n]+)",
        text,
        flags=re.MULTILINE,
    )
    if not match:
        return {"source": "unknown", "gpus": []}

    memory_text = match.group("memory")
    memory_match = re.search(r"(\d+)", memory_text)
    return {
        "source": match.group("source"),
        "gpus": [
            {
                "name": match.group("name").strip(),
                "driver": match.group("driver").strip(),
                "memory_mb": memory_match.group(1) if memory_match else memory_text.strip(),
            }
        ],
    }


def infer_hosting_platform(section: str, model_variant: str, endpoint: str) -> str:
    endpoint_lower = endpoint.lower()
    if section == "local" or "localhost:11434" in endpoint_lower:
        return "Ollama"
    if model_variant.lower().startswith("google ai studio "):
        return "Google AI Studio"
    if model_variant.lower().startswith("openai "):
        return "OpenAI"
    if model_variant.lower().startswith("mistral "):
        return "Mistral"
    if model_variant.lower().startswith("groq "):
        return "Groq"
    if model_variant.lower().startswith("cloudflare workers ai "):
        return "Cloudflare Workers AI"
    if "generativelanguage.googleapis.com" in endpoint_lower:
        return "Google AI Studio"
    if "api.openai.com" in endpoint_lower:
        return "OpenAI"
    if "api.mistral.ai" in endpoint_lower:
        return "Mistral"
    if "api.groq.com" in endpoint_lower:
        return "Groq"
    if "api.cloudflare.com" in endpoint_lower:
        return "Cloudflare Workers AI"
    return "unknown"


def infer_execution_engine(endpoint: str) -> str:
    endpoint_lower = endpoint.lower()
    if "localhost:11434" in endpoint_lower:
        return "Ollama API"
    if "generativelanguage.googleapis.com" in endpoint_lower:
        return "Gemini generateContent"
    if "api.openai.com" in endpoint_lower:
        return "OpenAI chat completions"
    if "api.mistral.ai" in endpoint_lower:
        return "Mistral chat completions"
    if "api.groq.com" in endpoint_lower:
        return "Groq OpenAI-compatible"
    if "api.cloudflare.com" in endpoint_lower:
        return "Cloudflare Workers AI"
    if endpoint:
        return "HTTP API"
    return "unknown"


def manifest_quantization(model_variant: str, manifests: Dict[str, Dict[str, str]]) -> str:
    normalized = model_variant.split("@", 1)[0].strip()
    normalized = normalized.removesuffix("-instruct")
    for key, info in manifests.items():
        if normalized == key or normalized.startswith(key) or key.startswith(normalized):
            return info.get("quantization_hint", "unknown")
        if "@sha256:" in model_variant and model_variant.split("@", 1)[1] == info.get("config_digest"):
            return info.get("quantization_hint", "unknown")
    return "unknown"


def normalize_ollama_model_variant(model_variant: str, manifests: Dict[str, Dict[str, str]]) -> str:
    if "@sha256:" not in model_variant:
        return model_variant

    digest = model_variant.split("@", 1)[1]
    for key, info in manifests.items():
        if key.endswith(":latest") and info.get("config_digest") == digest:
            return key
    for key, info in manifests.items():
        if ":" in key and info.get("config_digest") == digest:
            return key
    return model_variant


def parse_model_rows(source_path: Path, gpu_payload: dict, manifests: Dict[str, Dict[str, str]]) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    current: Optional[Dict[str, str]] = None
    section: Optional[str] = None
    known_attribute_keys = {"endpoint", "temperature", "top_p", "top_k", "max_tokens", "max_completion_tokens"}
    if not gpu_payload.get("gpus"):
        gpu_payload = gpu_from_source_text(source_path)

    for raw_line in source_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.strip()
        if not line:
            continue

        if line == "Local Ollama models:":
            if current:
                rows.append(current)
                current = None
            section = "local"
            continue

        if line == "Cloud model configurations:":
            if current:
                rows.append(current)
                current = None
            section = "cloud"
            continue

        if line.startswith("Reporting changes now implemented in code:"):
            if current:
                rows.append(current)
                current = None
            section = None
            continue

        if not line.startswith("- "):
            continue

        content = line[2:].strip()
        key, separator, value = content.partition(":")
        if current and separator and key.strip().lower() in known_attribute_keys:
            normalized_key = key.strip().lower()
            normalized_value = value.strip()
            if normalized_key == "max_completion_tokens":
                normalized_key = "max_tokens"
            current[normalized_key] = normalized_value
            if normalized_key == "endpoint":
                current["execution_engine"] = infer_execution_engine(normalized_value)
                current["hosting_platform"] = infer_hosting_platform(section or "", current["model_variant"], normalized_value)
            continue

        if current:
            rows.append(current)

        if section == "local":
            model_variant = normalize_ollama_model_variant(content, manifests)
            endpoint = "http://localhost:11434/api/generate"
            source = "Local"
            hosting_platform = "Ollama"
            execution_engine = "Ollama API"
            quantization = manifest_quantization(model_variant, manifests)
        elif section == "cloud":
            model_variant = content
            endpoint = ""
            source = "Cloud"
            hosting_platform = infer_hosting_platform(section, model_variant, endpoint)
            execution_engine = infer_execution_engine(endpoint)
            quantization = "not supported"
            lower_content = content.lower()
            if lower_content.startswith("google ai studio "):
                hosting_platform = "Google AI Studio"
                model_variant = content[len("Google AI Studio "):].strip()
            elif lower_content.startswith("openai "):
                hosting_platform = "OpenAI"
                model_variant = content[len("OpenAI "):].strip()
            elif lower_content.startswith("mistral "):
                hosting_platform = "Mistral"
                model_variant = content[len("Mistral "):].strip()
            elif lower_content.startswith("groq "):
                hosting_platform = "Groq"
                model_variant = content[len("Groq "):].strip()
            elif lower_content.startswith("cloudflare workers ai "):
                hosting_platform = "Cloudflare Workers AI"
                model_variant = content[len("Cloudflare Workers AI "):].strip()
        else:
            continue

        current = {
            "model_variant": model_variant,
            "source": source,
            "hosting_platform": hosting_platform,
            "execution_engine": execution_engine,
            "quantization": quantization,
            "temperature": "",
            "top_p": "",
            "top_k": "",
            "max_tokens": "",
            "gpu": summarize_gpu(gpu_payload),
            "endpoint": endpoint,
            "source_file": project_relative(source_path),
        }

    if current:
        rows.append(current)

    # Fill any missing values with the global defaults from the source text.
    defaults = {
        "temperature": "0.0",
        "top_p": "1.0",
        "top_k": "not set",
        "max_tokens": "unknown",
    }
    for row in rows:
        for key, value in defaults.items():
            if not row.get(key):
                row[key] = value

    return rows


def write_csv(output_path: Path, rows: List[Dict[str, str]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "model_variant",
        "source",
        "hosting_platform",
        "execution_engine",
        "quantization",
        "temperature",
        "top_p",
        "top_k",
        "max_tokens",
        "gpu",
        "endpoint",
        "source_file",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def project_relative(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export model configuration information to CSV.")
    parser.add_argument("--source", default=str(DEFAULT_SOURCE), help="Path to model_configurations_current.txt")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Output CSV path")
    args = parser.parse_args()

    source_path = Path(args.source)
    if not source_path.exists():
        if source_path == DEFAULT_SOURCE:
            source_path = next((legacy for legacy in LEGACY_SOURCES if legacy.exists()), source_path)
        if not source_path.exists():
            raise SystemExit(f"Model configuration source file not found: {source_path}")

    gpu_payload = detect_gpu()
    manifests = read_ollama_manifests()
    rows = parse_model_rows(source_path, gpu_payload, manifests)

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path
    write_csv(output_path, rows)
    print(f"Wrote model inventory CSV to: {project_relative(output_path)}")
    print(f"Rows exported: {len(rows)}")
    print(f"Generated at: {datetime.now(UTC).isoformat()}")


if __name__ == "__main__":
    main()
