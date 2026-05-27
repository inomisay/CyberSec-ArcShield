import os
import csv
from datetime import UTC, datetime
from pathlib import Path

import pytest
import requests


OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
PROJECT_ROOT = Path(__file__).resolve().parents[3]
REPORT_DIR = PROJECT_ROOT / "output" / "model_availability"
CSV_REPORT = REPORT_DIR / "local_ollama_model_responses.csv"
TXT_REPORT = REPORT_DIR / "local_ollama_model_responses.txt"
_REPORT_INITIALIZED = False


def _ensure_report_files():
    global _REPORT_INITIALIZED
    if _REPORT_INITIALIZED:
        return

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    with CSV_REPORT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "checked_at_utc",
                "model_name",
                "success",
                "prompt",
                "answer",
                "thinking",
                "done_reason",
                "error",
                "duration_seconds",
            ],
        )
        writer.writeheader()

    TXT_REPORT.write_text(
        "Local Ollama Model Availability Report\n"
        f"Generated from: src/tests/integration/test_local_ollama_models.py\n"
        f"Base URL: {OLLAMA_BASE_URL}\n\n",
        encoding="utf-8",
    )
    _REPORT_INITIALIZED = True


def _write_model_result(result):
    _ensure_report_files()
    with CSV_REPORT.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(result.keys()))
        writer.writerow(result)

    answer = result["answer"].strip() or "<empty>"
    status = "SUCCESS" if result["success"] else "FAILED"
    with TXT_REPORT.open("a", encoding="utf-8") as handle:
        handle.write(f"[{status}] Hi {result['model_name']}: {answer}\n")
        if result["error"]:
            handle.write(f"  error: {result['error']}\n")
        handle.write(f"  prompt: {result['prompt']}\n")
        handle.write(f"  duration_seconds: {result['duration_seconds']}\n\n")


def _local_ollama_models():
    if os.getenv("RUN_LOCAL_OLLAMA") not in {"1", "true", "True"}:
        return []

    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        response.raise_for_status()
    except Exception:
        return []

    models = response.json().get("models", [])
    return [item["name"] for item in models if item.get("name")]


@pytest.mark.integration
@pytest.mark.ollama
@pytest.mark.slow
@pytest.mark.parametrize("model_name", _local_ollama_models() or [None])
def test_local_ollama_model_responds(model_name):
    """HIL-style local test: verify each installed Ollama model can answer.

    To run from the project root:
      $env:RUN_LOCAL_OLLAMA=1; pytest -q src/tests/integration/test_local_ollama_models.py

    If pytest cannot write `.pytest_cache`, disable the cache plugin:
      $env:RUN_LOCAL_OLLAMA=1; pytest -q -p no:cacheprovider src/tests/integration/test_local_ollama_models.py
    """
    if model_name is None:
        pytest.skip("Set RUN_LOCAL_OLLAMA=1 and start Ollama to test local models.")

    prompt = f"Hi {model_name}. /no_think Reply with exactly one word: OK"
    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.0,
            "num_predict": 128,
        },
    }

    started = datetime.now(UTC)
    body = {}
    error = ""
    try:
        response = requests.post(f"{OLLAMA_BASE_URL}/api/generate", json=payload, timeout=180)
        response.raise_for_status()
        body = response.json()
    except Exception as exc:
        error = str(exc)

    visible_response = body.get("response") or ""
    thinking_response = body.get("thinking") or ""
    success = (
        not error
        and body.get("model") in {model_name, model_name.split(":", 1)[0]}
        and not body.get("error")
        and body.get("done") is True
        and bool((visible_response + thinking_response).strip())
    )

    answer = visible_response.strip() or thinking_response.strip()
    result = {
        "checked_at_utc": started.isoformat(),
        "model_name": model_name,
        "success": str(success).lower(),
        "prompt": prompt,
        "answer": answer,
        "thinking": thinking_response,
        "done_reason": body.get("done_reason", ""),
        "error": error or body.get("error", ""),
        "duration_seconds": f"{(datetime.now(UTC) - started).total_seconds():.2f}",
    }
    _write_model_result(result)

    assert success, f"{model_name} did not return a usable response; see {CSV_REPORT.relative_to(PROJECT_ROOT)}"
