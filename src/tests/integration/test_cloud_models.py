import os
import csv
import re
from datetime import UTC, datetime
from pathlib import Path

import pytest
import requests
# Load environment variables from .env file in project root (prefer python-dotenv)
def _load_dotenv_from_project_root():
    try:
        from dotenv import load_dotenv as _load

        _load()
        return
    except Exception:
        pass

    env_path = Path(__file__).resolve().parents[3] / ".env"
    if not env_path.exists():
        return
    try:
        text = env_path.read_text(encoding="utf-8")
    except Exception:
        return
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        key = k.strip()
        val = v.strip().strip('"').strip("'")
        os.environ.setdefault(key, val)


_load_dotenv_from_project_root()

CLOUD_API_PROVIDER = os.getenv("CLOUD_API_PROVIDER", "openai")
CLOUD_API_BASE = os.getenv("CLOUD_API_BASE", "https://api.openai.com")
CLOUD_API_KEY = os.getenv("CLOUD_API_KEY", os.getenv("OPENAI_API_KEY", ""))
DEFAULT_CLOUD_MODELS = [
    ("openai", os.getenv("OPENAI_MODEL", "gpt-5-mini")),
    ("google", os.getenv("GEMINI_MODEL", "gemini-flash-lite-latest")),
    ("mistral", os.getenv("MISTRAL_MODEL", "mistral-small-latest")),
    ("groq", os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")),
    ("cloudflare", os.getenv("CLOUDFLARE_MODEL", "@cf/qwen/qwen3-30b-a3b-fp8")),
    ("cloudflare", os.getenv("CLOUDFLARE_GROK_MODEL", "xai/grok-4.20-0309-non-reasoning")),
]
DEFAULT_PROVIDER_BASES = {
    "openai": "https://api.openai.com",
    "google": "https://generativelanguage.googleapis.com",
    "gemini": "https://generativelanguage.googleapis.com",
    "mistral": "https://api.mistral.ai",
    "groq": "https://api.groq.com/openai",
    "cloudflare": "https://api.cloudflare.com",
}
REQUEST_TIMEOUT_SECONDS = {}
PROJECT_ROOT = Path(__file__).resolve().parents[3]
REPORT_DIR = PROJECT_ROOT / "output" / "model_availability"
CSV_REPORT = REPORT_DIR / "cloud_model_responses.csv"
TXT_REPORT = REPORT_DIR / "cloud_model_responses.txt"
_REPORT_INITIALIZED = False


def _ensure_report_files():
    global _REPORT_INITIALIZED
    if _REPORT_INITIALIZED:
        return

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    with CSV_REPORT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "checked_at_utc",
            "model_name",
            "success",
            "prompt",
            "answer",
            "error",
            "duration_seconds",
        ])
        writer.writeheader()

    TXT_REPORT.write_text(
        "Cloud Model Availability Report\n"
        f"Generated from: src/tests/integration/test_cloud_models.py\n"
        f"Default provider: {CLOUD_API_PROVIDER}\n\n",
        encoding="utf-8",
    )
    _REPORT_INITIALIZED = True


def _write_model_result(result):
    _ensure_report_files()
    with CSV_REPORT.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(result.keys()))
        writer.writerow(result)

    answer = result["answer"].strip() or "<empty>"
    status = "SUCCESS" if result["success"] in {True, "true", "True"} else "FAILED"
    with TXT_REPORT.open("a", encoding="utf-8") as handle:
        handle.write(f"[{status}] Hi {result['model_name']}: {answer}\n")
        if result["error"]:
            handle.write(f"  error: {result['error']}\n")
        handle.write(f"  prompt: {result['prompt']}\n")
        handle.write(f"  duration_seconds: {result['duration_seconds']}\n\n")


def _cloud_models():
    if os.getenv("RUN_CLOUD_MODELS") not in {"1", "true", "True"}:
        return []

    raw = os.getenv("CLOUD_MODELS", "")
    if raw:
        return [m.strip() for m in raw.split(",") if m.strip()]

    return [f"{provider}:{model}" for provider, model in DEFAULT_CLOUD_MODELS if model]


def _split_provider_model(model_name: str):
    if ":" not in model_name:
        return CLOUD_API_PROVIDER.lower(), model_name
    provider, model = model_name.split(":", 1)
    return provider.strip().lower(), model.strip()


def _provider_api_base(provider: str):
    return os.getenv(f"{provider.upper()}_API_BASE") or DEFAULT_PROVIDER_BASES.get(provider, CLOUD_API_BASE)


def _provider_api_key(provider: str):
    provider = provider.lower()
    if provider == "openai":
        return os.getenv("OPENAI_API_KEY") or CLOUD_API_KEY
    if provider in {"google", "gemini"}:
        return os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if provider == "mistral":
        return os.getenv("MISTRAL_API_KEY")
    if provider == "groq":
        return os.getenv("GROQ_API_KEY")
    if provider == "cloudflare":
        return os.getenv("CLOUDFLARE_API_TOKEN") or os.getenv("CLOUDFLARE_API_KEY")
    return os.getenv(f"{provider.upper()}_API_KEY") or CLOUD_API_KEY


def _response_error(response):
    try:
        body = response.json()
    except Exception:
        return _sanitize_error(response.text)

    if not isinstance(body, dict):
        return _sanitize_error(str(body))

    error = body.get("error")
    if isinstance(error, dict):
        return _sanitize_error(error.get("message") or str(error))
    if error:
        return _sanitize_error(str(error))

    errors = body.get("errors")
    if isinstance(errors, list) and errors:
        first = errors[0]
        if isinstance(first, dict):
            return _sanitize_error(first.get("message") or str(first))
        return _sanitize_error(str(first))

    return _sanitize_error(str(body))


def _sanitize_error(message: str):
    return re.sub(r"([?&]key=)[^&\s)]+", r"\1<redacted>", str(message))


def _chat_completion_answer(body):
    choices = body.get("choices") or []
    if not choices:
        return ""
    first = choices[0]
    msg = first.get("message") or {}
    if isinstance(msg, dict):
        return (msg.get("content") or "").strip()
    return (first.get("text") or "").strip()


def _provider_send_request(provider: str, model: str, prompt: str):
    """Send a request to the configured provider and return (answer, error, raw_body).

    This function uses best-effort endpoints and response parsing for several providers.
    It will skip if the provider-specific API key is missing.
    """
    provider = provider.lower()
    api_base = _provider_api_base(provider)
    api_key = _provider_api_key(provider)

    if not api_key:
        return "", f"missing API key for {provider}", {}

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    timeout = REQUEST_TIMEOUT_SECONDS.get(provider, 90)

    try:
        if provider == "openai":
            url = api_base.rstrip("/") + "/v1/chat/completions"
            payload = {"model": model, "messages": [{"role": "user", "content": prompt}]}
            if model.startswith(("o1", "o3", "gpt-5")):
                payload["max_completion_tokens"] = 512
            else:
                payload.update({"temperature": 0.0, "max_tokens": 128})
            resp = requests.post(url, json=payload, headers=headers, timeout=timeout)
            if resp.status_code != 200:
                return "", f"{resp.status_code} {resp.reason}: {_response_error(resp)}", {}
            body = resp.json()
            return (_chat_completion_answer(body), "", body)

        if provider in {"google", "gemini"}:
            url = api_base.rstrip("/") + f"/v1beta/models/{model}:generateContent"
            payload = {
                "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                "generationConfig": {"temperature": 0.0, "maxOutputTokens": 128},
            }
            resp = requests.post(f"{url}?key={api_key}", json=payload, headers={"Content-Type": "application/json"}, timeout=timeout)
            if resp.status_code != 200:
                return "", f"{resp.status_code} {resp.reason}: {_response_error(resp)}", {}
            body = resp.json()
            parts = body.get("candidates", [{}])[0].get("content", {}).get("parts", [])
            text = "".join(part.get("text", "") for part in parts if isinstance(part, dict)).strip()
            return (text, "", body)

        if provider == "mistral":
            url = api_base.rstrip("/") + "/v1/chat/completions"
            payload = {"model": model, "messages": [{"role": "user", "content": prompt}], "temperature": 0.0, "max_tokens": 128}
            resp = requests.post(url, json=payload, headers=headers, timeout=timeout)
            if resp.status_code != 200:
                return "", f"{resp.status_code} {resp.reason}: {_response_error(resp)}", {}
            body = resp.json()
            return (_chat_completion_answer(body), "", body)

        if provider == "groq":
            url = api_base.rstrip("/") + "/v1/chat/completions"
            payload = {"model": model, "messages": [{"role": "user", "content": prompt}], "temperature": 0.0, "max_tokens": 128}
            resp = requests.post(url, json=payload, headers=headers, timeout=timeout)
            if resp.status_code != 200:
                return "", f"{resp.status_code} {resp.reason}: {_response_error(resp)}", {}
            body = resp.json()
            return (_chat_completion_answer(body), "", body)

        if provider in {"cloudflare", "qwen"}:
            account_id = os.getenv("CLOUDFLARE_ACCOUNT_ID")
            if not account_id:
                return "", "missing CLOUDFLARE_ACCOUNT_ID for cloudflare", {}
            url = api_base.rstrip("/") + f"/client/v4/accounts/{account_id}/ai/v1/chat/completions"
            payload = {"model": model, "messages": [{"role": "user", "content": prompt}], "temperature": 0.0, "max_tokens": 128}
            resp = requests.post(url, json=payload, headers=headers, timeout=timeout)
            if resp.status_code != 200:
                return "", f"{resp.status_code} {resp.reason}: {_response_error(resp)}", {}
            body = resp.json()
            return (_chat_completion_answer(body) or _chat_completion_answer(body.get("result", {})), "", body)

    except Exception as exc:
        return "", _sanitize_error(str(exc)), {}

    return "", "unsupported provider or no parseable response", {}


@pytest.mark.integration
@pytest.mark.cloud
@pytest.mark.slow
@pytest.mark.parametrize("model_name", _cloud_models() or [None])
def test_cloud_model_responds(model_name):
    """Generate a live cloud model availability report.

    Usage from project root (PowerShell):
      $env:RUN_CLOUD_MODELS=1; pytest -q src/tests/integration/test_cloud_models.py
      $env:STRICT_CLOUD_MODELS=1; pytest -q src/tests/integration/test_cloud_models.py
    """
    if model_name is None:
        pytest.skip("Set RUN_CLOUD_MODELS=1 and CLOUD_MODELS to test cloud models.")

    provider, provider_model_name = _split_provider_model(model_name)

    if not _provider_api_key(provider):
        pytest.skip(f"API key for {provider} is required to run this cloud model test.")

    prompt = f"Hi {provider_model_name}. Reply with exactly one word: OK"
    started = datetime.now(UTC)
    answer, error, body = _provider_send_request(provider, provider_model_name, prompt)

    success = not error and bool(answer.strip())

    result = {
        "checked_at_utc": started.isoformat(),
        "model_name": f"{provider}:{provider_model_name}",
        "success": str(success).lower(),
        "prompt": prompt,
        "answer": answer,
        "error": error or body.get("error", {}).get("message", "") if isinstance(body.get("error"), dict) else error or body.get("error", ""),
        "duration_seconds": f"{(datetime.now(UTC) - started).total_seconds():.2f}",
    }

    _write_model_result(result)

    if os.getenv("STRICT_CLOUD_MODELS") in {"1", "true", "True"}:
        assert success, f"{provider}:{provider_model_name} did not return a usable response; see {CSV_REPORT.relative_to(PROJECT_ROOT)}"
