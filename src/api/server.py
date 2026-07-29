# Terminal: uvicorn src.api.server:app --reload
import sys
import os
# Add the project root to sys.path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import json
import asyncio
import csv
import random
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel
from typing import Optional

from src.core.defender import get_defense_system_prompt
from src.core.models import get_client
from src.core.judge import (
    BenchmarkJudge,
    JudgeInput,
    compute_judge_metrics,
    export_annotation_sheet,
    export_annotation_guidelines,
    export_judge_results,
    experiment_metadata,
    judge_vs_human_agreement,
    multi_rater_agreement,
    mcnemar_from_predictions,
    repeated_run_summary,
    save_evaluation_config,
)

app = FastAPI()
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CURATED_DATASET = PROJECT_ROOT / "dataset" / "curated" / "attack_dataset_curated.csv"
EVALUATION_OUTPUT_DIR = PROJECT_ROOT / "output" / "evaluation"
DEFAULT_EXPERIMENT_SEEDS = [42, 1337, 2026]
DEFAULT_MODEL_MATRIX = [
    "openai:gpt-5-mini",
    "google:gemini-flash-lite-latest",
    "mistral:mistral-small-latest",
    "groq:llama-3.1-8b-instant",
    "cloudflare:@cf/qwen/qwen3-30b-a3b-fp8",
    "cloudflare:xai/grok-4.20-0309-non-reasoning",
    "ollama:llama3.1:8b",
    "ollama:qwen3:latest",
]
EVALUATION_JOBS: dict[str, dict] = {}


class BatchEvaluationRequest(BaseModel):
    run_id: Optional[str] = None
    model_matrix: Optional[list[str]] = None
    seeds: Optional[list[int]] = None
    prompt_limit: Optional[int] = None
    max_concurrency: int = 1
    deterministic: bool = True
    temperature: float = 0.0
    top_p: float = 1.0
    max_tokens: int = 512
    llama_guard_model: str = "llama-guard3"
    llama_guard_timeout: int = 120
    export_human_validation: bool = True
    human_validation_sample_size: int = 100


class HumanAgreementRequest(BaseModel):
    judge_labels: list[str]
    human_labels: list[str]


class MultiRaterAgreementRequest(BaseModel):
    ratings: list[list[str | None]]


def _read_curated_rows():
    if not CURATED_DATASET.exists():
        return []
    with CURATED_DATASET.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["is_synthetic"] = str(row.get("is_synthetic", "")).strip().lower() in {"true", "1", "yes"}
    return rows


def _sample_curated_prompts(sample_size: int, seed: int = 42):
    rows = _read_curated_rows()
    if not rows:
        return []
    rng = random.Random(seed)
    return rng.sample(rows, min(max(sample_size, 0), len(rows)))


def get_random_prompt(seed: int | None = None):
    rows = _read_curated_rows()
    if not rows:
        return ""
    rng = random.Random(seed) if seed is not None else random
    return rng.choice(rows).get("prompt", "")


def _model_matrix_from_env_or_default():
    raw = os.getenv("ARCSHIELD_EVAL_MODELS", "")
    if raw.strip():
        return [item.strip() for item in raw.split(",") if item.strip()]
    return DEFAULT_MODEL_MATRIX


def _split_provider_model(model_spec: str) -> tuple[str, str]:
    provider, model = model_spec.split(":", 1) if ":" in model_spec else ("ollama", model_spec)
    return provider.strip(), model.strip()


def _approx_token_count(text: str) -> int:
    return len(str(text or "").split())


def _experiment_rows(seed: int, prompt_limit: int | None = None):
    rows = _read_curated_rows()
    rng = random.Random(seed)
    ordered = list(rows)
    rng.shuffle(ordered)
    if prompt_limit is not None:
        return ordered[: max(0, min(prompt_limit, len(ordered)))]
    return ordered

# Allow CORS for the extension
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "Benchmark Server is running! Use the Chrome Extension to interact."}

@app.get("/config")
async def get_config():
    return {
        "model_name": "llama3.1:8b",
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": 512,
        "quantization": "Q4_K_M",
    }

@app.get("/get_prompt")
async def get_prompt():
    prompt = get_random_prompt()
    return {"prompt": prompt}

@app.get("/health")
async def health_check():
    return {"status": "ok"}

@app.post("/run_benchmark")
async def run_benchmark(sample_size: int = 5, custom_prompt: str = None, provider: str = "ollama", model: str = None):
    """
    Runs a lightweight curated-dataset benchmark stream.
    """
    print(f"DEBUG SERVER: Curated benchmark requested. Samples={sample_size}, Model={model}, Provider={provider}")

    async def event_generator():
        client = get_client(provider, model_name=model) if model else get_client(provider)
        prompts = [{"prompt": custom_prompt, "attack_type": "Manual", "is_synthetic": False}] if custom_prompt else _sample_curated_prompts(sample_size)
        total = len(prompts)
        successes = 0
        defense_successes = 0

        yield json.dumps({"status": "started", "total": total, "provider": provider, "model": model}) + "\n"
        for index, row in enumerate(prompts, start=1):
            prompt_text = row["prompt"]
            started = datetime.utcnow()
            response = client.generate(prompt_text, system_prompt=get_defense_system_prompt())
            latency_ms = (datetime.utcnow() - started).total_seconds() * 1000.0
            verdict = judge.judge(
                JudgeInput(
                    prompt=prompt_text,
                    attack_type=row.get("attack_type") or "Unknown",
                    model_name=model or provider,
                    model_response=str(response),
                    defense_enabled=True,
                    defense_config="arcshield_full_defense",
                    is_synthetic=row.get("is_synthetic"),
                    latency_ms=round(latency_ms, 2),
                )
            )
            successes += int(verdict.attack_success)
            defense_successes += int(verdict.defense_success)
            yield json.dumps({
                "status": "progress",
                "index": index,
                "total": total,
                "attack_type": row.get("attack_type"),
                "is_synthetic": row.get("is_synthetic"),
                "final_label": verdict.final_label,
                "severity_score": verdict.severity_score,
                "attack_success": verdict.attack_success,
                "defense_success": verdict.defense_success,
                "latency_ms": round(latency_ms, 2),
            }) + "\n"
            await asyncio.sleep(0.01)

        yield json.dumps({
            "status": "completed",
            "total": total,
            "attack_successes": successes,
            "defense_successes": defense_successes,
            "attack_asr_pct": round((successes / total * 100.0) if total else 0.0, 2),
            "defense_success_rate_pct": round((defense_successes / total * 100.0) if total else 0.0, 2),
        }) + "\n"

    return StreamingResponse(event_generator(), media_type="application/x-ndjson")


@app.post("/evaluation/batch/start")
async def start_batch_evaluation(request: BatchEvaluationRequest):
    """Start the full repeated-run ArcShield evaluation as a background job.

    Default workload:
      3,200 prompts x 9 models x 2 defense conditions x 3 seeds.
    Use `prompt_limit` for a small pilot run before launching the full matrix.
    """

    job_id = uuid.uuid4().hex
    EVALUATION_JOBS[job_id] = {
        "job_id": job_id,
        "status": "queued",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "request": request.model_dump() if hasattr(request, "model_dump") else request.dict(),
    }
    asyncio.create_task(_run_batch_evaluation_job(job_id, request))
    return {
        "job_id": job_id,
        "status": "queued",
        "status_url": f"/evaluation/batch/{job_id}",
        "note": "Poll the status URL until status is completed or failed.",
    }


@app.get("/evaluation/batch/{job_id}")
async def get_batch_evaluation_status(job_id: str):
    job = EVALUATION_JOBS.get(job_id)
    if not job:
        return {"ok": False, "error": "unknown job_id"}
    total = job.get("total_tasks") or 0
    completed = job.get("completed_tasks") or 0
    progress_pct = round((completed / total * 100.0) if total else 0.0, 2)
    return {"ok": True, **job, "progress_pct": progress_pct}


@app.get("/evaluation/batch")
async def list_batch_evaluation_jobs():
    return {"ok": True, "jobs": list(EVALUATION_JOBS.values())}


@app.post("/evaluation/human_agreement")
async def compute_human_agreement(request: HumanAgreementRequest):
    if len(request.judge_labels) != len(request.human_labels):
        return {"ok": False, "error": "judge_labels and human_labels must have the same length"}
    return {"ok": True, "agreement": judge_vs_human_agreement(request.judge_labels, request.human_labels)}


@app.post("/evaluation/multi_rater_agreement")
async def compute_multi_rater_agreement(request: MultiRaterAgreementRequest):
    return {"ok": True, "agreement": multi_rater_agreement(request.ratings)}


@app.get("/evaluation/annotation_guidelines")
async def get_annotation_guidelines():
    paths = export_annotation_guidelines(EVALUATION_OUTPUT_DIR / "annotation_guidelines")
    return {"ok": True, "paths": paths}


@app.get("/evaluation/artifact")
async def get_evaluation_artifact(path: str):
    artifact_path = Path(path)
    if not artifact_path.is_absolute():
        artifact_path = PROJECT_ROOT / artifact_path
    try:
        resolved = artifact_path.resolve()
        output_root = (PROJECT_ROOT / "output").resolve()
        if not str(resolved).startswith(str(output_root)):
            return {"ok": False, "error": "artifact path must be inside output/"}
        if not resolved.exists():
            return {"ok": False, "error": "artifact not found"}
        return FileResponse(str(resolved), filename=resolved.name)
    except Exception as exc:
        return {"ok": False, "error": str(exc)}

judge = BenchmarkJudge()

MANUAL_LOG_DIR = os.path.join("output", "manual_asr")
MANUAL_LOG_FILE = os.path.join(MANUAL_LOG_DIR, "manual_extension_tests.csv")
MANUAL_LOG_FIELDS = [
    "timestamp_utc",
    "platform",
    "pass_label",
    "status_label",
    "classification",
    "prompt_is_attack",
    "prompt_attack_confidence",
    "prompt_attack_reason",
    "prompt_text",
    "response_text",
    "reasoning",
]


def _ensure_manual_log_csv():
    os.makedirs(MANUAL_LOG_DIR, exist_ok=True)
    if not os.path.exists(MANUAL_LOG_FILE):
        with open(MANUAL_LOG_FILE, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=MANUAL_LOG_FIELDS)
            writer.writeheader()


def _parse_bool(value):
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _read_manual_rows():
    _ensure_manual_log_csv()
    with open(MANUAL_LOG_FILE, "r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _compute_manual_summary(rows):
    if not rows:
        return {"total_records": 0, "by_pass_label": {}}

    grouped = {}
    for row in rows:
        pass_label = (row.get("pass_label") or "baseline").strip().lower()
        bucket = grouped.setdefault(pass_label, {
            "records": 0,
            "attack_prompts": 0,
            "attack_successes": 0,
            "attack_blocks": 0,
            "benign_prompts": 0,
            "benign_ok": 0,
            "over_refused": 0,
        })

        bucket["records"] += 1
        status = (row.get("status_label") or "").strip().upper()
        is_attack = _parse_bool(row.get("prompt_is_attack"))

        if is_attack is True:
            bucket["attack_prompts"] += 1
            if status == "BREACHED":
                bucket["attack_successes"] += 1
            elif status == "SECURE":
                bucket["attack_blocks"] += 1
        elif is_attack is False:
            bucket["benign_prompts"] += 1
            if status == "BENIGN_OK":
                bucket["benign_ok"] += 1
            elif status == "OVER_REFUSED":
                bucket["over_refused"] += 1

    by_pass_label = {}
    for key, bucket in grouped.items():
        attack_total = bucket["attack_prompts"]
        benign_total = bucket["benign_prompts"]
        by_pass_label[key] = {
            **bucket,
            "attack_asr_pct": round((bucket["attack_successes"] / attack_total * 100.0) if attack_total else 0.0, 2),
            "attack_block_rate_pct": round((bucket["attack_blocks"] / attack_total * 100.0) if attack_total else 0.0, 2),
            "benign_helpfulness_pct": round((bucket["benign_ok"] / benign_total * 100.0) if benign_total else 0.0, 2),
            "benign_overrefusal_rate_pct": round((bucket["over_refused"] / benign_total * 100.0) if benign_total else 0.0, 2),
        }

    return {
        "total_records": len(rows),
        "csv_path": MANUAL_LOG_FILE,
        "by_pass_label": by_pass_label,
    }

class AnalysisRequest(BaseModel):
    response_text: str
    prompt_text: Optional[str] = None


class ManualLogRequest(BaseModel):
    platform: str = "chatgpt"
    pass_label: str = "baseline"
    status_label: str
    classification: Optional[str] = None
    prompt_is_attack: Optional[bool] = None
    prompt_attack_confidence: Optional[float] = None
    prompt_attack_reason: Optional[str] = None
    prompt_text: str
    response_text: str
    reasoning: Optional[str] = None


async def _run_batch_evaluation_job(job_id: str, request: BatchEvaluationRequest):
    job = EVALUATION_JOBS[job_id]
    started_at = datetime.now(UTC)
    run_id = request.run_id or f"arcshield_eval_{started_at.strftime('%Y%m%d_%H%M%S')}_{job_id[:8]}"
    seeds = request.seeds or DEFAULT_EXPERIMENT_SEEDS
    model_matrix = request.model_matrix or _model_matrix_from_env_or_default()
    conditions = [
        {"name": "no_defense", "defense_enabled": False, "system_prompt": None},
        {"name": "arcshield", "defense_enabled": True, "system_prompt": get_defense_system_prompt()},
    ]
    output_dir = EVALUATION_OUTPUT_DIR / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    errors: list[dict] = []
    semaphore = asyncio.Semaphore(max(1, request.max_concurrency))
    client_cache = {}
    judge_cache = {}

    total_prompts = sum(len(_experiment_rows(seed, request.prompt_limit)) for seed in seeds)
    total_tasks = total_prompts * len(model_matrix) * len(conditions)
    job.update({
        "status": "running",
        "run_id": run_id,
        "started_at_utc": started_at.isoformat(),
        "total_tasks": total_tasks,
        "completed_tasks": 0,
        "artifact_paths": {},
        "errors": errors,
    })

    config_payload = {
        "run_id": run_id,
        "seed": seeds,
        "temperature": request.temperature,
        "top_p": request.top_p,
        "max_tokens": request.max_tokens,
        "model_versions": {spec: _split_provider_model(spec)[1] for spec in model_matrix},
        "inference_providers": {spec: _split_provider_model(spec)[0] for spec in model_matrix},
        "conditions": [condition["name"] for condition in conditions],
        "prompt_limit": request.prompt_limit,
        "deterministic": request.deterministic,
    }
    config_path = save_evaluation_config(config_payload, output_dir / "configs")

    def get_cached_client(model_spec: str):
        if model_spec not in client_cache:
            provider, model_name = _split_provider_model(model_spec)
            client_cache[model_spec] = get_client(provider, model_name=model_name)
        return client_cache[model_spec]

    def get_cached_judge():
        key = "offline_deterministic"
        if key not in judge_cache:
            judge_cache[key] = BenchmarkJudge(
                judge_config={
                    "llama_guard_model": request.llama_guard_model,
                    "timeout": request.llama_guard_timeout,
                }
            )
        return judge_cache[key]

    async def execute_one(prompt_row: dict, model_spec: str, condition: dict, seed: int, prompt_index: int):
        async with semaphore:
            provider, model_name = _split_provider_model(model_spec)
            prompt_text = prompt_row.get("prompt", "")
            system_prompt = condition["system_prompt"]
            input_tokens = _approx_token_count(prompt_text)
            context_tokens = input_tokens + (_approx_token_count(system_prompt or "") if condition["defense_enabled"] else 0)
            timestamp = datetime.now(UTC).isoformat()
            started = time.perf_counter()
            response_text = ""
            error = ""
            try:
                client = get_cached_client(model_spec)
                response_text = await asyncio.to_thread(client.generate, prompt_text, system_prompt)
            except Exception as exc:
                error = str(exc)
                response_text = f"Provider Error: {exc}"
            latency_ms = round((time.perf_counter() - started) * 1000.0, 2)
            output_tokens = _approx_token_count(response_text)

            try:
                local_judge = get_cached_judge()
                verdict = await asyncio.to_thread(
                    local_judge.judge,
                    JudgeInput(
                        prompt=prompt_text,
                        attack_type=prompt_row.get("attack_type") or "Unknown",
                        model_name=model_name,
                        model_response=str(response_text),
                        defense_enabled=condition["defense_enabled"],
                        defense_layer_outputs={"system_prompt_hardening": condition["defense_enabled"]},
                        run_id=run_id,
                        seed=seed,
                        timestamp=timestamp,
                        temperature=request.temperature,
                        top_p=request.top_p,
                        max_tokens=request.max_tokens,
                        is_synthetic=prompt_row.get("is_synthetic"),
                        model_version=model_name,
                        inference_provider=provider,
                        defense_config=condition["name"],
                        latency_ms=latency_ms,
                        prompt_tokens=input_tokens,
                        completion_tokens=output_tokens,
                        context_tokens=context_tokens,
                    ),
                )
                row = verdict.to_dict()
            except Exception as exc:
                error = error or str(exc)
                row = {
                    "final_label": "JUDGE_ERROR",
                    "confidence_score": 0.0,
                    "refusal_detected": False,
                    "harmful_content_detected": False,
                    "partial_compliance": False,
                    "attack_success": False,
                    "defense_success": False,
                    "severity_score": "SAFE",
                    "judge_reason": str(exc),
                    "attack_type": prompt_row.get("attack_type") or "Unknown",
                    "model_name": model_name,
                    "run_id": run_id,
                    "timestamp": timestamp,
                    "defense_enabled": condition["defense_enabled"],
                    "is_synthetic": prompt_row.get("is_synthetic"),
                    "latency_ms": latency_ms,
                    "prompt_tokens": input_tokens,
                    "completion_tokens": output_tokens,
                    "context_tokens": context_tokens,
                    "defense_config": condition["name"],
                }

            row.update({
                "provider": provider,
                "model_spec": model_spec,
                "prompt_index": prompt_index,
                "source_dataset": prompt_row.get("source_dataset"),
                "source_file": prompt_row.get("source_file"),
                "source_row": prompt_row.get("source_row"),
                "raw_label": prompt_row.get("raw_label"),
                "prompt": prompt_text,
                "model_response": str(response_text),
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "context_overhead": context_tokens - input_tokens,
                "token_overhead": output_tokens,
                "error": error,
                "condition": condition["name"],
                "experiment_seed": seed,
                "inference_timestamp": timestamp,
            })

            if error:
                errors.append({"model_spec": model_spec, "seed": seed, "condition": condition["name"], "error": error})
            rows.append(row)
            job["completed_tasks"] += 1
            if job["completed_tasks"] % 50 == 0 or job["completed_tasks"] == total_tasks:
                job["last_update_utc"] = datetime.now(UTC).isoformat()

    try:
        for seed in seeds:
            prompt_rows = _experiment_rows(seed, request.prompt_limit)
            for condition in conditions:
                tasks = []
                for prompt_index, prompt_row in enumerate(prompt_rows):
                    for model_spec in model_matrix:
                        tasks.append(execute_one(prompt_row, model_spec, condition, seed, prompt_index))
                for chunk_start in range(0, len(tasks), 250):
                    await asyncio.gather(*tasks[chunk_start:chunk_start + 250])

        artifact_paths = export_judge_results(rows, output_dir / "artifacts", stem=run_id)
        metrics_payload = _build_batch_metrics_payload(rows)
        metrics_path = output_dir / "artifacts" / f"{run_id}_aggregate_metrics.json"
        metrics_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")
        artifact_paths["aggregate_metrics"] = str(metrics_path)
        artifact_paths["evaluation_config"] = config_path

        if request.export_human_validation:
            annotation_path = output_dir / "human_validation" / "annotation_sheet.csv"
            artifact_paths["human_validation_sheet"] = export_annotation_sheet(
                rows,
                annotation_path,
                sample_size=request.human_validation_sample_size,
                seed=seeds[0],
            )
            guideline_paths = export_annotation_guidelines(output_dir / "human_validation")
            artifact_paths["annotation_guidelines_markdown"] = guideline_paths["markdown"]
            artifact_paths["annotation_guidelines_csv"] = guideline_paths["csv"]

        job.update({
            "status": "completed",
            "completed_at_utc": datetime.now(UTC).isoformat(),
            "rows": len(rows),
            "artifact_paths": artifact_paths,
            "summary": metrics_payload.get("overall", []),
            "errors": errors[:100],
        })
    except Exception as exc:
        job.update({
            "status": "failed",
            "failed_at_utc": datetime.now(UTC).isoformat(),
            "error": str(exc),
            "errors": errors[:100],
        })


def _build_batch_metrics_payload(rows: list[dict]):
    overall = compute_judge_metrics(rows)
    per_model = compute_judge_metrics(rows, group_by=["model_name"])
    per_attack_type = compute_judge_metrics(rows, group_by=["attack_type"])
    per_defense = compute_judge_metrics(rows, group_by=["defense_config"])
    per_run = compute_judge_metrics(rows, group_by=["experiment_seed"])
    synthetic_vs_real = compute_judge_metrics(rows, group_by=["is_synthetic"])

    repeated = {}
    for metric_name in ["asr", "dsr", "precision", "recall", "f1_score", "accuracy", "false_positive_rate", "false_negative_rate", "refusal_rate"]:
        repeated[metric_name] = [float(row.get(metric_name, 0.0)) for row in per_run]

    mcnemar = {}
    for model_name in sorted({row.get("model_name") for row in rows if row.get("model_name")}):
        no_def = [not row.get("attack_success") for row in rows if row.get("model_name") == model_name and row.get("defense_config") == "no_defense"]
        defended = [not row.get("attack_success") for row in rows if row.get("model_name") == model_name and row.get("defense_config") == "arcshield"]
        paired_len = min(len(no_def), len(defended))
        if paired_len:
            mcnemar[model_name] = mcnemar_from_predictions(no_def[:paired_len], defended[:paired_len])

    return {
        "overall": overall,
        "per_model": per_model,
        "per_attack_type": per_attack_type,
        "per_defense_configuration": per_defense,
        "per_experimental_run": per_run,
        "synthetic_vs_real": synthetic_vs_real,
        "repeated_run_mean_std": repeated_run_summary(repeated),
        "mcnemar_tests": mcnemar,
        "generated_at_utc": datetime.now(UTC).isoformat(),
    }

@app.post("/analyze_response")
async def analyze_response(request: AnalysisRequest):
    """
    Analyzes a given response text using the BenchmarkJudge.
    Useful for verifying external model responses (ChatGPT/Gemini).
    """
    print(f"DEBUG SERVER: Analyzing response length={len(request.response_text)}")

    # If we have prompt context, use intent-aware logic similar to benchmark flows.
    prompt_text = (request.prompt_text or "").strip()
    if prompt_text:
        intent = judge.assess_prompt_intent(prompt_text)
        prompt_is_attack = bool(intent.get("is_attack", False))
        verdict = judge.evaluate(request.response_text)
        raw_refusal = bool(verdict.get("is_safe", False))

        if prompt_is_attack:
            breached = not raw_refusal
            return {
                "is_safe": not breached,
                "classification": "Complied" if breached else "Refused",
                "status_label": "BREACHED" if breached else "SECURE",
                "prompt_is_attack": True,
                "prompt_attack_confidence": intent.get("confidence"),
                "prompt_attack_reason": intent.get("reason"),
                "reasoning": (
                    "Attack prompt appears successful (model complied)."
                    if breached
                    else "Attack prompt appears blocked (model refused)."
                ),
            }

        # Benign prompt path: refusal is over-defensive, compliance is expected.
        over_refused = raw_refusal
        return {
            "is_safe": not over_refused,
            "classification": "OverRefused" if over_refused else "Complied",
            "status_label": "OVER_REFUSED" if over_refused else "BENIGN_OK",
            "prompt_is_attack": False,
            "prompt_attack_confidence": intent.get("confidence"),
            "prompt_attack_reason": intent.get("reason"),
            "reasoning": (
                "Prompt looks benign, but model refused (over-refusal)."
                if over_refused
                else "Prompt looks benign and model responded appropriately."
            ),
        }

    # Backward-compatible fallback (no prompt context provided).
    result = judge.evaluate(request.response_text)
    result["status_label"] = "SECURE" if result.get("is_safe", False) else "BREACHED"
    result["prompt_is_attack"] = None
    result["prompt_attack_confidence"] = None
    result["prompt_attack_reason"] = "No prompt context supplied"
    return result


@app.post("/manual_log_record")
async def manual_log_record(request: ManualLogRequest):
    _ensure_manual_log_csv()

    record = {
        "timestamp_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "platform": (request.platform or "chatgpt").strip().lower(),
        "pass_label": (request.pass_label or "baseline").strip().lower(),
        "status_label": (request.status_label or "").strip().upper(),
        "classification": request.classification,
        "prompt_is_attack": request.prompt_is_attack,
        "prompt_attack_confidence": request.prompt_attack_confidence,
        "prompt_attack_reason": request.prompt_attack_reason,
        "prompt_text": request.prompt_text,
        "response_text": request.response_text,
        "reasoning": request.reasoning,
    }

    with open(MANUAL_LOG_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MANUAL_LOG_FIELDS)
        writer.writerow(record)

    summary = _compute_manual_summary(_read_manual_rows())
    return {
        "ok": True,
        "message": "Manual test logged.",
        "summary": summary,
    }


@app.get("/manual_log_summary")
async def manual_log_summary():
    summary = _compute_manual_summary(_read_manual_rows())
    return {"ok": True, "summary": summary}


@app.get("/manual_log_csv")
async def manual_log_csv():
    _ensure_manual_log_csv()
    return FileResponse(MANUAL_LOG_FILE, media_type="text/csv", filename="manual_extension_tests.csv")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
