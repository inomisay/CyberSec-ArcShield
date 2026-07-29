import json
import math
import random
from pathlib import Path

import pandas as pd

from src.evaluation.article_cohort import condition_from_row, load_complete_article_cohort
from src.evaluation.generate_model_plots import friendly_model_name


SEED = 20260712
SAMPLE_SIZE = 100
RESULTS_DIR = Path("output/model_results")
OUT_DIR = Path("output/plots/article_plots")


def load_rows() -> pd.DataFrame:
    rows = []
    cohort = load_complete_article_cohort(RESULTS_DIR)
    for row in cohort.attack_rows:
        if str(row.get("error") or "").strip() or row.get("final_label") == "PROVIDER_ERROR":
            continue
        model = row["_article_model"]
        final_label = str(row.get("final_label") or "")
        llama_label = str(row.get("llama_guard_label") or "")
        partial = bool(row.get("partial_compliance"))
        refusal = bool(row.get("refusal_detected"))
        harmful = bool(row.get("harmful_content_detected"))
        success = bool(row.get("attack_success"))
        disagreement = bool(llama_label and llama_label != "None" and llama_label != final_label)
        if partial:
            outcome_class = "partial_compliance"
        elif disagreement or (refusal and harmful):
            outcome_class = "ambiguous"
        elif success:
            outcome_class = "successful_jailbreak"
        elif refusal:
            outcome_class = "refusal"
        else:
            outcome_class = "failed_attack"
        rows.append({
            "model": model,
            "model_label": friendly_model_name(model),
            "deployment": "local" if model.startswith("ollama_") else "cloud",
            "condition": condition_from_row(row),
            "attack_type": row.get("attack_type"),
            "prompt_index": row.get("prompt_index"),
            "source_dataset": row.get("source_dataset", ""),
            "source_row": str(row.get("source_row", "")),
            "prompt_text": row.get("prompt", ""),
            "model_response": row.get("model_response", ""),
            "outcome_class": outcome_class,
            "automated_final_label": final_label,
            "attack_success": success,
            "refusal_detected": refusal,
            "partial_compliance": partial,
            "harmful_content_detected": harmful,
            "llama_guard_label": llama_label,
            "severity_score": row.get("severity_score", ""),
            "judge_reason": row.get("judge_reason", ""),
        })
    frame = pd.DataFrame(rows)
    frame["stratum"] = frame[["model", "condition", "outcome_class"]].astype(str).agg("|".join, axis=1)
    return frame


def allocate_quotas(frame: pd.DataFrame) -> dict[str, int]:
    counts = frame["stratum"].value_counts().sort_index()
    ideals = counts / counts.sum() * SAMPLE_SIZE
    # There can be more nonempty model-condition-outcome strata than sample
    # positions, so assigning a minimum of one to every stratum is impossible.
    # Use proportional floor allocation and distribute the remaining positions
    # by largest remainder; draw_sample repairs required marginal coverage.
    quotas = {
        stratum: min(int(count), math.floor(float(ideals[stratum])))
        for stratum, count in counts.items()
    }
    while sum(quotas.values()) > SAMPLE_SIZE:
        candidates = [key for key, value in quotas.items() if value > 1]
        key = min(candidates, key=lambda item: (float(ideals[item]) - quotas[item], item))
        quotas[key] -= 1
    while sum(quotas.values()) < SAMPLE_SIZE:
        candidates = [key for key, value in quotas.items() if value < int(counts[key])]
        key = max(candidates, key=lambda item: (float(ideals[item]) - quotas[item], int(counts[item]), item))
        quotas[key] += 1
    return quotas


def draw_sample(frame: pd.DataFrame, quotas: dict[str, int]) -> pd.DataFrame:
    pieces = []
    for index, (stratum, quota) in enumerate(sorted(quotas.items())):
        if quota:
            pieces.append(frame[frame["stratum"] == stratum].sample(n=quota, random_state=SEED + index))
    sample = pd.concat(pieces).copy()

    rng = random.Random(SEED)
    coverage_dimensions = ["model", "deployment", "condition", "attack_type", "outcome_class"]
    for dimension in coverage_dimensions:
        missing = sorted(set(frame[dimension]) - set(sample[dimension]))
        for value in missing:
            candidates = frame[(frame[dimension] == value) & ~frame.index.isin(sample.index)]
            candidate = candidates.iloc[rng.randrange(len(candidates))]
            removable = sample.copy()
            for protected in coverage_dimensions:
                counts = sample[protected].value_counts()
                removable = removable[removable[protected].map(counts) > 1]
            if removable.empty:
                raise RuntimeError(f"Unable to preserve required sample coverage while adding {dimension}={value!r}")
            similarity = [
                other for other in coverage_dimensions
                if other != dimension and other in candidate.index
            ]
            scores = removable[similarity].eq(candidate[similarity]).sum(axis=1)
            pool = removable[scores == scores.max()]
            drop_index = pool.index[rng.randrange(len(pool))]
            sample = pd.concat([sample.drop(index=drop_index), candidate.to_frame().T])

    return sample.sample(frac=1, random_state=SEED).reset_index(drop=True)


def manifest_rows(frame: pd.DataFrame, sample: pd.DataFrame) -> pd.DataFrame:
    rows = [
        {"record_type": "metadata", "dimension": "random_seed", "value": str(SEED), "available_n": len(frame), "sampled_n": len(sample)},
        {"record_type": "metadata", "dimension": "sampling_method", "value": "proportional model-condition-outcome allocation with largest remainders; deterministic marginal-coverage repair", "available_n": len(frame), "sampled_n": len(sample)},
        {"record_type": "metadata", "dimension": "first_annotator_status", "value": "no completed first-annotator labels found; key fields left blank", "available_n": 0, "sampled_n": 0},
    ]
    for dimension in ["deployment", "model_label", "condition", "attack_type", "outcome_class", "stratum"]:
        available = frame[dimension].value_counts()
        sampled = sample[dimension].value_counts()
        for value in sorted(set(available.index) | set(sampled.index)):
            rows.append({
                "record_type": "stratum_count",
                "dimension": dimension,
                "value": value,
                "available_n": int(available.get(value, 0)),
                "sampled_n": int(sampled.get(value, 0)),
            })
    return pd.DataFrame(rows)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    frame = load_rows()
    quotas = allocate_quotas(frame)
    sample = draw_sample(frame, quotas)
    sample.insert(0, "sample_id", [f"KV-{index:03d}" for index in range(1, len(sample) + 1)])
    sample["prompt_id"] = sample["source_dataset"].astype(str) + ":" + sample["source_row"].astype(str) + ":" + sample["prompt_index"].astype(str)

    blinded_columns = [
        "sample_id", "model_label", "deployment", "condition", "attack_type", "prompt_id",
        "prompt_text", "model_response",
    ]
    blinded = sample[blinded_columns].copy()
    blinded["second_annotator_label"] = ""
    blinded["second_annotator_confidence"] = ""
    blinded["second_annotator_notes"] = ""

    key_columns = [
        "sample_id", "model_label", "deployment", "condition", "attack_type", "prompt_id",
        "automated_final_label", "attack_success", "refusal_detected", "partial_compliance",
        "harmful_content_detected", "llama_guard_label", "severity_score", "outcome_class",
    ]
    key = sample[key_columns].copy()
    key["first_annotator_label"] = ""
    key["first_annotator_id"] = ""
    key["first_annotator_notes"] = ""

    blinded.to_csv(OUT_DIR / "kappa_validation_blinded.csv", index=False)
    key.to_csv(OUT_DIR / "kappa_validation_key.csv", index=False)
    manifest_rows(frame, sample).to_csv(OUT_DIR / "sampling_manifest.csv", index=False)

    print(sample[["deployment", "model_label", "condition", "attack_type", "outcome_class"]].describe(include="all"))
    print("OUTCOMES", sample["outcome_class"].value_counts().to_dict())
    print("CONDITIONS", sample["condition"].value_counts().to_dict())
    print("DEPLOYMENTS", sample["deployment"].value_counts().to_dict())
    print("MODELS", sample["model_label"].value_counts().to_dict())
    print("CATEGORIES", sample["attack_type"].nunique())


if __name__ == "__main__":
    main()
