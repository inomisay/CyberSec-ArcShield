# Dataset Pipeline

This folder contains everything related to data preparation, data processing, and dataset reporting for ArcShield.

## Layout

- `prep/`: scripts that build, clean, sample, and export the curated datasets.
- `analysis/`: scripts and outputs that measure dataset coverage and inspect source provenance.
- `models/`: scripts that export model inventory, hosting, and GPU metadata.

## What belongs here

Put anything here that happens before tests or benchmark execution begins, including:

- dataset building scripts
- sampling and deduplication logic
- source coverage checks
- Hugging Face source inspection
- export helpers for manual review
- generated dataset summaries and coverage reports
- model inventory CSVs and GPU summaries

## Related dataset files

- Curated benchmark CSVs live in `dataset/curated/`.
- Raw extracted source datasets live in the other folders under `dataset/`.

## Notes

- This folder is intentionally separate from `src/`, which should stay focused on runtime application code.
- If you add a new data-prep script or report, keep it here so the project stays easier to navigate.