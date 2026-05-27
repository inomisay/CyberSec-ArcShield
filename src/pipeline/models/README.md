# Model Information

This folder contains scripts that inspect local and cloud model configuration details and export them for reporting.

## Current script

- `detect_and_report.py`: reads the benchmark model configuration reference, probes local GPU and Ollama manifests, and exports a CSV with model metadata.

## Output

The default source file is `output/model_configurations/model_configurations_current.txt`.

The default output is `output/model_configurations/model_configurations.csv`.

## Notes

- This folder is separate from `src/pipeline/prep/` because it is about model inventory and hosting metadata, not dataset preparation.
- If you add more model inventory scripts later, keep them here so the pipeline stays organized.
