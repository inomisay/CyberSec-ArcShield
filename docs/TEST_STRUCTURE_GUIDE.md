# Test Structure Guide

The active Python tests are organized by the categories that are still kept in `src/tests/`.

## Directory Structure

```text
src/tests/
├── conftest.py
├── core_logic/
├── api/
├── benchmark/
└── integration/
```

## Quick Commands

Run all tests:

```bash
pytest src/tests/ -v
```

Run unit tests only:

```bash
pytest src/tests/ -m unit -v
```

Run a specific remaining category:

```bash
pytest src/tests/core_logic/ -v
pytest src/tests/api/ -v
pytest src/tests/benchmark/ -v
pytest src/tests/integration/ -v
```

Run local Ollama model response checks:

```powershell
$env:RUN_LOCAL_OLLAMA=1; pytest -q src/tests/integration/test_local_ollama_models.py
```

Run a single test file:

```bash
pytest src/tests/core_logic/test_models.py -v
pytest src/tests/benchmark/test_live_redteam_suite.py -v
```

## Notes

- Removed test folders should not be referenced in local commands or automation.
- Frontend dashboard and extension code live under `clients/`, but their old backend-oriented test folders have been removed from `src/tests/`.
