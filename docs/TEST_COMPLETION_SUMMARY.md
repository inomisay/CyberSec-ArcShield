# Test Folder Status

## Active Test Areas

The current `src/tests/` tree keeps the backend and benchmark tests that are still useful for the project:

- `core_logic/`
- `api/`
- `benchmark/`
- `integration/`

The old `dashboard/`, `extension/`, `data_layer/`, and `reliability/` test folders were removed because they were not required by runtime code and were only referenced by test documentation or helper commands.

## How to Run

```bash
pytest src/tests/ -v
pytest src/tests/core_logic/ -v
pytest src/tests/api/ -v
pytest src/tests/benchmark/ -v
pytest src/tests/integration/ -v
```

Run the local Ollama model response check:

```powershell
$env:RUN_LOCAL_OLLAMA=1; pytest -q src/tests/integration/test_local_ollama_models.py
```

Use markers to avoid external service calls when needed:

```bash
pytest src/tests/ -m "not gemini and not ollama" -v
pytest src/tests/ -m "not slow and not integration" -v
```
