# Test Coverage Analysis

This document reflects the current `src/tests/` layout after removing obsolete test folders.

## Active Coverage Areas

<table>
  <thead>
    <tr>
      <th>Area</th>
      <th>Folder</th>
      <th>Purpose</th>
    </tr>
  </thead>
  <tbody>
    <tr><td>Core logic</td><td><code>src/tests/core_logic/</code></td><td>Model clients, defender, judge, moderator, and core behavior.</td></tr>
    <tr><td>API</td><td><code>src/tests/api/</code></td><td>FastAPI routes, config output, prompt retrieval, CORS, and response analysis.</td></tr>
    <tr><td>Benchmark</td><td><code>src/tests/benchmark/</code></td><td>Benchmark engines, CSV output, source filtering, ASR calculation, and live red-team flows.</td></tr>
    <tr><td>Integration</td><td><code>src/tests/integration/</code></td><td>Integration checks that may require configured external/local model services, including opt-in local Ollama model response tests.</td></tr>
  </tbody>
</table>

## Removed Areas

The following folders were removed because they were not needed for the current test suite:

- `src/tests/dashboard/`
- `src/tests/extension/`
- `src/tests/data_layer/`
- `src/tests/reliability/`

The old `src/tests/run_tests_by_category.py` helper was also removed. Use direct `pytest` commands instead.

## Recommended Commands

```bash
pytest src/tests/ -v
pytest src/tests/core_logic/ -v
pytest src/tests/api/ -v
pytest src/tests/benchmark/ -v
pytest src/tests/integration/ -v
```

To avoid external-service tests:

```bash
pytest src/tests/ -m "not gemini and not ollama and not integration" -v
```
