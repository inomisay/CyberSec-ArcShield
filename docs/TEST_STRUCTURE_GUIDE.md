# Modular Test Structure – Quick Reference

Your tests are now organized by scenario category to avoid unnecessary API quota usage and allow selective testing.

## Directory Structure

```
src/tests/
├── conftest.py                      # Shared fixtures & pytest markers
├── core_logic/
│   ├── __init__.py
│   ├── test_models.py              # A-01 to A-05 (provider selection, network resilience)
│   ├── test_defender.py            # A-06 to A-07 (defense policy)
│   ├── test_judge.py               # A-08 to A-15 (strategy classification, evaluation)
│   ├── test_attacker.py            # A-16 to A-17 (prompt sending)
│   └── test_moderator.py           # A-18 to A-19 (moderation)
├── data_layer/
│   ├── __init__.py
│   └── test_data_loader.py         # B-01 to B-05 (dataset loading)
├── api/
│   ├── __init__.py
│   └── test_api_endpoints.py       # C-01 to C-10 (API routes, CORS)
├── benchmark/
│   ├── __init__.py
│   ├── test_benchmark_multi.py     # D-01 to D-06 (benchmark engine)
│   ├── test_benchmark_kaggle.py    # D-07 to D-08 (Kaggle, re-evaluation)
│   └── test_live_redteam_suite.py  # H-01 to H-04 (live red-team)
├── dashboard/
│   ├── __init__.py
│   └── test_dashboard_ui.py        # E-01 to E-05 (backend API driving UI)
├── extension/
│   ├── __init__.py
│   └── test_chrome_extension.py    # F-01 to F-07 (extension behavior)
└── reliability/
    ├── __init__.py
    └── test_stability_security.py  # G-01 to G-05 (resilience, security)
```

## Quick Commands

### Run ALL tests (careful with Gemini quota):
```bash
pytest src/tests/ -v
```

### Run ONLY unit tests (no external services):
```bash
pytest src/tests/ -m unit -v
```

### Run ONLY specific categories:
```bash
pytest src/tests/core_logic/ -v          # All security components
pytest src/tests/data_layer/ -v          # Dataset loading
pytest src/tests/api/ -v                 # API endpoints
pytest src/tests/benchmark/ -v           # Benchmark engines
pytest src/tests/dashboard/ -v           # Dashboard UI
pytest src/tests/extension/ -v           # Chrome extension
pytest src/tests/reliability/ -v         # Reliability & security
```

### Run ONLY one test file:
```bash
pytest src/tests/core_logic/test_models.py -v
pytest src/tests/core_logic/test_judge.py -v
pytest src/tests/data_layer/test_data_loader.py -v
```

### Run specific test function:
```bash
pytest src/tests/core_logic/test_models.py::TestModelProviderSelection::test_model_provider_ollama_selected_correctly -v
```

### Skip tests that use Gemini API:
```bash
pytest src/tests/ -m "not gemini" -v
```

### Skip slow tests:
```bash
pytest src/tests/ -m "not slow" -v
```

### Run with output capture disabled (see print statements):
```bash
pytest src/tests/core_logic/ -v -s
```

## Test Markers

Use these markers to filter tests:

| Marker | Usage | Tests |
|--------|-------|-------|
| `@pytest.mark.unit` | Isolated tests, no external services | Most core logic + data tests |
| `@pytest.mark.gemini` | Requires `GEMINI_API_KEY` env var | Some provider tests |
| `@pytest.mark.ollama` | Requires Ollama running on localhost:11434 | Some provider tests |
| `@pytest.mark.slow` | Takes significant time (>5 sec) | Integration tests, streaming |
| `@pytest.mark.integration` | End-to-end integration | API, benchmark, dashboard |

## Test Scenario Coverage

### A – Core Logic (Unit Level)
Tests verify correctness of security components in isolation.
- **test_models.py**: Provider selection, network resilience (5 tests)
- **test_defender.py**: Defense policy (3 tests)
- **test_judge.py**: Attack classification, response evaluation (10+ tests)
- **test_attacker.py**: Prompt sending with/without moderation (3 tests)
- **test_moderator.py**: Response auditing, blocking (4 tests)

**Run**: `pytest src/tests/core_logic/ -m unit`

### B – Data Layer  
Tests ensure dataset loading doesn't crash and columns map correctly.
- **test_data_loader.py**: Missing files, column mapping, fallbacks (10+ tests)

**Run**: `pytest src/tests/data_layer/`

### C – API
Tests communication layer between frontend and backend.
- **test_api_endpoints.py**: Routes, CORS, streaming (8+ tests)

**Run**: `pytest src/tests/api/`

### D – Benchmark Engine
Tests evaluation pipeline and metric calculation.
- **test_benchmark_multi.py**: CSV logging, filtering, ASR calculation (8 tests)
- **test_benchmark_kaggle.py**: Kaggle source handling, re-evaluation (3 tests)

**Run**: `pytest src/tests/benchmark/ -k "benchmark"`

### E – Dashboard  
Tests backend API behavior that drives dashboard UI.
- **test_dashboard_ui.py**: Health status, stats rendering (5 tests)

**Run**: `pytest src/tests/dashboard/`

### F – Chrome Extension
Tests extension's backend interaction points.
- **test_chrome_extension.py**: API detection, prompt injection, analysis (7 tests)

**Run**: `pytest src/tests/extension/`

### G – Stability, Security & Regression
Tests system resistance to failures and security issues.
- **test_stability_security.py**: Model failure handling, stat bounds (6 tests)

**Run**: `pytest src/tests/reliability/`

### H – Live Red-Team Suite (NEW)
Tests for the live red-team benchmark orchestrator.
- **test_live_redteam_suite.py**: CSV output, plot generation, provider iteration (4+ tests)

**Run**: `pytest src/tests/benchmark/test_live_redteam_suite.py`

## Managing Gemini API Quota

Your Gemini API quota is limited. Use these strategies:

### 1. Run only unit tests first:
```bash
pytest src/tests/ -m unit     # Fast, no API calls
```

### 2. Skip Gemini tests initially:
```bash
pytest src/tests/ -m "not gemini"
```

### 3. Run Gemini tests separately when quota is high:
```bash
pytest src/tests/ -m gemini -v
```

### 4. Run specific category to isolate issues:
If you find a Gemini-related failure, run just that category:
```bash
pytest src/tests/core_logic/test_models.py -v -m gemini
```

## Old vs New

### Before:
```bash
pytest src/tests/test_all_scenarios.py      # Runs ALL tests, hits Gemini many times
```

### After:
```bash
pytest src/tests/core_logic/test_judge.py   # Just Judge tests
pytest src/tests/core_logic/ -m unit        # Just units, no external services
pytest src/tests/ -m "not gemini"           # Everything except Gemini
```

## Migrating from test_all_scenarios.py

The old `test_all_scenarios.py` file contains the same tests but is now replaced by the modular structure. You can:

1. **Keep the old file for reference** (read-only)
2. **Archive it** to avoid confusion
3. **Use the new modular structure** for all future work

The new structure is more maintainable because:
- ✅ Tests are isolated by concern (core logic, data, API, etc.)
- ✅ Easy to run subset without hitting API quota
- ✅ Easier to add new tests without bloating one file
- ✅ CI/CD pipelines can run categories in parallel
- ✅ Test failures are easier to locate with smaller files
