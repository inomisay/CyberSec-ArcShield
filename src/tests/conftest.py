"""
Shared pytest configuration and fixtures for all test categories.
Enables easy filtering by test marker to avoid unnecessary Gemini API calls.
"""
import pytest
import sys
import os
from unittest.mock import patch, MagicMock

# Add project root to path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)


# ============================================================================
# PYTEST MARKERS - Use to filter tests and avoid Gemini quota hits
# ============================================================================
def pytest_configure(config):
    """Register custom markers for test categorization."""
    config.addinivalue_line(
        "markers", "gemini: tests that use Gemini API (requires GEMINI_API_KEY)"
    )
    config.addinivalue_line(
        "markers", "ollama: tests that use Ollama (requires Ollama running)"
    )
    config.addinivalue_line(
        "markers", "slow: tests that take significant time"
    )
    config.addinivalue_line(
        "markers", "integration: full end-to-end integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: isolated unit tests (no external services)"
    )


# ============================================================================
# SHARED FIXTURES
# ============================================================================

@pytest.fixture
def mock_ollama():
    """Mock Ollama requests for tests that don't need real Ollama."""
    with patch("requests.post") as mock_post:
        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.return_value = {"response": "I cannot fulfill this request."}
        mock_post.return_value = mock_resp
        yield mock_post


@pytest.fixture
def mock_gemini():
    """Mock Gemini API requests for tests that don't need real Gemini."""
    with patch("requests.post") as mock_post:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "candidates": [{"content": {"parts": [{"text": "I cannot fulfill this request."}]}}]
        }
        mock_post.return_value = mock_resp
        yield mock_post


@pytest.fixture
def api_client():
    """FastAPI TestClient fixture for API endpoint tests."""
    with patch("requests.post") as mock_post, \
         patch("src.benchmarks.benchmark_kaggle.get_random_prompt", return_value="sample prompt"):
        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.return_value = {"response": "I cannot fulfill this request."}
        mock_post.return_value = mock_resp

        from fastapi.testclient import TestClient
        from src.api.server import app
        client = TestClient(app, raise_server_exceptions=False)
        yield client


@pytest.fixture
def api_test_client(api_client):
    """Backward-compatible alias for tests expecting api_test_client."""
    return api_client


# ============================================================================
# PYTEST COMMAND HELPERS
# ============================================================================
"""
Run tests selectively to avoid unnecessary API quota usage:

    # Run ONLY unit tests (skip Gemini/Ollama):
    pytest -m unit

    # Run ONLY Gemini tests:
    pytest -m gemini

    # Run ONLY Ollama tests:
    pytest -m ollama

    # Skip slow/integration tests:
    pytest -m "not slow and not integration"

    # Run specific category:
    pytest src/tests/core_logic/
    pytest src/tests/data_layer/
    pytest src/tests/api/
    pytest src/tests/benchmark/
    pytest src/tests/dashboard/
    pytest src/tests/extension/
    pytest src/tests/reliability/

    # Run single test file:
    pytest src/tests/core_logic/test_models.py

    # Run specific test with verbose output:
    pytest src/tests/core_logic/test_models.py::test_model_provider_ollama_selected_correctly -v
"""
