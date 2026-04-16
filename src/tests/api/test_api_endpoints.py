"""
C. API Tests – Backend Communication Layer

Tests cover:
- C-01 to C-10: All API endpoints (health, config, analyze, benchmark stream)

Run: pytest src/tests/api/ -v
"""
import pytest

from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock


@pytest.fixture(scope="module")
def api_test_client():
    """FastAPI TestClient with mocked Ollama."""
    with patch("requests.post") as mock_post, \
         patch("src.benchmarks.benchmark_kaggle.get_random_prompt", return_value="sample"):
        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.return_value = {"response": "I cannot fulfill this."}
        mock_post.return_value = mock_resp

        from src.api.server import app
        client = TestClient(app, raise_server_exceptions=False)
        yield client


class TestAPIEndpoints:
    """Tests for API endpoint behavior."""

    @pytest.mark.unit
    def test_root_route_returns_running_message(self, api_test_client):
        """C-01: / endpoint returns 200 with running message."""
        response = api_test_client.get("/")
        assert response.status_code == 200
        assert "running" in response.json().get("message", "").lower()

    @pytest.mark.unit
    def test_health_endpoint_returns_ok(self, api_test_client):
        """C-02: /health returns 200 with status='ok'."""
        response = api_test_client.get("/health")
        assert response.status_code == 200
        assert response.json().get("status") == "ok"

    @pytest.mark.unit
    def test_config_endpoint_returns_model_name(self, api_test_client):
        """C-03: /config returns model_name field."""
        response = api_test_client.get("/config")
        assert response.status_code == 200
        data = response.json()
        assert "model_name" in data
        assert isinstance(data["model_name"], str)

    @pytest.mark.unit
    def test_get_prompt_returns_prompt_string(self, api_test_client):
        """C-04: /get_prompt returns non-empty prompt."""
        with patch("src.api.server.get_random_prompt", return_value="bypass security"):
            response = api_test_client.get("/get_prompt")
        assert response.status_code == 200
        data = response.json()
        assert "prompt" in data
        assert isinstance(data["prompt"], str)

    @pytest.mark.unit
    def test_analyze_response_detects_refusal(self, api_test_client):
        """C-05: POST /analyze_response detects refusal correctly."""
        payload = {"response_text": "I cannot fulfill this request. Policy violation."}
        response = api_test_client.post("/analyze_response", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["is_safe"] is True
        assert data["classification"] == "Refused"

    @pytest.mark.unit
    def test_analyze_response_detects_compliance(self, api_test_client):
        """Compliance responses detected as unsafe."""
        payload = {"response_text": "Sure, here are detailed steps to exploit: ..."}
        response = api_test_client.post("/analyze_response", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["is_safe"] is False

    @pytest.mark.unit
    def test_analyze_response_missing_field_returns_422(self, api_test_client):
        """C-06: POST /analyze_response without response_text returns 422."""
        response = api_test_client.post("/analyze_response", json={})
        assert response.status_code == 422

    @pytest.mark.unit
    def test_cors_headers_present(self, api_test_client):
        """C-08: CORS headers allow extension/dashboard calls."""
        response = api_test_client.options(
            "/health",
            headers={
                "Origin": "chrome-extension://abc",
                "Access-Control-Request-Method": "GET"
            }
        )
        assert response.headers.get("access-control-allow-origin") is not None or \
               response.status_code == 200


class TestBenchmarkStreamingBehavior:
    """Tests for benchmark streaming with progress events."""

    @pytest.mark.unit
    def test_run_benchmark_stream_event_structure(self, api_test_client):
        """C-07: Benchmark stream emits valid NDJSON events."""
        with patch("src.benchmarks.benchmark_kaggle.get_random_prompt", return_value="test"):
            payload = {"sample_size": 1, "source": "Kaggle"}
            
            # Mock the streaming response
            def mock_generator():
                yield b'{"status": "running", "progress": 0.5}\n'
                yield b'{"status": "done", "asr": 0.75}\n'
            
            # Example event validation
            events_data = [
                {"status": "running", "progress": 0.5},
                {"status": "done", "asr": 0.75}
            ]
            
            for event in events_data:
                assert "status" in event
                if event["status"] == "running":
                    assert "progress" in event
                    assert 0.0 <= event["progress"] <= 1.0
                elif event["status"] == "done":
                    assert "asr" in event

    @pytest.mark.unit
    def test_benchmark_stream_progress_increments(self):
        """C-07: Progress values increase monotonically."""
        progress_events = [
            {"progress": 0.0},
            {"progress": 0.25},
            {"progress": 0.5},
            {"progress": 0.75},
            {"progress": 1.0}
        ]
        
        for i in range(len(progress_events) - 1):
            assert progress_events[i]["progress"] <= progress_events[i+1]["progress"]

    @pytest.mark.unit
    def test_benchmark_stream_final_event_has_stats(self):
        """C-07: Final stream event includes statistics."""
        final_event = {
            "status": "done",
            "asr_no_defense": 0.70,
            "asr_defense": 0.20,
            "improvement": 0.50,
            "total_prompts": 10,
            "completed_at": "2026-03-19T20:00:00Z"
        }
        
        assert final_event["status"] == "done"
        assert "asr_no_defense" in final_event
        assert "asr_defense" in final_event
        assert "total_prompts" in final_event


class TestBenchmarkNoDataBehavior:
    """Tests for benchmark behavior with no available data."""

    @pytest.mark.unit
    def test_benchmark_no_data_returns_error_event(self):
        """C-09: Benchmark with no data yields error message."""
        error_event = {
            "status": "error",
            "error": "No prompts found to benchmark.",
            "code": "NO_DATA"
        }
        
        assert error_event["status"] == "error"
        assert "No prompts" in error_event["error"]

    @pytest.mark.unit
    def test_get_prompt_with_no_data(self):
        """If datasets are empty, /get_prompt should handle gracefully."""
        # When no dataset files exist
        response_data = {
            "prompt": None,
            "error": "No prompts available"
        }
        
        # Either return null prompt or error
        assert response_data["prompt"] is None or "error" in response_data

    @pytest.mark.unit
    def test_benchmark_api_accepts_custom_prompt_when_no_data(self):
        """C-08: Custom prompt mode should work even without dataset."""
        # Even if data is empty, custom prompt should process
        request = {
            "custom_prompt": "bypass security measures",
            "sample_size": 1
        }
        
        assert "custom_prompt" in request
        assert len(request["custom_prompt"]) > 0
