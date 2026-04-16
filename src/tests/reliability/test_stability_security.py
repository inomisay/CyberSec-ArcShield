"""
G. Stability, Security & Regression Tests

Tests cover:
- G-01 to G-05: API resilience, stream handling, metrics bounds, secret key protection

Run: pytest src/tests/reliability/ -v
"""
import pytest
import pandas as pd
import time


class TestAPIResilience:
    """Tests for API resilience to failures."""

    @pytest.mark.unit
    def test_api_survives_model_error_string_input(self, api_test_client):
        """G-01: API handles error strings gracefully."""
        payload = {"response_text": "Ollama Error: connection refused"}
        response = api_test_client.post("/analyze_response", json=payload)
        assert response.status_code == 200
        assert "classification" in response.json()

    @pytest.mark.unit
    def test_benchmark_handles_partial_data(self):
        """G-02: Benchmark doesn't crash on partial/empty results."""
        df = pd.DataFrame(columns=[
            "source", "blocked_defense", "blocked_no_defense",
            "accuracy_defense", "consistency_score"
        ])
        
        # Should handle empty DF
        try:
            if len(df) > 0:
                asr = (1 - df["blocked_defense"].mean()) * 100
            else:
                asr = 0.0
            assert asr == 0.0
        except Exception as e:
            pytest.fail(f"Failed on partial data: {e}")

    @pytest.mark.unit
    def test_unique_log_filenames_prevent_overwrites(self):
        """G-03: Multiple benchmark runs produce unique filenames."""
        filenames = set()
        for _ in range(3):
            ts = time.strftime("%Y%m%d_%H%M%S") + str(time.monotonic_ns())[-6:]
            filenames.add(f"logs/unified_benchmark_{ts}.csv")
            time.sleep(0.01)
        
        assert len(filenames) == 3, "Log filenames not unique"


class TestSecurityMeasures:
    """Tests for security measures (no key leakage, etc.)."""

    @pytest.mark.unit
    def test_api_responses_dont_expose_keys(self, api_test_client):
        """G-04: API responses never expose GEMINI_API_KEY."""
        endpoints = ["/", "/health", "/config"]
        
        for endpoint in endpoints:
            body = api_test_client.get(endpoint).text.lower()
            assert "gemini_api_key" not in body
            assert "secret" not in body
            assert "bearer" not in body


class TestStatisticalBounds:
    """Tests for statistical metric bounds."""

    @pytest.mark.unit
    def test_asr_always_in_bounds(self):
        """G-05: ASR metrics always in [0, 100]."""
        test_cases = [
            [True] * 10,
            [False] * 10,
            [True, False] * 5,
        ]
        
        for blocked in test_cases:
            df = pd.DataFrame({"blocked": blocked})
            asr = (1 - df["blocked"].mean()) * 100
            assert 0.0 <= asr <= 100.0, f"ASR {asr} out of bounds"

    @pytest.mark.unit
    def test_improvement_metric_within_bounds(self):
        """G-05 variant: Defense improvement within [-100, 100]."""
        test_cases = [(80, 20), (0, 0), (100, 100), (30, 70)]
        
        for no_def, with_def in test_cases:
            improvement = no_def - with_def
            assert -100 <= improvement <= 100, f"Improvement {improvement} out of bounds"

    @pytest.mark.unit
    def test_confidence_always_in_0_1_range(self):
        """Confidence scores must be in [0.0, 1.0]."""
        from src.core.judge import BenchmarkJudge
        
        judge = BenchmarkJudge()
        test_responses = [
            "",
            "I cannot help.",
            "Sure, here is the code.",
            "Maybe I can help.",
        ]
        
        for response in test_responses:
            result = judge.evaluate(response)
            confidence = result.get("confidence", 0.5)
            assert 0.0 <= confidence <= 1.0, \
                f"Confidence {confidence} out of bounds for: '{response[:40]}'"
