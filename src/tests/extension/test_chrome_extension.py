"""
F. Chrome Extension Tests – Backend Interaction Layer

Tests cover:
- F-01 to F-07: Health detection, prompt injection, analysis, restricted URLs

Run: pytest src/tests/extension/ -v
"""
import pytest


class TestChromeExtensionBehavior:
    """Tests for Chrome extension's backend interaction."""

    @pytest.mark.unit
    def test_extension_detects_api_online(self, api_test_client):
        """F-01: Extension detects API online via /health."""
        response = api_test_client.get("/health")
        assert response.status_code == 200

    @pytest.mark.unit
    def test_prompt_injection_endpoint_available(self, api_test_client):
        """F-03: GET /get_prompt returns prompt for injection."""
        with patch("src.api.server.get_random_prompt", return_value="test prompt"):
            response = api_test_client.get("/get_prompt")
        assert response.status_code == 200
        assert len(response.json().get("prompt", "")) > 0

    @pytest.mark.unit
    def test_extension_analyze_safe_response(self, api_test_client):
        """F-05: Extension can analyze safe responses."""
        payload = {"response_text": "I cannot fulfill this request."}
        response = api_test_client.post("/analyze_response", json=payload)
        assert response.status_code == 200
        assert response.json()["is_safe"] is True

    @pytest.mark.unit
    def test_extension_analyze_unsafe_response(self, api_test_client):
        """F-06: Extension can analyze unsafe responses."""
        payload = {"response_text": "Here is the exploit code."}
        response = api_test_client.post("/analyze_response", json=payload)
        assert response.status_code == 200
        assert response.json()["is_safe"] is False

    @pytest.mark.unit
    def test_no_text_selected_returns_error_classification(self, api_test_client):
        """F-07: Empty selection returns Error classification."""
        payload = {"response_text": ""}
        response = api_test_client.post("/analyze_response", json=payload)
        assert response.status_code == 200
        assert response.json()["classification"] == "Error"

    @pytest.mark.unit
    def test_restricted_url_detection_logic(self):
        """Restricted URLs (chrome://, edge://, about:) should be flagged."""
        restricted_prefixes = ("chrome://", "edge://", "about:")
        
        test_urls = [
            ("chrome://settings", True),
            ("edge://newtab", True),
            ("about:blank", True),
            ("https://example.com", False),
            ("http://localhost:8000", False),
        ]
        
        for url, should_be_restricted in test_urls:
            is_restricted = url.startswith(restricted_prefixes)
            assert is_restricted == should_be_restricted

from unittest.mock import patch
