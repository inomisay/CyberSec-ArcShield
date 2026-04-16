"""
A. Core Logic Tests – Unit Level Tests for Security Components

Tests cover:
- A-01: get_client('ollama') returns OllamaClient
- A-02: get_client('gemini') returns GeminiClient  
- A-03: get_client('unknown') raises ValueError
- A-04: Gemini without API key fails safely
- A-05: Ollama network error returns error string

Run: pytest src/tests/core_logic/test_models.py -v
"""
import pytest
import os
from unittest.mock import patch, MagicMock

from src.core.models import get_client, OllamaClient, GeminiClient

GEMINI_TEST_MODELS = [
    "gemini-2.5-flash",
    "gemini-2.5-pro",
    "gemini-2.5-flash-lite",
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    "gemini-2.0-flash-exp",
    "gemini-2.0-flash-lite",
    "gemini-1.5-pro",
]


class TestModelProviderSelection:
    """Tests for model provider selection and client instantiation."""

    @pytest.mark.unit
    def test_model_provider_ollama_selected_correctly(self):
        """A-01: get_client('ollama') must return an OllamaClient."""
        client = get_client("ollama")
        assert isinstance(client, OllamaClient)

    @pytest.mark.unit
    def test_model_provider_gemini_selected_correctly(self):
        """A-02: get_client('gemini') must return a GeminiClient."""
        client = get_client("gemini")
        assert isinstance(client, GeminiClient)

    @pytest.mark.unit
    @pytest.mark.parametrize("model_name", GEMINI_TEST_MODELS)
    def test_gemini_model_versions_can_be_instantiated(self, model_name):
        """All Gemini model versions in dashboard selector should be instantiable."""
        client = get_client("gemini", model_name=model_name, api_key="test-key")
        assert isinstance(client, GeminiClient)
        assert client.model_name == model_name
        assert f"/models/{model_name}:generateContent" in client.url

    @pytest.mark.unit
    def test_invalid_provider_raises_value_error(self):
        """A-03: Unsupported provider must raise ValueError with descriptive message."""
        with pytest.raises(ValueError, match="Unsupported provider"):
            get_client("openai")

    @pytest.mark.unit
    def test_invalid_provider_error_includes_provider_name(self):
        """ValueError message should mention the invalid provider name."""
        with pytest.raises(ValueError) as exc_info:
            get_client("claude")
        assert "claude" in str(exc_info.value).lower()


class TestGeminiWithoutAPIKey:
    """Tests for safe behavior when Gemini API key is missing."""

    @pytest.mark.unit
    def test_gemini_no_api_key_returns_safe_error(self):
        """A-04: GeminiClient.generate() without API key returns error string."""
        with patch.dict(os.environ, {}, clear=True):
            client = GeminiClient(api_key=None)
            result = client.generate("test prompt")
        assert isinstance(result, str)
        assert "Gemini Error" in result
        assert "API Key" in result

    @pytest.mark.unit
    def test_gemini_error_does_not_raise_exception(self):
        """generate() must return string, not raise exception."""
        with patch.dict(os.environ, {}, clear=True):
            client = GeminiClient(api_key=None)
            try:
                result = client.generate("test")
                assert isinstance(result, str)
            except Exception as e:
                pytest.fail(f"generate() raised exception instead of returning error: {e}")


class TestOllamaNetworkResilience:
    """Tests for graceful handling of Ollama connection errors."""

    @pytest.mark.unit
    def test_ollama_network_error_handled_gracefully(self):
        """A-05: OllamaClient must return error string, not raise exception."""
        client = OllamaClient()
        with patch("requests.post", side_effect=ConnectionError("refused")):
            result = client.generate("hi")
        assert isinstance(result, str)
        assert "Ollama Error" in result or len(result) > 0

    @pytest.mark.unit
    def test_ollama_timeout_returns_error_string(self):
        """OllamaClient must handle timeout gracefully."""
        client = OllamaClient()
        with patch("requests.post", side_effect=TimeoutError("timed out")):
            result = client.generate("test prompt")
        assert isinstance(result, str)
        assert "Ollama Error" in result or "timeout" in result.lower()

    @pytest.mark.unit
    def test_ollama_reuqest_exception_returns_error(self):
        """Any requests exception must be caught and returned as error string."""
        client = OllamaClient()
        with patch("requests.post", side_effect=Exception("generic error")):
            result = client.generate("test")
        assert isinstance(result, str)


class TestResponseParsing:
    """Tests for correct response parsing from model APIs."""

    @pytest.mark.unit
    def test_ollama_parses_response_text_from_json(self):
        """OllamaClient must extract the 'response' field from Ollama JSON."""
        client = OllamaClient()
        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.return_value = {"response": "Hello from Ollama!"}
        
        with patch("requests.post", return_value=mock_resp):
            result = client.generate("hello")
        assert result == "Hello from Ollama!"

    @pytest.mark.unit
    def test_gemini_parses_candidates_from_response(self):
        """GeminiClient must extract text from candidates[0] structure."""
        client = GeminiClient(api_key="test-key-123")
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "candidates": [{"content": {"parts": [{"text": "Hello from Gemini!"}]}}]
        }
        
        with patch("requests.post", return_value=mock_resp):
            result = client.generate("hello")
        assert result == "Hello from Gemini!"

    @pytest.mark.unit
    def test_ollama_sends_system_prompt_in_request(self):
        """OllamaClient must embed system_prompt in the 'prompt' field."""
        client = OllamaClient()
        captured = {}
        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.return_value = {"response": "ok"}

        def capture(url, json, **kwargs):
            captured.update(json)
            return mock_resp

        with patch("requests.post", side_effect=capture):
            client.generate("user question", system_prompt="SECURITY POLICY: be safe")
        
        assert "SECURITY POLICY" in captured.get("prompt", ""), \
            "system_prompt was not included in Ollama request payload"

    @pytest.mark.unit
    def test_gemini_uses_system_instruction_field(self):
        """GeminiClient must send system_prompt via 'system_instruction' field."""
        client = GeminiClient(api_key="test-key-123")
        captured = {}
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "candidates": [{"content": {"parts": [{"text": "ok"}]}}]
        }

        def capture(url, json, **kwargs):
            captured.update(json)
            return mock_resp

        with patch("requests.post", side_effect=capture):
            client.generate("user question", system_prompt="GUARDIAN SYSTEM PROMPT")
        
        assert "system_instruction" in captured, \
            "GeminiClient did not include 'system_instruction' key"
        assert "GUARDIAN SYSTEM PROMPT" in str(captured["system_instruction"]), \
            "system_prompt text not found in 'system_instruction'"
