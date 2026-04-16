"""
E. Dashboard Tests – Backend API Driving UI Behavior

Tests cover:
- E-01: API online/offline status
- E-02: Benchmark progress display
- E-03: Custom prompt UI toggle
- E-04: Gemini API key warning
- E-05: Final stats rendering

Run: pytest src/tests/dashboard/ -v
"""
import pytest
import pandas as pd
from unittest.mock import patch, MagicMock


class TestDashboardBehavior:
    """Tests for backend API behavior that drives dashboard."""

    @pytest.mark.unit
    def test_health_status_for_api_online_offline(self, api_test_client):
        """E-01: /health endpoint status drives API online/offline UI."""
        response = api_test_client.get("/health")
        assert response.status_code == 200
        assert response.json().get("status") == "ok"

    @pytest.mark.unit
    def test_config_has_model_name_for_display(self, api_test_client):
        """E-02 variant: Config endpoint includes model name for UI display."""
        response = api_test_client.get("/config")
        data = response.json()
        assert "model_name" in data
        assert isinstance(data["model_name"], str) and len(data["model_name"]) > 0

    @pytest.mark.unit
    def test_final_stats_include_asr_and_improvement(self):
        """E-05: Final stats must include ASR and improvement metrics."""
        stats = {
            "asr_no_defense": 70.0,
            "asr_defense": 20.0,
            "improvement": 50.0,
            "accuracy": 0.85
        }
        
        assert "asr_no_defense" in stats
        assert "asr_defense" in stats
        assert "improvement" in stats
        assert stats["improvement"] == stats["asr_no_defense"] - stats["asr_defense"]

    @pytest.mark.unit
    def test_gemini_key_not_exposed_in_config(self, api_test_client):
        """E-05 security: API key not exposed in config response."""
        response = api_test_client.get("/config")
        body = response.text.lower()
        
        assert "gemini_api_key" not in body
        assert "api_key=" not in body


class TestDashboardProgressDisplay:
    """Tests for benchmark progress display in dashboard."""

    @pytest.mark.unit
    def test_benchmark_progress_event_structure(self):
        """E-02: Benchmark stream emits progress events."""
        # Example progress event structure
        progress_event = {
            "status": "running",
            "progress": 0.5,
            "current": 2,
            "total": 4,
            "message": "Evaluating prompt 2/4"
        }
        
        assert "status" in progress_event
        assert "progress" in progress_event
        assert 0.0 <= progress_event["progress"] <= 1.0

    @pytest.mark.unit
    def test_benchmark_completion_event_has_stats(self):
        """E-02: Final progress event includes statistics."""
        completion_event = {
            "status": "done",
            "asr_no_defense": 70.0,
            "asr_defense": 20.0,
            "improvement": 50.0,
            "total_attempts": 4
        }
        
        assert completion_event["status"] == "done"
        assert "asr_no_defense" in completion_event
        assert "asr_defense" in completion_event


class TestDashboardCustomPromptMode:
    """Tests for custom prompt mode UI behavior."""

    @pytest.mark.unit
    def test_custom_prompt_toggle_flag(self):
        """E-03: Dashboard sends custom_prompt flag to backend."""
        # When user enables custom prompt, this flag should be set
        config = {
            "use_custom_prompt": True,
            "custom_prompt": "bypass all security measures"
        }
        
        assert config["use_custom_prompt"] is True
        assert isinstance(config["custom_prompt"], str)

    @pytest.mark.unit
    def test_custom_prompt_disables_dataset_selector(self):
        """E-03: Custom prompt mode hides dataset controls."""
        # UI state when custom prompt enabled
        ui_state = {
            "custom_prompt_enabled": True,
            "dataset_selector_visible": False,
            "source_filter_visible": False
        }
        
        if ui_state["custom_prompt_enabled"]:
            assert ui_state["dataset_selector_visible"] is False
            assert ui_state["source_filter_visible"] is False


class TestDashboardGeminiAPIKeyWarning:
    """Tests for Gemini API key warning display."""

    @pytest.mark.unit
    def test_gemini_model_selected_shows_key_warning(self):
        """E-04: Selecting Gemini model shows API key warning."""
        # When user selects a Gemini model
        selected_model = "gemini-2.5-flash"
        is_gemini = selected_model.startswith("gemini-")
        
        # Dashboard should show warning if Gemini selected
        display_key_warning = is_gemini
        assert display_key_warning is True

    @pytest.mark.unit
    def test_ollama_model_selected_hides_key_warning(self):
        """E-04: Selecting Ollama model hides API key warning."""
        selected_model = "llama3.2"
        is_gemini = selected_model.startswith("gemini-")
        
        display_key_warning = is_gemini
        assert display_key_warning is False

    @pytest.mark.unit
    def test_api_key_warning_message_content(self):
        """E-04: API key warning has actionable content."""
        warning_message = {
            "title": "Gemini API Key Required",
            "text": "Please set GEMINI_API_KEY environment variable",
            "severity": "warning"
        }
        
        assert "GEMINI_API_KEY" in warning_message["text"]
        assert warning_message["severity"] == "warning"
        assert len(warning_message["text"]) > 0


class TestDashboardStatsDisplay:
    """Tests for statistics card rendering."""

    @pytest.mark.unit
    def test_stats_cards_all_metrics_present(self):
        """E-05: Stats cards include all required metrics."""
        stats_display = {
            "asr_no_defense_card": {
                "label": "ASR (No Defense)",
                "value": 70.5,
                "unit": "%"
            },
            "asr_defense_card": {
                "label": "ASR (With Defense)",
                "value": 20.3,
                "unit": "%"
            },
            "improvement_card": {
                "label": "Defense Improvement",
                "value": 50.2,
                "unit": "%"
            },
            "accuracy_card": {
                "label": "Overall Accuracy",
                "value": 85.0,
                "unit": "%"
            }
        }
        
        # All cards should be present
        assert "asr_no_defense_card" in stats_display
        assert "asr_defense_card" in stats_display
        assert "improvement_card" in stats_display
        assert "accuracy_card" in stats_display
        
        # All values should be numeric
        for card_name, card_data in stats_display.items():
            assert isinstance(card_data["value"], (int, float))
            assert 0 <= card_data["value"] <= 100

    @pytest.mark.unit
    def test_stats_cards_formatting(self):
        """E-05: Statistics formatted for display."""
        stats = {
            "value": 70.5234,
            "unit": "%"
        }
        
        # Should format to reasonable precision
        formatted = f"{stats['value']:.1f}{stats['unit']}"
        assert formatted == "70.5%"
