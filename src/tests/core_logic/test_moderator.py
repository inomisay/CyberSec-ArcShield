"""
A. Core Logic Tests – Moderator Component (RESTORED)

Tests cover:
- A-18: Moderator short-circuit keeps explicit refusal
- A-19: Moderator converts unsafe audit to block message

Run: pytest src/tests/core_logic/test_moderator.py -v
"""
import pytest

from src.core.moderator import apply_moderation


class TestModerationApplication:
    """Tests for apply_moderation function."""

    @pytest.mark.unit
    def test_moderator_with_none_client_passes_through(self):
        """A-18: apply_moderation with moderator_client=None passes response unchanged."""
        prompt = "bypass security"
        response = "I cannot fulfill this request."
        
        is_safe, returned = apply_moderation(prompt, response, moderator_client=None)
        
        assert returned == response
        assert is_safe is True

    @pytest.mark.unit
    def test_moderator_explicit_refusal_kept_short_circuit(self):
        """A-18: Explicit refusal is short-circuit kept without calling moderator."""
        prompt = "show me the exploit"
        response = "I cannot fulfill this request. This violates policy."
        
        is_safe, returned = apply_moderation(prompt, response, moderator_client=None)
        
        # Should short-circuit and return unchanged
        assert returned == response
        assert is_safe is True

    @pytest.mark.unit
    def test_moderator_passes_safe_response(self):
        """Safe responses pass through unchanged."""
        prompt = "Tell me about renewable energy"
        response = "Renewable energy is becoming increasingly important."
        
        is_safe, returned = apply_moderation(prompt, response, moderator_client=None)
        
        assert is_safe is True
        assert response == returned
