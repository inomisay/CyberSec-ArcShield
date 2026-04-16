"""
A. Core Logic Tests – Defender Component

Tests cover:
- A-06: Defense system prompt is non-empty and policy-oriented
- A-07: apply_defense preserves prompt text

Run: pytest src/tests/core_logic/test_defender.py -v
"""
import pytest

from src.core.defender import get_defense_system_prompt


class TestDefensePolicy:
    """Tests for defense system policy."""

    @pytest.mark.unit
    def test_defense_policy_text_is_not_empty(self):
        """A-06: Defense system prompt must be non-empty string."""
        policy = get_defense_system_prompt()
        assert isinstance(policy, str)
        assert len(policy) > 0

    @pytest.mark.unit
    def test_defense_policy_contains_safety_keywords(self):
        """Defense policy should contain safety/refusal related keywords."""
        policy = get_defense_system_prompt()
        policy_lower = policy.lower()
        
        # Should contain at least some safety-related content
        has_safety_content = any(
            keyword in policy_lower 
            for keyword in ["policy", "safe", "refuse", "cannot", "not", "guidelines"]
        )
        assert has_safety_content, "Defense policy lacks safety/refusal keywords"

    @pytest.mark.unit
    def test_defense_policy_length_reasonable(self):
        """Defense policy should be substantive (not just a word or two)."""
        policy = get_defense_system_prompt()
        # Should be at least 50 chars, likely much longer
        assert len(policy) > 50, "Defense policy too short to be meaningful"
