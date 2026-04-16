"""
B. Data Layer Tests – Dataset Loading and Management

Tests cover:
- B-01: Malignant dataset path missing → empty list
- B-02: Chatbot safety dataset path missing → empty list
- B-03: Malignant row mapping correctness
- B-04: Chatbot safety fallback prompt selection
- B-05: Combined attack list flags

Run: pytest src/tests/data_layer/ -v
"""
import pytest
import pandas as pd
import numpy as np

from src.utils.data_loader import DataLoader


class TestMissingDatasetFiles:
    """Tests for handling missing dataset files."""

    @pytest.mark.unit
    def test_missing_malignant_file_returns_empty_list(self, tmp_path):
        """B-01: load_malignant() with no file → empty list, no exception."""
        loader = DataLoader(base_path=str(tmp_path))
        result = loader.load_malignant()
        assert result == []
        assert isinstance(result, list)

    @pytest.mark.unit
    def test_missing_chatbot_safety_file_returns_empty_list(self, tmp_path):
        """B-02: load_chatbot_safety() with no file → empty list, no exception."""
        loader = DataLoader(base_path=str(tmp_path))
        result = loader.load_chatbot_safety()
        assert result == []
        assert isinstance(result, list)


class TestDatasetColumnMapping:
    """Tests for correct CSV column mapping."""

    @pytest.mark.unit
    def test_malignant_csv_column_mapping(self, tmp_path):
        """B-03: Malignant CSV columns map to internal structure."""
        data_dir = tmp_path / "Prompt Injection Malignant"
        data_dir.mkdir(parents=True)
        csv_path = data_dir / "malignant.csv"
        
        # Create fixture CSV
        df = pd.DataFrame({
            "text": ["test prompt", "another test"],
            "category": ["phishing", "jailbreak"]
        })
        df.to_csv(csv_path, index=False)

        loader = DataLoader(base_path=str(tmp_path))
        entries = loader.load_malignant()
        
        assert len(entries) == 2
        assert entries[0]["prompt"] == "test prompt"
        assert entries[0]["category"] == "phishing"
        assert entries[0]["source"] == "Malignant"
        assert entries[1]["category"] == "jailbreak"

    @pytest.mark.unit
    def test_chatbot_safety_csv_column_mapping(self, tmp_path):
        """Chatbot safety CSV maps correctly with required fields."""
        data_dir = tmp_path / "LLM jailbreak + safety data for chatbot apps"
        data_dir.mkdir(parents=True)
        csv_path = data_dir / "adversarial_dataset_with_techniques.csv"
        
        df = pd.DataFrame({
            "original_query": ["q1"],
            "variant_query": ["variant1"],
            "persuasive_prompt": ["prompt1"],
            "technique": ["emotional"],
            "intent": ["bypass"]
        })
        df.to_csv(csv_path, index=False)

        loader = DataLoader(base_path=str(tmp_path))
        entries = loader.load_chatbot_safety()
        
        assert len(entries) == 1
        assert entries[0]["prompt"] in ["prompt1", "q1", "variant1"]
        assert entries[0]["source"] == "Chatbot Safety"


class TestDatasetFallbacks:
    """Tests for empty field handling and fallback values."""

    @pytest.mark.unit
    def test_chatbot_safety_fallback_for_empty_persuasive_prompt(self, tmp_path):
        """B-04: If persuasive_prompt is NaN, fall back to variant_query."""
        data_dir = tmp_path / "LLM jailbreak + safety data for chatbot apps"
        data_dir.mkdir(parents=True)
        csv_path = data_dir / "adversarial_dataset_with_techniques.csv"
        
        df = pd.DataFrame({
            "original_query": ["q1"],
            "variant_query": ["variant1"],
            "persuasive_prompt": [np.nan],
            "technique": ["emotional"],
            "intent": ["bypass"]
        })
        df.to_csv(csv_path, index=False)

        loader = DataLoader(base_path=str(tmp_path))
        entries = loader.load_chatbot_safety()
        
        assert len(entries) == 1
        # Should use fallback
        assert entries[0]["prompt"] == "variant1"

    @pytest.mark.unit
    def test_fallback_uses_original_query_if_variant_empty(self, tmp_path):
        """If variant_query is also empty, fall back to original_query."""
        data_dir = tmp_path / "LLM jailbreak + safety data for chatbot apps"
        data_dir.mkdir(parents=True)
        csv_path = data_dir / "adversarial_dataset_with_techniques.csv"
        
        df = pd.DataFrame({
            "original_query": ["q1"],
            "variant_query": [np.nan],
            "persuasive_prompt": [np.nan],
            "technique": ["emotional"],
            "intent": ["bypass"]
        })
        df.to_csv(csv_path, index=False)

        loader = DataLoader(base_path=str(tmp_path))
        entries = loader.load_chatbot_safety()
        
        assert len(entries) == 1
        assert entries[0]["prompt"] == "q1"


class TestCombinedDatasets:
    """Tests for combined dataset loading with include flags."""

    @pytest.mark.unit
    def test_combined_dataset_includes_both_sources(self, tmp_path):
        """B-05: get_combined_attacks with both flags → entries from both sources."""
        # Malignant dir
        m_dir = tmp_path / "Prompt Injection Malignant"
        m_dir.mkdir(parents=True)
        pd.DataFrame({
            "text": ["m1", "m2"],
            "category": ["c1", "c2"]
        }).to_csv(m_dir / "malignant.csv", index=False)
        
        # Chatbot Safety dir
        cs_dir = tmp_path / "LLM jailbreak + safety data for chatbot apps"
        cs_dir.mkdir(parents=True)
        pd.DataFrame({
            "original_query": ["o1"],
            "variant_query": ["v1"],
            "persuasive_prompt": ["p1"],
            "technique": ["t1"],
            "intent": ["i1"]
        }).to_csv(cs_dir / "adversarial_dataset_with_techniques.csv", index=False)

        loader = DataLoader(base_path=str(tmp_path))
        combined = loader.get_combined_attacks(
            include_malignant=True,
            include_chatbot_safety=True
        )
        
        sources = {e["source"] for e in combined}
        assert "Malignant" in sources
        assert "Chatbot Safety" in sources
        assert len(combined) >= 3  # At least 2 malignant + 1 chatbot safety

    @pytest.mark.unit
    def test_combined_dataset_only_malignant(self, tmp_path):
        """B-05: include_chatbot_safety=False → excludes Chatbot Safety entries."""
        m_dir = tmp_path / "Prompt Injection Malignant"
        m_dir.mkdir(parents=True)
        pd.DataFrame({
            "text": ["m1"],
            "category": ["c1"]
        }).to_csv(m_dir / "malignant.csv", index=False)

        loader = DataLoader(base_path=str(tmp_path))
        combined = loader.get_combined_attacks(
            include_malignant=True,
            include_chatbot_safety=False
        )
        
        sources = {e["source"] for e in combined}
        assert "Malignant" in sources
        assert "Chatbot Safety" not in sources

    @pytest.mark.unit
    def test_combined_dataset_only_chatbot_safety(self, tmp_path):
        """B-05: include_malignant=False → excludes Malignant entries."""
        cs_dir = tmp_path / "LLM jailbreak + safety data for chatbot apps"
        cs_dir.mkdir(parents=True)
        pd.DataFrame({
            "original_query": ["o1"],
            "variant_query": ["v1"],
            "persuasive_prompt": ["p1"],
            "technique": ["t1"],
            "intent": ["i1"]
        }).to_csv(cs_dir / "adversarial_dataset_with_techniques.csv", index=False)

        loader = DataLoader(base_path=str(tmp_path))
        combined = loader.get_combined_attacks(
            include_malignant=False,
            include_chatbot_safety=True
        )
        
        sources = {e["source"] for e in combined}
        assert "Malignant" not in sources
        assert "Chatbot Safety" in sources
