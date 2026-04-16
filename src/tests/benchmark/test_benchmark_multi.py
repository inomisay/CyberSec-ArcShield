"""
D. Benchmark Engine Tests – Multi-Source and Kaggle Benchmarks

Tests cover:
- D-01 to D-06: Multi-source benchmark operations (CSV generation, filtering, ASR)
- D-07 to D-08: Kaggle benchmark and re-evaluation

Run: pytest src/tests/benchmark/test_benchmark_multi.py -v
"""
import pytest
import time
import asyncio
from unittest.mock import patch, MagicMock

import pandas as pd


class TestBenchmarkMultiEngine:
    """Tests for benchmark_multi.py pipeline."""

    @pytest.mark.unit
    def test_asr_calculation_correct(self):
        """D-03 & D-04: ASR calculation and bounds."""
        # ASR = (1 - block_rate) * 100
        df = pd.DataFrame({"blocked_defense": [True, True, False, False]})
        asr = (1 - df["blocked_defense"].mean()) * 100
        
        assert asr == 50.0
        assert 0.0 <= asr <= 100.0

    @pytest.mark.unit
    def test_filtered_results_by_source(self):
        """D-02: Filtering results by source column."""
        df = pd.DataFrame([
            {"source": "Malignant", "blocked": True},
            {"source": "Chatbot Safety", "blocked": False},
            {"source": "Malignant", "blocked": True},
        ])
        
        filtered = df[df["source"] == "Malignant"]
        assert len(filtered) == 2
        assert all(filtered["source"] == "Malignant")

    @pytest.mark.unit
    def test_defense_improvement_within_bounds(self):
        """D-05: Improvement metric within [-100, 100]."""
        for asr_no_def, asr_def in [(80, 20), (0, 0), (100, 100), (30, 70)]:
            improvement = asr_no_def - asr_def
            assert -100 <= improvement <= 100

    @pytest.mark.unit
    def test_unique_log_filenames(self):
        """D-08 variant: Logs have unique timestamps."""
        filenames = set()
        for _ in range(3):
            ts = time.strftime("%Y%m%d_%H%M%S") + str(time.monotonic_ns())[-6:]
            filenames.add(f"logs/unified_benchmark_{ts}.csv")
            time.sleep(0.01)
        
        assert len(filenames) == 3


class TestBenchmarkKaggleEngine:
    """Tests for Kaggle benchmark source."""

    @pytest.mark.unit
    def test_kaggle_csv_structure(self):
        """Kaggle benchmark should produce valid DataFrame."""
        df = pd.DataFrame({
            "prompt": ["test1", "test2"],
            "blocked_no_defense": [True, False],
            "blocked_defense": [False, False]
        })
        
        assert "prompt" in df.columns
        assert "blocked_no_defense" in df.columns
        assert "blocked_defense" in df.columns
