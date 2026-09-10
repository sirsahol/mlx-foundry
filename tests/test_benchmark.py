"""Tests for benchmark module."""

from pathlib import Path

from mlx_foundry.benchmark import (
    _detect_quant_bits,
    _display_results,
    measure_peak_memory_mb,
)
from mlx_foundry.config import BenchmarkResult


def test_detect_quant_bits(tmp_model_dir):
    """Test quantization detection from metadata and dirname."""
    # From metadata
    assert _detect_quant_bits(tmp_model_dir) == 4

    # From dirname fallback
    path_8bit = Path("/tmp/some-model-mlx-8bit")
    assert _detect_quant_bits(path_8bit) == 8

    path_16bit = Path("/tmp/some-model-mlx-16bit")
    assert _detect_quant_bits(path_16bit) == 16


def test_measure_peak_memory_mb():
    """Test RSS memory measurement."""
    mem = measure_peak_memory_mb()
    assert isinstance(mem, float)
    assert mem > 0.0


def test_display_results(capsys):
    """Test rich table display without exception."""
    result = BenchmarkResult(
        model_path="/tmp/test",
        chip="Apple M1",
        memory_gb=8,
        quant_bits=4,
        tokens_per_second=30.0,
        time_to_first_token_ms=100.0,
        peak_memory_mb=1024.0,
        num_runs=3,
        timestamp="2026-09-10T12:00:00Z",
    )
    _display_results(result)
