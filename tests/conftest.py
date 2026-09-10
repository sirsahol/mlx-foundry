"""Shared test fixtures."""

import json
from pathlib import Path

import pytest


@pytest.fixture
def tmp_model_dir(tmp_path: Path):
    """Create a temporary directory with mock model files."""
    model_dir = tmp_path / "test-model-mlx-4bit"
    model_dir.mkdir()

    # Create mock conversion metadata
    metadata = {
        "source_model": "test-org/test-model",
        "output_path": str(model_dir),
        "quant_bits": 4,
        "mlx_lm_version": "0.31.3",
        "conversion_time_seconds": 42.5,
        "output_size_bytes": 500_000_000,
        "timestamp": "2026-09-10T12:00:00+00:00",
        "command": "python3 -m mlx_lm.convert --hf-path test-org/test-model -q --q-bits 4",
    }
    with open(model_dir / "conversion_metadata.json", "w") as f:
        json.dump(metadata, f)

    # Create mock benchmark results
    benchmark = {
        "model_path": str(model_dir),
        "chip": "Apple M1",
        "memory_gb": 8,
        "quant_bits": 4,
        "tokens_per_second": 25.5,
        "time_to_first_token_ms": 120.3,
        "peak_memory_mb": 2048.0,
        "perplexity": None,
        "prompt": "test prompt",
        "max_tokens": 256,
        "num_runs": 5,
        "timestamp": "2026-09-10T12:00:00+00:00",
    }
    with open(model_dir / "benchmark_results.json", "w") as f:
        json.dump(benchmark, f)

    return model_dir


@pytest.fixture
def mock_model_info():
    """Mock HuggingFace model info."""
    return {
        "model_id": "test-org/test-model",
        "author": "test-org",
        "license": "apache-2.0",
        "pipeline_tag": "text-generation",
        "tags": ["safetensors", "conversational"],
        "base_model": "test-org/test-model",
    }
