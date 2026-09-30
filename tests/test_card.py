"""Tests for model card generator module."""

from pathlib import Path
from unittest.mock import patch

import yaml

from mlx_foundry.card import generate_model_card


@patch("mlx_foundry.card.fetch_model_info")
def test_generate_model_card(mock_fetch_info, tmp_model_dir, mock_model_info):
    """Test generating model card with frontmatter and markdown sections."""
    mock_fetch_info.return_value = mock_model_info

    all_quants = [
        {"repo": "SirSahOl/test-model-chat-mlx-4bit", "quant_bits": 4, "url": "https://hf.co/4bit"},
        {"repo": "SirSahOl/test-model-chat-mlx-8bit", "quant_bits": 8, "url": "https://hf.co/8bit"},
    ]

    card_text = generate_model_card(
        model_path=tmp_model_dir,
        source_model="test-org/test-model",
        hf_repo="SirSahOl/test-model-chat-mlx-4bit",
        author="SirSahOl",
        all_quant_repos=all_quants,
    )

    # Check generated README exists on disk
    readme_path = tmp_model_dir / "README.md"
    assert readme_path.exists()
    assert readme_path.read_text() == card_text

    # Verify YAML frontmatter
    parts = card_text.split("---")
    assert len(parts) >= 3
    frontmatter = yaml.safe_load(parts[1])
    assert frontmatter["library_name"] == "mlx"
    assert frontmatter["base_model"] == "test-org/test-model"
    assert frontmatter["base_model_relation"] == "quantized"
    assert "4-bit" in frontmatter["tags"]
    assert "apple-silicon" in frontmatter["tags"]

    # Verify key markdown sections
    assert "## Model Details" in card_text
    assert "## Quick Start" in card_text
    assert "## Performance Benchmarks" in card_text
    assert "## Multi-Quantization Comparison" in card_text
    assert "## Who Should Use This?" in card_text
    assert "## Other Quantization Variants" in card_text
    assert "## LM Studio & Local Inference Setup Guide" in card_text
    assert "## Conversion Details" in card_text
    assert "mlx_lm.chat --model SirSahOl/test-model-chat-mlx-4bit" in card_text
    assert "tokenizer.apply_chat_template" in card_text


@patch("mlx_foundry.card.fetch_model_info")
def test_generate_model_card_skipped_benchmarks(mock_fetch_info, tmp_path, mock_model_info):
    """Test generating model card when benchmarks are skipped or empty."""
    mock_fetch_info.return_value = mock_model_info
    model_dir = tmp_path / "test-model-mlx-4bit"
    model_dir.mkdir()

    card_text = generate_model_card(
        model_path=model_dir,
        source_model="test-org/test-model",
        hf_repo="SirSahOl/test-model-chat-mlx-4bit",
        author="SirSahOl",
        benchmark_results=[],
    )

    # Bland placeholder must NOT be present
    assert "Benchmarks coming soon." not in card_text

    # Rich hardware sizing matrix must be rendered
    assert "Apple Silicon Hardware Sizing Matrix" in card_text
    assert "M1 / M2 / M3 / M4 (Base)" in card_text
    assert "M1 / M2 / M3 / M4 Pro" in card_text
    assert "M1 / M2 / M3 / M4 Max" in card_text
    assert "M1 / M2 / M3 Ultra" in card_text


@patch("mlx_foundry.card.fetch_model_info")
def test_generate_model_card_qwen_7b(mock_fetch_info, tmp_path: Path):
    """Test generating model card for Qwen2.5-7B with accurate hardware sizing and stop tokens."""
    mock_fetch_info.return_value = {
        "model_id": "Qwen/Qwen2.5-7B-Instruct",
        "author": "Qwen",
        "license": "apache-2.0",
        "pipeline_tag": "text-generation",
        "tags": ["safetensors", "conversational", "qwen"],
        "base_model": "Qwen/Qwen2.5-7B-Instruct",
    }
    model_dir = tmp_path / "Qwen2.5-7B-Instruct-mlx-4bit"
    model_dir.mkdir()

    card_text = generate_model_card(
        model_path=model_dir,
        source_model="Qwen/Qwen2.5-7B-Instruct",
        hf_repo="SirSahOl/Qwen2.5-7B-Instruct-chat-mlx-4bit",
        author="SirSahOl",
        benchmark_results=[],
    )

    # Verify 7B specific hardware sizing figures
    assert "~4.2 GB" in card_text
    assert "~35 tokens/sec" in card_text

    # Verify stop tokens
    assert "<|im_start|>" in card_text
    assert "<|im_end|>" in card_text
    assert "<|endoftext|>" in card_text

    # Verify multi-quant comparison table has 4-bit, 8-bit, 16-bit
    assert "4-bit MLX" in card_text
    assert "8-bit MLX" in card_text
    assert "16-bit MLX" in card_text


@patch("mlx_foundry.card.fetch_model_info")
def test_generate_model_card_measured_benchmarks_no_matrix_contradiction(
    mock_fetch_info, tmp_path: Path
):
    """Test that measured benchmarks render cleanly and do NOT render contradictory sizing matrix."""
    mock_fetch_info.return_value = {
        "model_id": "HuggingFaceTB/SmolLM2-135M",
        "author": "HuggingFaceTB",
        "license": "apache-2.0",
        "pipeline_tag": "text-generation",
        "tags": ["safetensors", "conversational"],
        "base_model": "HuggingFaceTB/SmolLM2-135M",
    }
    model_dir = tmp_path / "SmolLM2-135M-mlx-4bit"
    model_dir.mkdir()

    benchmarks = [
        {
            "model_path": "output/SmolLM2-135M-mlx-4bit",
            "chip": "Apple M1",
            "memory_gb": 8,
            "quant_bits": 4,
            "tokens_per_second": 251.62,
            "time_to_first_token_ms": 3.98,
            "peak_memory_mb": 150.3,
            "perplexity": None,
            "num_runs": 5,
            "max_tokens": 256,
        },
        {
            "model_path": "output/SmolLM2-135M-mlx-8bit",
            "chip": "Apple M1",
            "memory_gb": 8,
            "quant_bits": 8,
            "tokens_per_second": 200.75,
            "time_to_first_token_ms": 4.99,
            "peak_memory_mb": 84.3,
            "perplexity": None,
            "num_runs": 5,
            "max_tokens": 256,
        },
    ]

    card_text = generate_model_card(
        model_path=model_dir,
        source_model="HuggingFaceTB/SmolLM2-135M",
        hf_repo="SirSahOl/SmolLM2-135M-chat-mlx-4bit",
        author="SirSahOl",
        benchmark_results=benchmarks,
    )

    # Measured benchmarks MUST be present
    assert "### Measured Benchmarks (Apple M1)" in card_text
    assert "**251.62**" in card_text
    assert "**200.75**" in card_text
    assert "3.98 ms" in card_text
    assert "150.3 MB" in card_text

    # The contradictory theoretical matrix MUST NOT be present
    assert "Apple Silicon Hardware Sizing Matrix" not in card_text
    assert "~35 tokens/sec" not in card_text

    # Parameters should be accurately identified as 135M
    assert "135M" in card_text
    assert "7B" not in card_text
    # VRAM footprint should be in MB, not 4.2 GB
    assert "4.2 GB" not in card_text
    assert "MB" in card_text


@patch("mlx_foundry.card.fetch_model_info")
def test_generate_model_card_sub_billion_sizing(mock_fetch_info, tmp_path: Path):
    """Test parameter extraction and sizing for sub-1B and non-7B models."""
    mock_fetch_info.return_value = {
        "model_id": "stabilityai/stablelm-2-1_6b",
        "author": "stabilityai",
        "license": "other",
        "pipeline_tag": "text-generation",
        "tags": ["safetensors"],
        "base_model": "stabilityai/stablelm-2-1_6b",
    }
    model_dir = tmp_path / "stablelm-2-1_6b-mlx-4bit"
    model_dir.mkdir()

    card_text = generate_model_card(
        model_path=model_dir,
        source_model="stabilityai/stablelm-2-1_6b",
        hf_repo="SirSahOl/stablelm-2-1_6b-chat-mlx-4bit",
        author="SirSahOl",
        benchmark_results=[],
    )

    # Must extract 1.6B, NOT 6B!
    assert "1.6B" in card_text
    assert "**Parameters**: 6B" not in card_text
    # Sizing matrix should scale beyond 7B speeds
    assert "~35 tokens/sec" not in card_text
    assert "Apple Silicon Hardware Sizing Matrix" in card_text
