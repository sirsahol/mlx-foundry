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
