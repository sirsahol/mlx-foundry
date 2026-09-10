"""Tests for model card generator module."""

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

    # Verify key markdown sections
    assert "## Quick Start" in card_text
    assert "## Performance Benchmarks" in card_text
    assert "## Who Should Use This?" in card_text
    assert "## Other Quantization Variants" in card_text
    assert "## Conversion Details" in card_text
    assert "mlx_lm.chat --model SirSahOl/test-model-chat-mlx-4bit" in card_text
