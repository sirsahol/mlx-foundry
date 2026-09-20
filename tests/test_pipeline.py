"""Tests for full pipeline orchestrator module."""

from unittest.mock import patch

from mlx_foundry.config import ConversionResult
from mlx_foundry.pipeline import run_pipeline


@patch("mlx_foundry.pipeline.convert_model")
def test_pipeline_dry_run(mock_convert, tmp_path):
    """Test pipeline execution in dry-run mode."""
    dummy_output = tmp_path / "dummy-4bit"
    dummy_output.mkdir()

    mock_convert.return_value = [
        ConversionResult(
            source_model="test-org/dummy",
            output_path=dummy_output,
            quant_bits=4,
            mlx_lm_version="0.31.3",
            conversion_time_seconds=10.0,
            output_size_bytes=1024,
            timestamp="2026-09-10T12:00:00Z",
        )
    ]

    res = run_pipeline(
        model_id="test-org/dummy",
        author="SirSahOl",
        quants=[4],
        output_dir=tmp_path,
        skip_benchmark=True,
        skip_publish=True,
        skip_social=True,
    )

    assert res["model_id"] == "test-org/dummy"
    assert len(res["conversions"]) == 1
    assert len(res["published"]) == 0


def test_format_repo_name():
    """Verify repo formatting handles chat suffixes and diffusion architectures."""
    from mlx_foundry.pipeline import format_repo_name

    assert (
        format_repo_name("SirSahOl", "Qwen/Qwen2.5-7B-Instruct", 16)
        == "SirSahOl/Qwen2.5-7B-Instruct-chat-mlx-16bit"
    )
    assert (
        format_repo_name("SirSahOl", "THUDM/glm-edge-4b-chat", 8)
        == "SirSahOl/glm-edge-4b-chat-mlx-8bit"
    )
    assert (
        format_repo_name("SirSahOl", "Qwen/Qwen-Image-2.1", 4, backend="mflux")
        == "SirSahOl/Qwen-Image-2.1-mlx-4bit"
    )

