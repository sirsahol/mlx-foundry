"""Tests for model conversion module."""

import subprocess
from unittest.mock import MagicMock, patch

import pytest

from mlx_foundry.convert import convert_model, estimate_required_space_gb, validate_quants


def test_validate_quants():
    """Test valid and invalid quantization levels."""
    validate_quants([4, 8, 16])
    validate_quants([2, 3, 6])

    with pytest.raises(ValueError, match="Invalid quantization levels"):
        validate_quants([1, 4, 8])

    with pytest.raises(ValueError, match="Invalid quantization levels"):
        validate_quants([32])


@patch("huggingface_hub.model_info")
def test_estimate_required_space_gb(mock_model_info):
    """Test required disk space estimation."""
    mock_info = MagicMock()
    mock_sibling1 = MagicMock()
    mock_sibling1.size = 1024 * 1024 * 1024  # 1 GB
    mock_sibling2 = MagicMock()
    mock_sibling2.size = 1024 * 1024 * 1024  # 1 GB
    mock_info.siblings = [mock_sibling1, mock_sibling2]
    mock_model_info.return_value = mock_info

    # Original size: 2 GB
    # Output quants: 4-bit (0.25 * 2 = 0.5 GB) + 8-bit (0.5 * 2 = 1.0 GB) = 1.5 GB
    # Buffer: 1 GB
    # Total: 2 + 1.5 + 1.0 = 4.5 GB
    est = estimate_required_space_gb("test/model", [4, 8])
    assert round(est, 1) == 4.5


@patch("mlx_foundry.convert.get_disk_free_gb", return_value=100.0)
@patch("mlx_foundry.convert.estimate_required_space_gb", return_value=5.0)
@patch("subprocess.run")
def test_convert_model(mock_subproc, mock_est_space, mock_free_space, tmp_path):
    """Test convert_model subprocess execution and result generation."""
    mock_subproc.return_value = subprocess.CompletedProcess(
        args=["python3", "-m", "mlx_lm.convert"],
        returncode=0,
        stdout="",
        stderr="",
    )

    results = convert_model(
        model_id="test-org/dummy-model",
        quants=[4],
        output_dir=tmp_path,
        force=True,
    )

    assert len(results) == 1
    assert results[0].quant_bits == 4
    assert results[0].source_model == "test-org/dummy-model"
    assert (tmp_path / "dummy-model-mlx-4bit" / "conversion_metadata.json").exists()
