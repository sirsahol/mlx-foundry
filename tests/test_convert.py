import json
import subprocess
from unittest.mock import MagicMock, patch

import pytest

from mlx_foundry.convert import (
    check_architecture_support,
    convert_model,
    ensure_safetensors_symlinks,
    estimate_required_space_gb,
    validate_quants,
)


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


def test_check_architecture_support(tmp_path):
    """Test architecture support checks for k2_horizon and qwen3_5_text."""
    # Test k2_horizon
    k2_dir = tmp_path / "k2_model"
    k2_dir.mkdir()
    (k2_dir / "config.json").write_text(
        json.dumps({"model_type": "k2_horizon", "architectures": ["K2HorizonForCausalLM"]})
    )
    supported, arch = check_architecture_support(str(k2_dir))
    assert supported is True
    assert arch == "k2_horizon"

    # Test qwen3_5_text
    qwen_dir = tmp_path / "qwen_model"
    qwen_dir.mkdir()
    (qwen_dir / "config.json").write_text(
        json.dumps({"model_type": "qwen3_5_text", "architectures": ["Qwen3_5ForCausalLM"]})
    )
    supported, arch = check_architecture_support(str(qwen_dir))
    assert supported is True
    assert arch == "qwen3_5_text"

    # Test unsupported
    unsupported_dir = tmp_path / "custom_unsupported"
    unsupported_dir.mkdir()
    (unsupported_dir / "config.json").write_text(
        json.dumps({"model_type": "custom_obscure_arch", "architectures": ["ObscureLM"]})
    )
    supported, arch = check_architecture_support(str(unsupported_dir))
    assert supported is False


def test_ensure_safetensors_symlinks(tmp_path):
    """Test scanning and creating symlinks for pytorch_model-*.safetensors."""

    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    pt1 = snapshot / "pytorch_model-00001-of-00002.safetensors"
    pt2 = snapshot / "pytorch_model-00002-of-00002.safetensors"
    pt_index = snapshot / "pytorch_model.safetensors.index.json"
    pt1.write_bytes(b"dummy1")
    pt2.write_bytes(b"dummy2")
    pt_index.write_text("{}")

    created = ensure_safetensors_symlinks(snapshot)
    assert len(created) == 3

    m1 = snapshot / "model-00001-of-00002.safetensors"
    m2 = snapshot / "model-00002-of-00002.safetensors"
    m_index = snapshot / "model.safetensors.index.json"

    assert m1.is_symlink()
    assert m1.read_bytes() == b"dummy1"
    assert m2.is_symlink()
    assert m2.read_bytes() == b"dummy2"
    assert m_index.is_symlink()

    # Second run does not re-create since model-*.safetensors already exist
    created_again = ensure_safetensors_symlinks(snapshot)
    assert len(created_again) == 0


@patch("mlx_foundry.convert.get_disk_free_gb", return_value=100.0)
@patch("mlx_foundry.convert.estimate_required_space_gb", return_value=5.0)
@patch("huggingface_hub.snapshot_download")
@patch("subprocess.run")
def test_convert_model_uses_snapshot_path(
    mock_subproc, mock_snapshot, mock_est_space, mock_free_space, tmp_path
):
    """Test convert_model creates symlinks and passes snapshot path to --hf-path."""
    snapshot_dir = tmp_path / "snapshot_cache"
    snapshot_dir.mkdir()
    pt = snapshot_dir / "pytorch_model-00001-of-00001.safetensors"
    pt.write_bytes(b"shard1")

    mock_snapshot.return_value = str(snapshot_dir)
    mock_subproc.return_value = subprocess.CompletedProcess(
        args=["python3", "-m", "mlx_lm.convert"],
        returncode=0,
        stdout="",
        stderr="",
    )

    results = convert_model(
        model_id="IFM/K2-Horizon-7B",
        quants=[4],
        output_dir=tmp_path / "output",
        force=True,
    )

    assert len(results) == 1
    # Check that subprocess.run was called with --hf-path pointing to snapshot_dir
    cmd_called = mock_subproc.call_args[0][0]
    hf_path_idx = cmd_called.index("--hf-path")
    assert cmd_called[hf_path_idx + 1] == str(snapshot_dir)

    # Check that symlink was created in snapshot_dir
    assert (snapshot_dir / "model-00001-of-00001.safetensors").is_symlink()
