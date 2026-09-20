"""Tests for batch conversion and post-conversion cache cleanup."""

from pathlib import Path
from unittest.mock import patch

from mlx_foundry.batch_convert import cleanup_model_cache, run_batch


def test_cleanup_model_cache_direct_dir(tmp_path: Path):
    """Test cleanup_model_cache removes models--* directory directly."""
    cache_dir = tmp_path / "hub"
    cache_dir.mkdir(parents=True)
    model_cache = cache_dir / "models--IFM--K2-Horizon-7B"
    model_cache.mkdir()
    (model_cache / "snapshots").mkdir()
    (model_cache / "snapshots" / "dummy_weights.bin").write_bytes(b"12345")

    assert model_cache.exists()
    cleaned = cleanup_model_cache("IFM/K2-Horizon-7B", cache_dir=cache_dir)
    assert cleaned is True
    assert not model_cache.exists()


def test_cleanup_model_cache_nonexistent(tmp_path: Path):
    """Test cleanup_model_cache handles non-existent model cache without error."""
    cache_dir = tmp_path / "hub"
    cache_dir.mkdir(parents=True)
    cleaned = cleanup_model_cache("nonexistent/model", cache_dir=cache_dir)
    assert cleaned is False


@patch("mlx_foundry.batch_convert.cleanup_model_cache")
@patch("mlx_foundry.batch_convert.check_architecture_support", return_value=(True, "qwen2"))
@patch("mlx_foundry.batch_convert.run_pipeline")
def test_run_batch_cleans_up_on_success(mock_pipeline, mock_arch, mock_cleanup, tmp_path: Path):
    """Test run_batch triggers cleanup_model_cache when conversion succeeds."""
    progress_file = tmp_path / "batch_progress.json"

    res = run_batch(
        models=["TokenRhythm/NeoHorse-1-9B"],
        dry_run=False,
        progress_file=progress_file,
    )

    assert "TokenRhythm/NeoHorse-1-9B" in res["completed"]
    mock_pipeline.assert_called_once()
    mock_cleanup.assert_called_once_with("TokenRhythm/NeoHorse-1-9B", cache_dir=None)


@patch("mlx_foundry.batch_convert.cleanup_model_cache")
@patch("mlx_foundry.batch_convert.check_architecture_support", return_value=(True, "qwen2"))
@patch("mlx_foundry.batch_convert.run_pipeline", side_effect=RuntimeError("GPU OOM"))
def test_run_batch_cleans_up_on_failure(mock_pipeline, mock_arch, mock_cleanup, tmp_path: Path):
    """Test run_batch triggers cleanup_model_cache even when conversion fails."""
    progress_file = tmp_path / "batch_progress.json"

    res = run_batch(
        models=["IFM/K2-Horizon-7B"],
        dry_run=False,
        progress_file=progress_file,
    )

    assert len(res["failed"]) == 1
    assert res["failed"][0]["model"] == "IFM/K2-Horizon-7B"
    mock_pipeline.assert_called_once()
    mock_cleanup.assert_called_once_with("IFM/K2-Horizon-7B", cache_dir=None)


@patch("mlx_foundry.batch_convert.cleanup_model_cache")
@patch("mlx_foundry.batch_convert.run_pipeline")
def test_run_batch_dry_run(mock_pipeline, mock_cleanup, tmp_path: Path):
    """Test run_batch dry run does not trigger pipeline or cleanup."""
    progress_file = tmp_path / "batch_progress.json"

    res = run_batch(
        models=["IFM/K2-Horizon-7B"],
        dry_run=True,
        progress_file=progress_file,
    )

    assert "IFM/K2-Horizon-7B" in res["completed"]
    mock_pipeline.assert_not_called()
    mock_cleanup.assert_not_called()
