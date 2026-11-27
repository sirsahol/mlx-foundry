from unittest.mock import MagicMock, patch

from mlx_foundry.downloader import (
    download_model_snapshot,
    find_aria2c,
    is_aria2_available,
)


def test_is_aria2_available():
    """Test checking aria2c availability."""
    assert isinstance(is_aria2_available(), bool)


def test_find_aria2c_disabled_env(monkeypatch):
    """Test that MLX_FOUNDRY_NO_ARIA2=1 disables aria2c."""
    monkeypatch.setenv("MLX_FOUNDRY_NO_ARIA2", "1")
    assert find_aria2c() is None


def test_download_model_snapshot_local_directory(tmp_path):
    """Test passing an existing local directory returns it immediately."""
    local_dir = tmp_path / "my_local_model"
    local_dir.mkdir()
    res = download_model_snapshot(str(local_dir))
    assert res == local_dir


@patch("huggingface_hub.snapshot_download")
def test_download_model_snapshot_cache_hit(mock_snapshot, tmp_path):
    """Test that local_files_only cache hit avoids downloading."""
    cached_dir = tmp_path / "cached_model"
    cached_dir.mkdir()
    mock_snapshot.return_value = str(cached_dir)

    res = download_model_snapshot("org/some-model")
    assert res == cached_dir
    mock_snapshot.assert_called_once()
    assert mock_snapshot.call_args.kwargs.get("local_files_only") is True


@patch("mlx_foundry.downloader.find_aria2c")
@patch("huggingface_hub.snapshot_download")
@patch("huggingface_hub.HfApi")
@patch("subprocess.run")
def test_download_model_snapshot_aria2c_invoked(mock_subproc, mock_hf_api, mock_snapshot, mock_find_aria, tmp_path):
    """Test that aria2c is invoked with 16 connections for large weight files."""
    mock_find_aria.return_value = "/opt/homebrew/bin/aria2c"
    # First snapshot_download call (local_files_only) raises exception (cache miss)
    mock_snapshot.side_effect = [Exception("Not in cache"), None]

    mock_api_instance = MagicMock()
    info_mock = MagicMock()
    info_mock.sha = "abcdef123456"
    mock_api_instance.model_info.return_value = info_mock

    sibling = MagicMock()
    sibling.rfilename = "model.safetensors"
    sibling.size = 100 * 1024 * 1024
    mock_api_instance.list_repo_tree.return_value = [sibling]
    mock_hf_api.return_value = mock_api_instance

    res = download_model_snapshot("org/big-model")
    assert res.name == "abcdef123456"
    assert mock_subproc.called
    cmd = mock_subproc.call_args[0][0]
    assert cmd[0] == "/opt/homebrew/bin/aria2c"
    assert "-s16" in cmd
    assert "-x16" in cmd
    assert "-k1M" in cmd
