"""Tests for model publishing module."""

from unittest.mock import MagicMock, patch

import yaml

from mlx_foundry.publish import _ensure_frontmatter, publish_model


def test_ensure_frontmatter_creates_file(tmp_path):
    """Test _ensure_frontmatter when README does not exist."""
    model_dir = tmp_path / "model"
    model_dir.mkdir()

    meta = {
        "library_name": "mlx",
        "pipeline_tag": "text-generation",
        "tags": ["mlx", "4-bit"],
    }
    _ensure_frontmatter(model_dir, meta)

    readme = model_dir / "README.md"
    assert readme.exists()
    content = readme.read_text()
    assert content.startswith("---")
    parsed = yaml.safe_load(content.split("---")[1])
    assert parsed["library_name"] == "mlx"


def test_ensure_frontmatter_prepends(tmp_path):
    """Test _ensure_frontmatter when README exists without frontmatter."""
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    readme = model_dir / "README.md"
    readme.write_text("# Existing Readme\nSome text")

    meta = {"library_name": "mlx"}
    _ensure_frontmatter(model_dir, meta)

    content = readme.read_text()
    assert content.startswith("---")
    assert "# Existing Readme" in content


@patch("mlx_foundry.publish.fetch_model_info")
@patch("huggingface_hub.HfApi")
@patch("huggingface_hub.create_repo")
def test_publish_model(
    mock_create_repo, mock_hf_api, mock_fetch_info, tmp_model_dir, mock_model_info
):
    """Test publishing flow with mocked HF API."""
    mock_fetch_info.return_value = mock_model_info
    mock_create_repo.return_value = "https://huggingface.co/SirSahOl/test-model"

    api_instance = MagicMock()
    mock_hf_api.return_value = api_instance

    url = publish_model(
        model_path=tmp_model_dir,
        hf_repo="SirSahOl/test-model",
        source_model="test-org/test-model",
        quant_bits=4,
        cleanup=False,
    )

    assert url == "https://huggingface.co/SirSahOl/test-model"
    mock_create_repo.assert_called_once_with(
        repo_id="SirSahOl/test-model",
        repo_type="model",
        exist_ok=True,
        private=False,
    )
    api_instance.upload_folder.assert_called_once()
