"""Publishing module — uploads models to HuggingFace Hub with proper metadata."""

import shutil
from pathlib import Path
from typing import Any

from rich.console import Console

from mlx_foundry.utils import fetch_model_info

console = Console()


def publish_model(
    model_path: Path,
    hf_repo: str,
    source_model: str,
    quant_bits: int,
    private: bool = False,
    cleanup: bool = True,
) -> str:
    """Publish a converted MLX model to HuggingFace Hub.

    This function:
    1. Creates the HF repo if it doesn't exist
    2. Sets all metadata tags (library_name, pipeline_tag, base_model, license, etc.)
    3. Uploads all files from the model directory
    4. Optionally cleans up the local model directory after successful upload

    Args:
        model_path: Path to the converted model directory.
        hf_repo: Target HuggingFace repo ID (e.g., 'SirSahOl/Qwen3-0.6B-chat-mlx-4bit').
        source_model: Original HuggingFace model ID.
        quant_bits: Quantization level.
        private: Whether the repo should be private.
        cleanup: Whether to delete local files after successful upload.

    Returns:
        str: URL of the published model.
    """
    from huggingface_hub import HfApi, create_repo

    api = HfApi()
    model_path = Path(model_path)

    # Fetch source model info for tags
    try:
        source_info = fetch_model_info(source_model)
    except Exception as e:  # noqa: BLE001
        console.print(f"[yellow]Warning: Could not fetch source model info: {e}[/yellow]")
        source_info = {
            "license": "unknown",
            "pipeline_tag": "text-generation",
            "tags": [],
        }

    # Build tags
    tags = ["mlx", "safetensors", "apple-silicon"]

    if quant_bits < 16:
        tags.append(f"{quant_bits}-bit")

    # Add conversational tag if source model has it or if it's a chat model
    if "conversational" in source_info.get("tags", []) or "chat" in hf_repo.lower():
        tags.append("conversational")

    # Add model-specific tags from source
    for tag in source_info.get("tags", []):
        if tag not in tags and not tag.startswith("base_model:") and tag != "region:us":
            tags.append(tag)

    console.print(f"[bold blue]Publishing to {hf_repo}...[/bold blue]")

    # Create repo
    try:
        repo_url = create_repo(
            repo_id=hf_repo,
            repo_type="model",
            exist_ok=True,
            private=private,
        )
        console.print(f"[green]✓[/green] Repo created/confirmed: {repo_url}")
    except Exception as e:
        console.print(f"[red]✗ Failed to create repo: {e}[/red]")
        raise

    # Update repo metadata via model card header (YAML frontmatter)
    model_card_meta = {
        "library_name": "mlx",
        "pipeline_tag": source_info.get("pipeline_tag", "text-generation"),
        "license": source_info.get("license", "unknown"),
        "tags": tags,
        "base_model": source_model,
    }

    if quant_bits < 16:
        model_card_meta["base_model_relation"] = "quantized"

    # Ensure README.md has proper YAML frontmatter
    _ensure_frontmatter(model_path, model_card_meta)

    # Upload all files
    console.print("  Uploading files...")
    try:
        api.upload_folder(
            folder_path=str(model_path),
            repo_id=hf_repo,
            repo_type="model",
            commit_message=f"Upload {quant_bits}-bit MLX conversion of {source_model}",
        )
    except Exception as e:
        console.print(f"[red]✗ Upload failed: {e}[/red]")
        raise

    url = f"https://huggingface.co/{hf_repo}"
    console.print(f"[green]✓[/green] Published: {url}")

    # Cleanup
    if cleanup:
        console.print(f"  Cleaning up local files at {model_path}...")
        shutil.rmtree(model_path)
        console.print("[green]✓[/green] Local files cleaned up")

    return url


def _ensure_frontmatter(model_path: Path, metadata: dict[str, Any]) -> None:
    """Ensure the README.md has proper YAML frontmatter for HuggingFace.

    If the README.md already has frontmatter, this merges the metadata.
    If it doesn't, this prepends frontmatter.

    Args:
        model_path: Path to the model directory.
        metadata: Metadata dict to include in frontmatter.
    """

    def _dump_yaml(data: dict[str, Any]) -> str:
        try:
            import yaml

            return yaml.dump(data, default_flow_style=False, sort_keys=False)
        except ImportError:
            lines = []
            for k, v in data.items():
                if isinstance(v, list):
                    lines.append(f"{k}:")
                    for item in v:
                        lines.append(f"  - {item}")
                else:
                    lines.append(f"{k}: {v}")
            return "\n".join(lines) + "\n"

    readme_path = model_path / "README.md"

    if not readme_path.exists():
        frontmatter = _dump_yaml(metadata)
        content = f"---\n{frontmatter}---\n\n# {model_path.name}\n"
        with open(readme_path, "w") as f:
            f.write(content)
        return

    content = readme_path.read_text()

    if content.startswith("---"):
        # Already has frontmatter
        return

    # Prepend frontmatter
    frontmatter = _dump_yaml(metadata)
    content = f"---\n{frontmatter}---\n\n{content}"
    with open(readme_path, "w") as f:
        f.write(content)
