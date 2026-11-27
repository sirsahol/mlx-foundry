"""High-speed multi-connection model downloader for MLX Foundry using aria2c."""

from __future__ import annotations

import fnmatch
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from rich.console import Console

console = Console()

DEFAULT_IGNORE_PATTERNS = [
    "*.onnx",
    "*.onnx_data",
    "*.bin",
    "*.pt",
    "onnx/*",
    "*.msgpack",
    "*.h5",
    "*.tflite",
    "*.ot",
    "*.mp4",
    "*.avi",
    "*.mov",
    "assets/*",
]


def find_aria2c() -> str | None:
    """Locate the aria2c executable on the system."""
    if os.environ.get("MLX_FOUNDRY_NO_ARIA2", "0") == "1":
        return None
    which_bin = shutil.which("aria2c")
    if which_bin:
        return which_bin
    homebrew_bin = Path("/opt/homebrew/bin/aria2c")
    if homebrew_bin.exists() and os.access(homebrew_bin, os.X_OK):
        return str(homebrew_bin)
    return None


def is_aria2_available() -> bool:
    """Check if aria2c is installed and usable."""
    return find_aria2c() is not None


def get_hf_token() -> str | None:
    """Get Hugging Face authorization token if available."""
    token = os.environ.get("HF_TOKEN")
    if token:
        return token
    try:
        from huggingface_hub import get_token

        return get_token()
    except Exception:
        return None


def download_model_snapshot(
    model_id: str,
    ignore_patterns: list[str] | None = None,
    max_connections_per_server: int = 16,
    split_pieces: int = 16,
    max_concurrent_downloads: int = 4,
) -> Path:
    """Download a Hugging Face model repository snapshot.

    If aria2c is available, fetches large files using 16-connection parallel chunking
    with automatic resume support. Falls back gracefully to standard huggingface_hub
    snapshot_download.

    Args:
        model_id: HuggingFace model identifier (e.g., 'nvidia/LocateAnything-3B').
        ignore_patterns: Glob patterns for files to skip downloading.
        max_connections_per_server: aria2c -x connection limit (default 16).
        split_pieces: aria2c -s split pieces per file (default 16).
        max_concurrent_downloads: aria2c -j max concurrent files (default 4).

    Returns:
        Path: Path to the local snapshot directory containing the downloaded model.
    """
    patterns = ignore_patterns if ignore_patterns is not None else DEFAULT_IGNORE_PATTERNS

    # Step 1: If model is already a local directory, return it directly
    local_candidate = Path(model_id).expanduser()
    if local_candidate.is_dir():
        return local_candidate

    from huggingface_hub import HfApi, snapshot_download

    # Step 2: Check if model snapshot is already fully cached and satisfied locally
    try:
        cached_result = snapshot_download(
            model_id,
            ignore_patterns=patterns,
            local_files_only=True,
        )
        if cached_result and Path(cached_result).is_dir():
            console.print(f"[green]✓[/green] Model snapshot already cached locally at {cached_result}")
            return Path(cached_result)
    except Exception:
        pass

    aria2c_bin = find_aria2c()
    if not aria2c_bin:
        console.print(f"[bold blue]Downloading {model_id} via standard HuggingFace Hub...[/bold blue]")
        result = snapshot_download(model_id, ignore_patterns=patterns)
        return Path(result)

    # Step 3: Use aria2c high-speed parallel downloader
    console.print(f"[bold green]⚡ Accelerated multi-connection download enabled via aria2c for {model_id}[/bold green]")
    api = HfApi()
    try:
        info = api.model_info(model_id)
        commit_sha = info.sha
        if not commit_sha:
            raise ValueError(f"Could not retrieve commit SHA for {model_id}")
    except Exception as e:
        console.print(f"[yellow]Failed to fetch model info from HuggingFace API: {e}. Falling back to standard snapshot_download.[/yellow]")
        result = snapshot_download(model_id, ignore_patterns=patterns)
        return Path(result)

    repo_dir_name = f"models--{model_id.replace('/', '--')}"
    hub_cache_dir = Path.home() / ".cache" / "huggingface" / "hub" / repo_dir_name
    snapshot_dir = hub_cache_dir / "snapshots" / commit_sha
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    # First, download metadata and small files (<10MB) via snapshot_download to ensure
    # tokenizer, configs, and helper scripts are cleanly synchronized
    weight_extensions = ("*.safetensors", "*.bin", "*.pt", "*.onnx", "*.pth", "*.ckpt")
    combined_small_file_ignores = list(patterns) + list(weight_extensions)
    try:
        snapshot_download(
            model_id,
            ignore_patterns=combined_small_file_ignores,
        )
    except Exception as e:
        console.print(f"[yellow]Metadata sync notice: {e}[/yellow]")

    # Identify large weight files to download via aria2c
    hf_token = get_hf_token()
    tree = api.list_repo_tree(model_id, revision=commit_sha)
    large_files: list[str] = []

    for item in tree:
        if not hasattr(item, "rfilename") or not item.rfilename:
            continue
        rfilename = item.rfilename

        # Skip ignored patterns
        if any(fnmatch.fnmatch(rfilename, pat) for pat in patterns):
            continue

        # Check if file matches weight extensions or is large (>10MB)
        size = getattr(item, "size", 0) or 0
        is_weight = any(fnmatch.fnmatch(rfilename, ext) for ext in weight_extensions)
        if is_weight or size > 10 * 1024 * 1024:
            dest_file = snapshot_dir / rfilename
            dest_aria2 = dest_file.with_name(dest_file.name + ".aria2")
            # If file exists and size matches and no .aria2 file exists, skip
            if dest_file.exists() and not dest_aria2.exists() and (size == 0 or dest_file.stat().st_size == size):
                continue
            large_files.append(rfilename)

    if not large_files:
        console.print(f"[green]✓[/green] All weights for {model_id} already present in {snapshot_dir}")
        return snapshot_dir

    console.print(f"📥 Downloading {len(large_files)} weight shard(s) with {max_connections_per_server} connections per file...")

    # Write aria2c batch file
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        batch_file = f.name
        for rfilename in large_files:
            url = f"https://huggingface.co/{model_id}/resolve/{commit_sha}/{rfilename}"
            f.write(f"{url}\n")
            f.write(f"  out={rfilename}\n")
            if hf_token:
                f.write(f"  header=Authorization: Bearer {hf_token}\n")

    cmd = [
        aria2c_bin,
        "-c",
        f"-s{split_pieces}",
        f"-x{max_connections_per_server}",
        "-k1M",
        f"-j{max_concurrent_downloads}",
        "-i", batch_file,
        f"--dir={snapshot_dir}",
        "--summary-interval=10",
        "--auto-file-renaming=false",
    ]

    try:
        subprocess.run(cmd, check=True)
    finally:
        if os.path.exists(batch_file):
            try:
                os.unlink(batch_file)
            except Exception:
                pass

    return snapshot_dir


def prune_source_cache(model_id: str) -> bool:
    """Safely delete unquantized source model checkpoint from Hugging Face hub cache.

    Args:
        model_id: Hugging Face model ID (e.g. 'nvidia/LocateAnything-3B').

    Returns:
        bool: True if cache directory was found and deleted, False otherwise.
    """
    sanitized = model_id.replace("/", "--")
    cache_dir = Path(os.path.expanduser("~/.cache/huggingface/hub")) / f"models--{sanitized}"
    if cache_dir.exists():
        console.print(f"[yellow]Pruning unquantized source model cache: {cache_dir}...[/yellow]")
        shutil.rmtree(cache_dir, ignore_errors=True)
        console.print("[green]✓ Source model cache pruned successfully.[/green]")
        return True
    return False
