"""Utility functions for MLX Foundry."""

import platform
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def get_chip_name() -> str:
    """Get the Apple Silicon chip name (e.g., 'Apple M1', 'Apple M2 Pro').

    Uses system_profiler on macOS to extract the chip name.
    Falls back to platform.processor() if system_profiler fails.

    Returns:
        str: The chip name.
    """
    try:
        result = subprocess.run(
            ["system_profiler", "SPHardwareDataType"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        for line in result.stdout.splitlines():
            if "Chip:" in line or "Processor Name:" in line:
                return line.split(":")[-1].strip()
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        pass
    return platform.processor() or "Apple Silicon"


def get_system_memory_gb() -> int:
    """Get total system memory in GB.

    Returns:
        int: Total memory in GB.
    """
    import psutil

    return round(psutil.virtual_memory().total / (1024**3))


def get_mlx_lm_version() -> str:
    """Get the installed mlx-lm version.

    Returns:
        str: Version string, e.g., '0.21.0'.
    """
    try:
        import mlx_lm

        return getattr(mlx_lm, "__version__", "unknown")
    except (ImportError, AttributeError):
        return "unknown"


def get_disk_free_gb(path: Path | None = None) -> float:
    """Get free disk space in GB at the given path.

    Args:
        path: Path to check disk space for. Defaults to home directory.

    Returns:
        float: Free disk space in GB.
    """
    target = Path.home() if path is None else Path(path)
    if not target.exists():
        target = target.parent if target.parent.exists() else Path.home()
    usage = shutil.disk_usage(target)
    return round(usage.free / (1024**3), 1)


def dir_size_bytes(path: Path) -> int:
    """Calculate total size of a directory in bytes.

    Args:
        path: Directory path.

    Returns:
        int: Total size in bytes.
    """
    total = 0
    if not path.exists():
        return 0
    for f in path.rglob("*"):
        try:
            if f.is_file():
                total += f.stat().st_size
        except (OSError, PermissionError):
            continue
    return total


def format_size(size_bytes: float) -> str:
    """Format bytes to human-readable string (e.g., '2.3 GB').

    Args:
        size_bytes: Size in bytes.

    Returns:
        str: Human-readable size string.
    """
    size = float(size_bytes)
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size < 1024.0:
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} PB"


def now_iso() -> str:
    """Get current UTC time in ISO 8601 format.

    Returns:
        str: ISO 8601 timestamp.
    """
    return datetime.now(timezone.utc).isoformat()


def fetch_model_info(model_id: str) -> dict[str, Any]:
    """Fetch model metadata from HuggingFace Hub API.

    Uses huggingface_hub library to get model card data, tags, license, etc.

    Args:
        model_id: HuggingFace model ID (e.g., 'Qwen/Qwen3-0.6B').

    Returns:
        dict: Model metadata including license, tags, pipeline_tag, etc.
    """
    from huggingface_hub import model_info

    info = model_info(model_id)

    license_str = "unknown"
    if info.card_data:
        if isinstance(info.card_data, dict):
            license_str = info.card_data.get("license", "unknown") or "unknown"
        else:
            license_str = getattr(info.card_data, "license", "unknown") or "unknown"

    return {
        "model_id": info.id,
        "author": info.author or model_id.split("/")[0],
        "license": license_str,
        "pipeline_tag": info.pipeline_tag or "text-generation",
        "tags": info.tags or [],
        "base_model": model_id,
    }
