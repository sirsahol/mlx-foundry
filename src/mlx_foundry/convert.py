"""Model conversion module — converts HuggingFace models to MLX format."""

import json
import subprocess
import sys
import time
from pathlib import Path

from rich.console import Console

from mlx_foundry.config import (
    DEFAULT_OUTPUT_DIR,
    DEFAULT_QUANTS,
    VALID_QUANTS,
    ConversionResult,
)
from mlx_foundry.utils import (
    dir_size_bytes,
    format_size,
    get_disk_free_gb,
    get_mlx_lm_version,
    now_iso,
)

console = Console()


def estimate_required_space_gb(model_id: str, quants: list[int]) -> float:
    """Estimate disk space required for conversion.

    Rough heuristic:
    - Download the original model (full size)
    - Each quantization level produces output roughly:
      - 4-bit: ~25% of original
      - 8-bit: ~50% of original
      - 16-bit: ~100% of original

    Args:
        model_id: HuggingFace model ID.
        quants: List of quantization levels.

    Returns:
        float: Estimated required space in GB.
    """
    total_bytes = 0
    try:
        from huggingface_hub import model_info

        info = model_info(model_id)
        if info.siblings:
            for sibling in info.siblings:
                if getattr(sibling, "size", None):
                    total_bytes += sibling.size
    except Exception:  # noqa: BLE001
        total_bytes = 0

    if total_bytes == 0:
        console.print(
            "[yellow]Warning: Could not determine model size. Using rough estimate.[/yellow]"
        )
        total_bytes = 2 * 1024**3  # 2GB default

    total_gb = total_bytes / (1024**3)

    # Estimate output sizes
    quant_multipliers = {2: 0.15, 3: 0.20, 4: 0.25, 6: 0.375, 8: 0.50, 16: 1.0}
    output_gb = sum(total_gb * quant_multipliers.get(q, 0.5) for q in quants)

    # Total = download + outputs + buffer
    return total_gb + output_gb + 1.0  # 1GB buffer


def validate_quants(quants: list[int]) -> None:
    """Validate quantization levels.

    Args:
        quants: List of quantization levels to validate.

    Raises:
        ValueError: If any quantization level is invalid.
    """
    invalid = set(quants) - VALID_QUANTS
    if invalid:
        raise ValueError(
            f"Invalid quantization levels: {invalid}. Valid options are: {sorted(VALID_QUANTS)}"
        )


def convert_model(
    model_id: str,
    quants: list[int] = DEFAULT_QUANTS,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    force: bool = False,
) -> list[ConversionResult]:
    """Convert a HuggingFace model to MLX format at multiple quantization levels.

    This function:
    1. Validates inputs and checks disk space
    2. For each quantization level:
       a. Runs mlx_lm.convert via subprocess (for isolation)
       b. Records timing and output size
       c. Saves conversion metadata as JSON
    3. Returns a list of ConversionResult objects

    Args:
        model_id: HuggingFace model ID (e.g., 'Qwen/Qwen3-0.6B').
        quants: List of quantization levels (e.g., [4, 8, 16]).
        output_dir: Base output directory. Each quant gets a subdirectory.
        force: If True, overwrite existing output directories.

    Returns:
        list[ConversionResult]: Results for each successful conversion.

    Raises:
        ValueError: If quantization levels are invalid.
        RuntimeError: If disk space is insufficient.
    """
    validate_quants(quants)

    # Extract model short name for output directory naming
    model_name = model_id.split("/")[-1]

    # Check disk space
    required_gb = estimate_required_space_gb(model_id, quants)
    available_gb = get_disk_free_gb(
        output_dir.parent if output_dir.parent.exists() else Path.home()
    )

    if available_gb < required_gb:
        raise RuntimeError(
            f"Insufficient disk space. Need ~{required_gb:.1f}GB, have {available_gb:.1f}GB free."
        )

    console.print(
        f"[green]✓[/green] Disk space check passed: "
        f"need ~{required_gb:.1f}GB, have {available_gb:.1f}GB free"
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    results: list[ConversionResult] = []

    for quant in sorted(quants):
        quant_label = f"{quant}bit"
        output_path = output_dir / f"{model_name}-mlx-{quant_label}"

        if output_path.exists() and not force:
            console.print(
                f"[yellow]⚠ Skipping {quant_label}: "
                f"output already exists at {output_path}. Use --force to overwrite.[/yellow]"
            )
            continue

        console.print(f"\n[bold blue]Converting {model_id} to {quant_label}...[/bold blue]")

        start_time = time.time()

        # Build the mlx_lm.convert command
        # For 16-bit, we don't pass -q (no quantization)
        cmd = [
            sys.executable,
            "-m",
            "mlx_lm.convert",
            "--hf-path",
            model_id,
            "--mlx-path",
            str(output_path),
        ]

        if quant < 16:
            cmd.extend(["-q", "--q-bits", str(quant)])

        # Run the conversion
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=3600,  # 1 hour timeout
                check=False,
            )

            if result.returncode != 0:
                console.print(f"[red]✗ Conversion failed for {quant_label}[/red]")
                console.print(f"[red]stderr: {result.stderr}[/red]")
                continue

        except subprocess.TimeoutExpired:
            console.print(f"[red]✗ Conversion timed out for {quant_label}[/red]")
            continue

        elapsed = time.time() - start_time
        output_size = dir_size_bytes(output_path)

        # Save conversion metadata
        metadata = {
            "source_model": model_id,
            "output_path": str(output_path),
            "quant_bits": quant,
            "mlx_lm_version": get_mlx_lm_version(),
            "conversion_time_seconds": round(elapsed, 2),
            "output_size_bytes": output_size,
            "timestamp": now_iso(),
            "command": " ".join(cmd),
        }

        output_path.mkdir(parents=True, exist_ok=True)
        metadata_path = output_path / "conversion_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        conversion_result = ConversionResult(
            source_model=model_id,
            output_path=output_path,
            quant_bits=quant,
            mlx_lm_version=get_mlx_lm_version(),
            conversion_time_seconds=round(elapsed, 2),
            output_size_bytes=output_size,
            timestamp=now_iso(),
        )
        results.append(conversion_result)

        console.print(
            f"[green]✓[/green] {quant_label} complete in {elapsed:.1f}s "
            f"({format_size(output_size)})"
        )

    return results
