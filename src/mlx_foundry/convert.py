"""Model conversion module — converts HuggingFace models to MLX format."""

import json
import shutil
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


def check_architecture_support(model_id: str) -> tuple[bool, str]:
    """Check if the model's architecture is supported by mlx-lm or local ports."""
    try:
        model_path = Path(model_id)
        if model_path.is_dir() and (model_path / "config.json").exists():
            config_path = model_path / "config.json"
        else:
            from huggingface_hub import hf_hub_download

            config_path = hf_hub_download(repo_id=model_id, filename="config.json")
        with open(config_path) as f:
            cfg = json.load(f)
        arch = cfg.get("architectures", [""])[0]
        model_type = cfg.get("model_type", "")

        known_supported = {
            "qwen2",
            "qwen",
            "llama",
            "mistral",
            "phi",
            "phi3",
            "phi4",
            "gemma",
            "gemma2",
            "minicpm",
            "minicpm3",
            "stablelm",
            "k2_horizon",
            "k2-horizon",
            "k2horizon",
            "deepseek",
            "cohere",
            "starcoder2",
            "qwen3_moe",
            "qwen3_5",
            "qwen3_5_text",
        }
        model_type_clean = model_type.lower().strip()
        arch_clean = arch.lower().strip()
        if (
            model_type_clean in known_supported
            or any(k in arch_clean for k in known_supported)
            or any(k.replace("_", "") in arch_clean.replace("_", "") for k in known_supported)
        ):
            return True, model_type or arch
        return False, model_type or arch
    except Exception:  # noqa: BLE001
        return True, "unknown"


def ensure_safetensors_symlinks(snapshot_path: Path) -> list[Path]:
    """Scan snapshot directory and symlink pytorch_model-*.safetensors to model-*.safetensors if needed.

    Args:
        snapshot_path: Path to the downloaded model snapshot directory.

    Returns:
        list[Path]: List of newly created symlink paths.
    """
    created: list[Path] = []
    if not snapshot_path.is_dir():
        return created

    pytorch_safetensors = sorted(snapshot_path.glob("pytorch_model-*.safetensors"))
    if not pytorch_safetensors and (snapshot_path / "pytorch_model.safetensors").exists():
        pytorch_safetensors = [snapshot_path / "pytorch_model.safetensors"]

    existing_model_safetensors = list(snapshot_path.glob("model-*.safetensors"))
    if not existing_model_safetensors and (snapshot_path / "model.safetensors").exists():
        existing_model_safetensors = [snapshot_path / "model.safetensors"]

    if pytorch_safetensors and not existing_model_safetensors:
        for pt_file in pytorch_safetensors:
            if pt_file.name.startswith("pytorch_model-"):
                target_name = pt_file.name.replace("pytorch_model-", "model-", 1)
            elif pt_file.name == "pytorch_model.safetensors":
                target_name = "model.safetensors"
            else:
                target_name = pt_file.name.replace("pytorch_model", "model", 1)

            symlink_path = snapshot_path / target_name
            if not symlink_path.exists():
                try:
                    symlink_path.symlink_to(pt_file.name)
                    created.append(symlink_path)
                    console.print(
                        f"  [green]✓[/green] Created weight symlink: {target_name} -> {pt_file.name}"
                    )
                except OSError as e:
                    console.print(
                        f"  [yellow]Warning: Could not create symlink {target_name}: {e}[/yellow]"
                    )

        pt_index = snapshot_path / "pytorch_model.safetensors.index.json"
        model_index = snapshot_path / "model.safetensors.index.json"
        if pt_index.exists() and not model_index.exists():
            try:
                model_index.symlink_to(pt_index.name)
                created.append(model_index)
                console.print(
                    f"  [green]✓[/green] Created index symlink: {model_index.name} -> {pt_index.name}"
                )
            except OSError:
                pass

    return created


def convert_model(
    model_id: str,
    quants: list[int] = DEFAULT_QUANTS,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    force: bool = False,
    backend: str = "auto",
) -> list[ConversionResult]:
    """Convert a HuggingFace model to MLX format at multiple quantization levels.

    This function:
    1. Resolves backend (auto-detected or explicit)
    2. Dispatches to diffusion backend (mflux) or LLM backend (mlx-lm)
    3. Returns a list of ConversionResult objects
    """
    from mlx_foundry.backends import get_backend

    resolved_backend = get_backend(model_id, backend)
    if resolved_backend.name == "mflux":
        return resolved_backend.convert(
            model_id=model_id, quants=quants, output_dir=output_dir, force=force
        )

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

    hf_path_arg = model_id
    try:
        from huggingface_hub import snapshot_download

        console.print(f"  Ensuring complete repository snapshot for {model_id}...")
        snapshot_result = snapshot_download(model_id)
        if snapshot_result:
            snapshot_path = Path(snapshot_result)
            if snapshot_path.is_dir():
                ensure_safetensors_symlinks(snapshot_path)
                hf_path_arg = str(snapshot_path)
    except Exception as e:  # noqa: BLE001
        console.print(f"[yellow]Warning: Pre-download snapshot check skipped: {e}[/yellow]")

    for quant in sorted(quants):
        quant_label = f"{quant}bit"
        output_path = output_dir / f"{model_name}-mlx-{quant_label}"

        if output_path.exists():
            if not force:
                console.print(
                    f"[yellow]⚠ Skipping {quant_label}: "
                    f"output already exists at {output_path}. Use --force to overwrite.[/yellow]"
                )
                continue
            shutil.rmtree(output_path)

        console.print(f"\n[bold blue]Converting {model_id} to {quant_label}...[/bold blue]")

        start_time = time.time()

        # Build the mlx_lm.convert command
        # For 16-bit, we don't pass -q (no quantization)
        cmd = [
            sys.executable,
            "-m",
            "mlx_lm.convert",
            "--hf-path",
            hf_path_arg,
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
            "command": " ".join(["python3"] + cmd[1:]),
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
