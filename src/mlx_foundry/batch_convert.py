#!/usr/bin/env python3
"""Batch model conversion script for MLX Foundry (Tier 1 and Tier 2)."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

# Disable experimental HF XET client on macOS to prevent CAS decoding errors
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

# Ensure src is in sys.path
SRC_DIR = Path(__file__).resolve().parent.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from rich.console import Console
from rich.panel import Panel

from mlx_foundry.convert import check_architecture_support
from mlx_foundry.pipeline import run_pipeline
from mlx_foundry.utils import get_disk_free_gb

console = Console()

TIER_1_MODELS = [
    "Qwen/Qwen3-0.6B",
    "IFM/K2-Horizon-0.9B",
    "Qwen/Qwen2.5-1.5B-Instruct",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    "stabilityai/stablelm-2-1_6b",
    "openbmb/MiniCPM5-2B",
    "Qwen/Qwen2.5-3B-Instruct",
    "HuggingFaceTB/SmolLM3-3B",
    "microsoft/Phi-4-mini-instruct",
]

TIER_2_MODELS = [
    "Qwen/Qwen2.5-7B-Instruct",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    "IFM/K2-Horizon-7B",
    "TokenRhythm/NeoHorse-1-9B",
]

COLLECTION_SLUG = (
    "SirSahOl/mlx-models-by-sirsahol-optimized-for-apple-silicon-6aa2b239913bcab23b1ed59a"
)
AUTHOR = "SirSahOl"
QUANTS = [4, 8, 16]
OUTPUT_DIR = Path("output")
PROGRESS_FILE = Path("batch_progress.json")
MIN_DISK_GB = 8.0


def cleanup_model_cache(model_id: str, cache_dir: Path | str | None = None) -> bool:
    """Purge cached HuggingFace snapshots for a given model to free disk space.

    Reclaims ~15-20GB disk space after conversion completes or fails.

    Args:
        model_id: Hugging Face model ID (e.g., 'IFM/K2-Horizon-7B').
        cache_dir: Optional custom Hugging Face hub cache directory.

    Returns:
        bool: True if any cached snapshot directory or revision was purged.
    """
    cleaned = False
    base_cache_dir: Path
    if cache_dir is not None:
        base_cache_dir = Path(cache_dir)
    else:
        env_cache = os.environ.get("HF_HUB_CACHE")
        if env_cache:
            base_cache_dir = Path(env_cache)
        else:
            hf_home = os.environ.get("HF_HOME")
            if hf_home:
                base_cache_dir = Path(hf_home) / "hub"
            else:
                base_cache_dir = Path.home() / ".cache" / "huggingface" / "hub"

    # 1. Attempt scan_cache_dir from huggingface_hub
    try:
        from huggingface_hub import scan_cache_dir

        if base_cache_dir.exists():
            report = scan_cache_dir(cache_dir=base_cache_dir)
            matching_repos = [repo for repo in report.repos if repo.repo_id == model_id]
            if matching_repos:
                revisions = [rev.commit_hash for repo in matching_repos for rev in repo.revisions]
                if revisions:
                    delete_strategy = report.delete_revisions(*revisions)
                    delete_strategy.execute()
                    cleaned = True
                    console.print(
                        f"[green]✓[/green] Purged cache for {model_id} via scan_cache_dir"
                    )
    except Exception as e:  # noqa: BLE001
        console.print(f"[dim]Notice: scan_cache_dir cleanup skipped ({e})[/dim]")

    # 2. Direct removal of models--* directory
    try:
        safe_name = f"models--{model_id.replace('/', '--')}"
        target_dir = base_cache_dir / safe_name
        if target_dir.exists():
            shutil.rmtree(target_dir, ignore_errors=True)
            cleaned = True
            console.print(f"[green]✓[/green] Purged cached snapshot directory: {target_dir}")
    except Exception as e:  # noqa: BLE001
        console.print(
            f"[yellow]Warning: Could not remove cache directory for {model_id}: {e}[/yellow]"
        )

    return cleaned


def load_progress(progress_file: Path = PROGRESS_FILE) -> dict[str, Any]:
    """Load batch execution progress from json file."""
    if progress_file.exists():
        try:
            return json.loads(progress_file.read_text())
        except (json.JSONDecodeError, OSError):
            pass
    return {"completed": [], "failed": [], "skipped": []}


def save_progress(progress: dict[str, Any], progress_file: Path = PROGRESS_FILE) -> None:
    """Save batch execution progress to json file."""
    progress_file.write_text(json.dumps(progress, indent=2) + "\n")


def run_batch(
    models: list[Any],
    dry_run: bool = False,
    start_from: int = 0,
    progress_file: Path = PROGRESS_FILE,
    quants: list[int] | None = None,
    skip_benchmark: bool = False,
    skip_publish: bool = False,
    cache_dir: Path | str | None = None,
) -> dict[str, Any]:
    """Run batch conversion for a list of models with post-conversion cache cleanup."""
    progress = load_progress(progress_file)
    active_quants = quants or QUANTS

    console.print(
        Panel.fit(
            f"[bold cyan]MLX Foundry Batch Conversion[/bold cyan]\n"
            f"Total Models: {len(models)} | Quants: {active_quants}\n"
            f"Dry Run: {dry_run} | Start Index: {start_from}\n"
            f"Skip Benchmark: {skip_benchmark} | Skip Publish: {skip_publish}\n"
            f"Author: [green]{AUTHOR}[/green]\n"
            f"Collection: [yellow]{COLLECTION_SLUG}[/yellow]",
            title="Batch Runner",
        )
    )

    for i, model_entry in enumerate(models):
        if i < start_from:
            continue

        model_id = model_entry if isinstance(model_entry, str) else model_entry.get("model", "")
        model_quants = (
            model_entry.get("quants", active_quants)
            if isinstance(model_entry, dict)
            else active_quants
        )
        progress_key = (
            f"{model_id}:{','.join(map(str, model_quants))}"
            if model_quants != active_quants
            else model_id
        )

        if progress_key in progress["completed"]:
            console.print(f"[dim]Skipping completed: {progress_key}[/dim]")
            continue

        free_gb = get_disk_free_gb(Path.home())
        if free_gb < MIN_DISK_GB and not dry_run:
            msg = f"Low disk space: {free_gb:.1f}GB < {MIN_DISK_GB}GB required"
            console.print(f"[bold red]HALTING:[/bold red] {msg}")
            progress["skipped"].append({"model": progress_key, "reason": msg})
            save_progress(progress, progress_file)
            break

        console.print(
            f"\n[bold yellow]Processing [{i + 1}/{len(models)}]: {model_id}[/bold yellow]"
        )

        if not dry_run:
            arch_ok, arch_name = check_architecture_support(model_id)
            if not arch_ok:
                console.print(f"[yellow]Skipping unsupported architecture: {arch_name}[/yellow]")
                progress["skipped"].append(
                    {"model": progress_key, "reason": f"Unsupported arch: {arch_name}"}
                )
                save_progress(progress, progress_file)
                continue

        t0 = time.time()
        try:
            if dry_run:
                console.print(f"[cyan][DRY RUN] Would convert {model_id} to {model_quants}[/cyan]")
            else:
                run_pipeline(
                    model_id=model_id,
                    author=AUTHOR,
                    quants=model_quants,
                    output_dir=OUTPUT_DIR,
                    skip_benchmark=skip_benchmark,
                    skip_publish=skip_publish,
                    skip_social=True,
                    cleanup=True,
                    private=False,
                    force=False,
                    collection_slug=COLLECTION_SLUG,
                )

            elapsed = time.time() - t0
            progress["completed"].append(progress_key)
            save_progress(progress, progress_file)
            console.print(f"[green]✓ Completed {progress_key} in {elapsed:.1f}s[/green]")

        except Exception as e:  # noqa: BLE001
            elapsed = time.time() - t0
            err_msg = str(e)
            console.print(f"[red]✗ Failed {progress_key} after {elapsed:.1f}s: {err_msg}[/red]")
            progress["failed"].append(
                {
                    "model": progress_key,
                    "reason": err_msg,
                    "error": err_msg,
                    "elapsed_seconds": round(elapsed, 1),
                }
            )
            save_progress(progress, progress_file)
        finally:
            if not dry_run:
                console.print(f"▸ Purging snapshot cache for {model_id} to reclaim disk space...")
                cleanup_model_cache(model_id, cache_dir=cache_dir)

    return progress


def main() -> None:
    """Entry point for CLI batch conversion."""
    parser = argparse.ArgumentParser(description="Batch model conversion for MLX Foundry")
    parser.add_argument("--dry-run", action="store_true", help="Preview actions without converting")
    parser.add_argument("--start-from", type=int, default=0, help="Index to start from")
    parser.add_argument("--tier2", action="store_true", help="Run Tier 2 models (7B-9B)")
    parser.add_argument(
        "--quants", type=int, nargs="+", default=[4, 8, 16], help="Quantization bit levels"
    )
    parser.add_argument(
        "--models", nargs="+", default=None, help="Explicit list of Hugging Face model IDs to run"
    )
    parser.add_argument(
        "--skip-benchmark", action="store_true", help="Skip local performance benchmarking"
    )
    parser.add_argument(
        "--skip-publish", action="store_true", help="Skip Hugging Face Hub publication"
    )
    args = parser.parse_args()

    if args.models:
        target_models = args.models
    else:
        target_models = TIER_2_MODELS if args.tier2 else TIER_1_MODELS

    run_batch(
        models=target_models,
        dry_run=args.dry_run,
        start_from=args.start_from,
        quants=args.quants,
        skip_benchmark=args.skip_benchmark,
        skip_publish=args.skip_publish,
    )


if __name__ == "__main__":
    main()
