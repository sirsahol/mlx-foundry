#!/usr/bin/env python3
"""Convert, benchmark, and publish openbmb/MiniCPM5-2B to Hugging Face Hub."""

from pathlib import Path
import sys

# Add src to sys.path
SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from rich.console import Console
from mlx_foundry.pipeline import run_pipeline

console = Console()

MODEL_ID = "openbmb/MiniCPM5-2B"
AUTHOR = "SirSahOl"
COLLECTION_SLUG = "SirSahOl/mlx-models-by-sirsahol-optimized-for-apple-silicon-6aa2b239913bcab23b1ed59a"
QUANTS = [4, 8, 16]


def main():
    console.print(f"[bold blue]Starting full pipeline for {MODEL_ID}...[/bold blue]")
    results = run_pipeline(
        model_id=MODEL_ID,
        author=AUTHOR,
        quants=QUANTS,
        output_dir=Path("output"),
        force=True,
        cleanup=True,
        collection_slug=COLLECTION_SLUG,
    )
    console.print(f"[bold green]Pipeline finished for {MODEL_ID}![/bold green]")
    print(results)


if __name__ == "__main__":
    main()
