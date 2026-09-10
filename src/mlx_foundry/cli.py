"""CLI interface for MLX Foundry using Typer."""

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from mlx_foundry import __version__
from mlx_foundry.config import DEFAULT_OUTPUT_DIR

app = typer.Typer(
    name="mlx-foundry",
    help="Convert, benchmark, and publish HuggingFace models to Apple MLX format.",
    add_completion=False,
    rich_markup_mode="rich",
)
console = Console()


def version_callback(value: bool):
    if value:
        console.print(f"mlx-foundry v{__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: Annotated[
        bool | None,
        typer.Option(
            "--version",
            "-v",
            callback=version_callback,
            is_eager=True,
            help="Show version and exit.",
        ),
    ] = None,
):
    """MLX Foundry — Professional model conversion pipeline for Apple Silicon."""


def parse_quants(quants_str: str) -> list[int]:
    """Parse comma-separated quantization levels.

    Args:
        quants_str: Comma-separated string like '4,8,16'.

    Returns:
        list[int]: Parsed quantization levels.
    """
    return [int(q.strip()) for q in quants_str.split(",")]


@app.command()
def convert(
    model: Annotated[str, typer.Option("--model", "-m", help="HuggingFace model ID to convert.")],
    quants: Annotated[
        str, typer.Option("--quants", "-q", help="Comma-separated quantization levels.")
    ] = "4,8,16",
    output_dir: Annotated[
        Path, typer.Option("--output", "-o", help="Output directory.")
    ] = DEFAULT_OUTPUT_DIR,
    force: Annotated[
        bool, typer.Option("--force", "-f", help="Overwrite existing outputs.")
    ] = False,
):
    """Convert a HuggingFace model to MLX format at multiple quantization levels."""
    from mlx_foundry.convert import convert_model

    quant_list = parse_quants(quants)
    results = convert_model(model_id=model, quants=quant_list, output_dir=output_dir, force=force)

    if results:
        console.print(f"\n[green]✓ Converted {len(results)} model(s) successfully.[/green]")
    else:
        console.print("\n[yellow]No models were converted.[/yellow]")


@app.command()
def benchmark(
    model: Annotated[
        str, typer.Option("--model", "-m", help="Path to converted MLX model directory.")
    ],
    prompt: Annotated[
        str | None, typer.Option("--prompt", "-p", help="Custom benchmark prompt.")
    ] = None,
    max_tokens: Annotated[int, typer.Option("--max-tokens", help="Max tokens to generate.")] = 256,
    runs: Annotated[int, typer.Option("--runs", "-r", help="Number of evaluation runs.")] = 5,
):
    """Benchmark a converted MLX model (tokens/sec, memory, TTFT)."""
    from mlx_foundry.benchmark import run_benchmark
    from mlx_foundry.config import BENCHMARK_PROMPT

    run_benchmark(
        model_path=model,
        prompt=prompt or BENCHMARK_PROMPT,
        max_tokens=max_tokens,
        eval_runs=runs,
    )


@app.command()
def card(
    model: Annotated[
        str, typer.Option("--model", "-m", help="Path to converted MLX model directory.")
    ],
    source: Annotated[str, typer.Option("--source", "-s", help="Original HuggingFace model ID.")],
    repo: Annotated[str, typer.Option("--repo", "-r", help="Target HuggingFace repo ID.")],
    author: Annotated[
        str, typer.Option("--author", "-a", help="HuggingFace username.")
    ] = "SirSahOl",
):
    """Generate a professional model card (README.md) for a converted model."""
    from mlx_foundry.card import generate_model_card

    generate_model_card(
        model_path=Path(model),
        source_model=source,
        hf_repo=repo,
        author=author,
    )


@app.command()
def publish(
    model: Annotated[
        str, typer.Option("--model", "-m", help="Path to converted MLX model directory.")
    ],
    repo: Annotated[str, typer.Option("--repo", "-r", help="Target HuggingFace repo ID.")],
    source: Annotated[str, typer.Option("--source", "-s", help="Original HuggingFace model ID.")],
    quant_bits: Annotated[int, typer.Option("--quant", "-q", help="Quantization level.")],
    private: Annotated[bool, typer.Option("--private", help="Make the repo private.")] = False,
    no_cleanup: Annotated[
        bool, typer.Option("--no-cleanup", help="Don't delete local files after upload.")
    ] = False,
):
    """Publish a converted model to HuggingFace Hub with proper metadata."""
    from mlx_foundry.publish import publish_model

    publish_model(
        model_path=Path(model),
        hf_repo=repo,
        source_model=source,
        quant_bits=quant_bits,
        private=private,
        cleanup=not no_cleanup,
    )


@app.command()
def social(
    model_name: Annotated[str, typer.Option("--model-name", "-n", help="Short model name.")],
    repo: Annotated[str, typer.Option("--repo", "-r", help="HuggingFace repo ID.")],
    source: Annotated[str, typer.Option("--source", "-s", help="Original model ID.")],
    quant_bits: Annotated[int, typer.Option("--quant", "-q", help="Quantization level.")] = 4,
    output_dir: Annotated[
        Path | None, typer.Option("--output", "-o", help="Output directory for posts.")
    ] = None,
):
    """Generate social media posts (Twitter, Reddit) for a published model."""
    from mlx_foundry.social import generate_social_posts

    generate_social_posts(
        model_name=model_name,
        hf_repo=repo,
        source_model=source,
        quant_bits=quant_bits,
        output_dir=output_dir,
    )


@app.command()
def pipeline(
    model: Annotated[str, typer.Option("--model", "-m", help="HuggingFace model ID to convert.")],
    author: Annotated[
        str, typer.Option("--author", "-a", help="HuggingFace username.")
    ] = "SirSahOl",
    quants: Annotated[
        str, typer.Option("--quants", "-q", help="Comma-separated quantization levels.")
    ] = "4,8,16",
    output_dir: Annotated[
        Path, typer.Option("--output", "-o", help="Output directory.")
    ] = DEFAULT_OUTPUT_DIR,
    skip_benchmark: Annotated[
        bool, typer.Option("--skip-benchmark", help="Skip benchmarking.")
    ] = False,
    skip_publish: Annotated[bool, typer.Option("--skip-publish", help="Skip publishing.")] = False,
    skip_social: Annotated[
        bool, typer.Option("--skip-social", help="Skip social post generation.")
    ] = False,
    no_cleanup: Annotated[
        bool, typer.Option("--no-cleanup", help="Keep local files after upload.")
    ] = False,
    private: Annotated[bool, typer.Option("--private", help="Make repos private.")] = False,
    force: Annotated[
        bool, typer.Option("--force", "-f", help="Overwrite existing outputs.")
    ] = False,
):
    """Run the full pipeline: convert → benchmark → card → publish → social."""
    from mlx_foundry.pipeline import run_pipeline

    quant_list = parse_quants(quants)
    run_pipeline(
        model_id=model,
        author=author,
        quants=quant_list,
        output_dir=output_dir,
        skip_benchmark=skip_benchmark,
        skip_publish=skip_publish,
        skip_social=skip_social,
        cleanup=not no_cleanup,
        private=private,
        force=force,
    )
