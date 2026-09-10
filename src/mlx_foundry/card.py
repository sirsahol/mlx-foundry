"""Model card generator — creates professional README.md files for HuggingFace repos."""

import json
from pathlib import Path
from typing import Any

from jinja2 import Environment, PackageLoader
from rich.console import Console

from mlx_foundry.config import BenchmarkResult
from mlx_foundry.utils import fetch_model_info, format_size

console = Console()


def generate_model_card(
    model_path: Path,
    source_model: str,
    hf_repo: str,
    author: str = "SirSahOl",
    benchmark_results: list[BenchmarkResult] | None = None,
    all_quant_repos: list[dict[str, Any]] | None = None,
) -> str:
    """Generate a professional model card README.md.

    This function:
    1. Loads conversion metadata and benchmark results from the model directory
    2. Fetches source model info from HuggingFace
    3. Renders the Jinja2 template with all data
    4. Writes the README.md to the model directory
    5. Returns the rendered markdown string

    Args:
        model_path: Path to the converted model directory.
        source_model: Original HuggingFace model ID.
        hf_repo: Target HuggingFace repo ID (e.g., 'SirSahOl/Qwen3-0.6B-chat-mlx-4bit').
        author: HuggingFace username.
        benchmark_results: List of BenchmarkResult objects (can include results from
            other quant levels for comparison).
        all_quant_repos: List of dicts with keys 'repo', 'quant_bits' for cross-linking.

    Returns:
        str: The rendered model card markdown.
    """
    model_path = Path(model_path)

    # Load conversion metadata
    conversion_meta: dict[str, Any] = {}
    meta_path = model_path / "conversion_metadata.json"
    if meta_path.exists():
        try:
            with open(meta_path) as f:
                conversion_meta = json.load(f)
        except (json.JSONDecodeError, OSError):
            conversion_meta = {}

    # Load benchmark results from file if not provided
    if benchmark_results is None:
        benchmark_results = []
        bench_path = model_path / "benchmark_results.json"
        if bench_path.exists():
            try:
                with open(bench_path) as f:
                    data = json.load(f)
                    benchmark_results = [BenchmarkResult(**data)]
            except (json.JSONDecodeError, TypeError, OSError):
                benchmark_results = []

    # Fetch source model info
    try:
        source_info = fetch_model_info(source_model)
    except Exception as e:  # noqa: BLE001
        console.print(f"[yellow]Warning: Could not fetch source model info: {e}[/yellow]")
        source_info = {
            "model_id": source_model,
            "author": source_model.split("/")[0],
            "license": "unknown",
            "pipeline_tag": "text-generation",
            "tags": [],
            "base_model": source_model,
        }

    # Determine quant bits from metadata or path
    quant_bits = conversion_meta.get("quant_bits", _detect_quant_from_path(model_path))

    # Build template context
    context = {
        "model_name": model_path.name,
        "hf_repo": hf_repo,
        "source_model": source_model,
        "source_info": source_info,
        "author": author,
        "quant_bits": quant_bits,
        "conversion_meta": conversion_meta,
        "benchmark_results": [
            r.__dict__ if hasattr(r, "__dict__") else r for r in benchmark_results
        ],
        "model_size": format_size(conversion_meta.get("output_size_bytes", 0)),
        "all_quant_repos": all_quant_repos or [],
        "mlx_lm_version": conversion_meta.get("mlx_lm_version", "unknown"),
    }

    # Render template
    env = Environment(
        loader=PackageLoader("mlx_foundry", "templates"),
        keep_trailing_newline=True,
        trim_blocks=False,
        lstrip_blocks=False,
    )
    template = env.get_template("model_card.md.j2")
    rendered = template.render(**context)

    # Write to model directory
    readme_path = model_path / "README.md"
    with open(readme_path, "w") as f:
        f.write(rendered)

    console.print(f"[green]✓[/green] Model card written to {readme_path}")
    return rendered


def _detect_quant_from_path(path: Path) -> int:
    """Detect quant bits from directory name."""
    name = path.name.lower()
    for bits in [16, 8, 6, 4, 3, 2]:
        if f"{bits}bit" in name:
            return bits
    return 16
