"""Social media post generator — creates copy-paste posts for Twitter, Reddit, and HF blog."""

from pathlib import Path
from typing import Any

from jinja2 import Environment, PackageLoader
from rich.console import Console

from mlx_foundry.config import BenchmarkResult

console = Console()


def generate_social_posts(
    model_name: str,
    hf_repo: str,
    source_model: str,
    quant_bits: int,
    benchmark: BenchmarkResult | None = None,
    all_quant_repos: list[dict[str, Any]] | None = None,
    output_dir: Path | None = None,
) -> dict[str, str]:
    """Generate social media posts for a published model.

    Generates:
    - Twitter/X post (short, with tags and stats)
    - Reddit post (formatted markdown with benchmark table)

    Args:
        model_name: Short model name (e.g., 'Qwen3-0.6B').
        hf_repo: HuggingFace repo ID (e.g., 'SirSahOl/Qwen3-0.6B-chat-mlx-4bit').
        source_model: Original model ID.
        quant_bits: Quantization level.
        benchmark: Optional benchmark results to include.
        all_quant_repos: List of all quant variants for cross-linking.
        output_dir: Directory to save the generated posts. If None, prints to console.

    Returns:
        dict[str, str]: Dict with keys 'twitter', 'reddit' containing the posts.
    """
    env = Environment(
        loader=PackageLoader("mlx_foundry", "templates"),
        keep_trailing_newline=True,
        trim_blocks=True,
        lstrip_blocks=True,
    )

    context = {
        "model_name": model_name,
        "hf_repo": hf_repo,
        "hf_url": f"https://huggingface.co/{hf_repo}",
        "source_model": source_model,
        "source_author": source_model.split("/")[0],
        "quant_bits": quant_bits,
        "benchmark": benchmark.__dict__ if hasattr(benchmark, "__dict__") else benchmark,
        "all_quant_repos": all_quant_repos or [],
    }

    posts: dict[str, str] = {}

    # Twitter
    twitter_template = env.get_template("tweet.txt.j2")
    posts["twitter"] = twitter_template.render(**context)

    # Reddit
    reddit_template = env.get_template("reddit.md.j2")
    posts["reddit"] = reddit_template.render(**context)

    # Save to files if output_dir provided
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        (output_path / "tweet.txt").write_text(posts["twitter"])
        (output_path / "reddit_post.md").write_text(posts["reddit"])

        console.print(f"[green]✓[/green] Social posts saved to {output_path}/")
    else:
        console.print("\n[bold cyan]═══ Twitter/X Post ═══[/bold cyan]")
        console.print(posts["twitter"])
        console.print("\n[bold cyan]═══ Reddit Post ═══[/bold cyan]")
        console.print(posts["reddit"])

    return posts
