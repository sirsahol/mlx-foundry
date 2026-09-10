"""Pipeline orchestrator — runs the full conversion-to-publication workflow."""

from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel

from mlx_foundry.benchmark import run_benchmark
from mlx_foundry.card import generate_model_card
from mlx_foundry.config import DEFAULT_OUTPUT_DIR, DEFAULT_QUANTS
from mlx_foundry.convert import convert_model
from mlx_foundry.publish import publish_model
from mlx_foundry.social import generate_social_posts

console = Console()


def run_pipeline(
    model_id: str,
    author: str = "SirSahOl",
    quants: list[int] = DEFAULT_QUANTS,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    skip_benchmark: bool = False,
    skip_publish: bool = False,
    skip_social: bool = False,
    cleanup: bool = True,
    private: bool = False,
    force: bool = False,
) -> dict[str, Any]:
    """Run the full MLX Foundry pipeline.

    Steps:
    1. Convert model to all requested quantization levels
    2. Benchmark each converted model
    3. Generate model cards with benchmark data
    4. Publish each model to HuggingFace Hub
    5. Generate social media posts

    Args:
        model_id: HuggingFace model ID (e.g., 'Qwen/Qwen3-0.6B').
        author: HuggingFace username for repo naming.
        quants: List of quantization levels.
        output_dir: Base output directory.
        skip_benchmark: Skip benchmarking step.
        skip_publish: Skip publishing step (useful for testing).
        skip_social: Skip social post generation.
        cleanup: Delete local files after successful upload.
        private: Make HF repos private.
        force: Overwrite existing outputs.

    Returns:
        dict: Summary of pipeline results.
    """
    model_name = model_id.split("/")[-1]

    console.print(
        Panel(
            f"[bold]MLX Foundry Pipeline[/bold]\n\n"
            f"Model: {model_id}\n"
            f"Quantizations: {quants}\n"
            f"Author: {author}",
            title="🔥 Starting Pipeline",
            border_style="blue",
        )
    )

    results: dict[str, Any] = {
        "model_id": model_id,
        "conversions": [],
        "benchmarks": [],
        "published": [],
        "social_posts": {},
    }

    # Step 1: Convert
    console.print("\n[bold]═══ Step 1/5: Converting ═══[/bold]")
    conversion_results = convert_model(
        model_id=model_id,
        quants=quants,
        output_dir=output_dir,
        force=force,
    )
    results["conversions"] = [r.__dict__ for r in conversion_results]

    if not conversion_results:
        console.print("[red]No conversions completed. Pipeline stopped.[/red]")
        return results

    # Step 2: Benchmark
    benchmark_results = []
    if not skip_benchmark:
        console.print("\n[bold]═══ Step 2/5: Benchmarking ═══[/bold]")
        for cr in conversion_results:
            try:
                br = run_benchmark(cr.output_path)
                benchmark_results.append(br)
            except Exception as e:  # noqa: BLE001
                console.print(f"[yellow]⚠ Benchmark failed for {cr.output_path}: {e}[/yellow]")
        results["benchmarks"] = [r.__dict__ for r in benchmark_results]
    else:
        console.print("\n[bold]═══ Step 2/5: Benchmarking (SKIPPED) ═══[/bold]")

    # Build cross-reference list of all quant repos
    all_quant_repos = []
    for cr in conversion_results:
        quant_label = f"{cr.quant_bits}bit"
        repo_name = f"{author}/{model_name}-chat-mlx-{quant_label}"
        all_quant_repos.append(
            {
                "repo": repo_name,
                "quant_bits": cr.quant_bits,
                "url": f"https://huggingface.co/{repo_name}",
            }
        )

    # Step 3: Generate Model Cards
    console.print("\n[bold]═══ Step 3/5: Generating Model Cards ═══[/bold]")
    for cr in conversion_results:
        quant_label = f"{cr.quant_bits}bit"
        hf_repo = f"{author}/{model_name}-chat-mlx-{quant_label}"
        try:
            generate_model_card(
                model_path=cr.output_path,
                source_model=model_id,
                hf_repo=hf_repo,
                author=author,
                benchmark_results=benchmark_results,  # All benchmarks for comparison
                all_quant_repos=all_quant_repos,
            )
        except Exception as e:  # noqa: BLE001
            console.print(f"[yellow]⚠ Card generation failed for {quant_label}: {e}[/yellow]")

    # Step 4: Publish
    if not skip_publish:
        console.print("\n[bold]═══ Step 4/5: Publishing ═══[/bold]")
        for cr in conversion_results:
            quant_label = f"{cr.quant_bits}bit"
            hf_repo = f"{author}/{model_name}-chat-mlx-{quant_label}"
            try:
                url = publish_model(
                    model_path=cr.output_path,
                    hf_repo=hf_repo,
                    source_model=model_id,
                    quant_bits=cr.quant_bits,
                    private=private,
                    cleanup=cleanup,
                )
                results["published"].append({"repo": hf_repo, "url": url})
            except Exception as e:  # noqa: BLE001
                console.print(f"[yellow]⚠ Publish failed for {quant_label}: {e}[/yellow]")
    else:
        console.print("\n[bold]═══ Step 4/5: Publishing (SKIPPED) ═══[/bold]")

    # Step 5: Social Posts
    if not skip_social and benchmark_results:
        console.print("\n[bold]═══ Step 5/5: Generating Social Posts ═══[/bold]")
        # Generate social posts for the smallest quant (most practical for edge)
        primary_benchmark = benchmark_results[0] if benchmark_results else None
        primary_repo = all_quant_repos[0] if all_quant_repos else {"repo": "", "quant_bits": 4}

        posts = generate_social_posts(
            model_name=model_name,
            hf_repo=primary_repo["repo"],
            source_model=model_id,
            quant_bits=primary_repo["quant_bits"],
            benchmark=primary_benchmark,
            all_quant_repos=all_quant_repos,
            output_dir=output_dir / "social_posts",
        )
        results["social_posts"] = posts
    else:
        console.print("\n[bold]═══ Step 5/5: Social Posts (SKIPPED) ═══[/bold]")

    # Summary
    console.print(
        Panel(
            f"[bold green]Pipeline Complete![/bold green]\n\n"
            f"Conversions: {len(conversion_results)}\n"
            f"Benchmarks: {len(benchmark_results)}\n"
            f"Published: {len(results['published'])}\n"
            f"Social posts: {'generated' if results['social_posts'] else 'skipped'}",
            title="✅ Summary",
            border_style="green",
        )
    )

    return results
