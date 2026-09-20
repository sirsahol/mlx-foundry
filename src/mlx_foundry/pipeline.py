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


def format_repo_name(
    author: str,
    model_id: str,
    quant_bits: int,
    backend: str = "auto",
) -> str:
    """Format Hugging Face repository ID consistently.

    Avoids duplicate '-chat-chat' suffixes and handles diffusion model naming.
    """
    model_name = model_id.split("/")[-1]
    quant_label = f"{quant_bits}bit"
    if model_name.endswith("-chat"):
        return f"{author}/{model_name}-mlx-{quant_label}"
    elif backend == "mflux" or any(
        h in model_name.lower() for h in ["image", "flux", "diffusion", "dit"]
    ):
        return f"{author}/{model_name}-mlx-{quant_label}"
    else:
        return f"{author}/{model_name}-chat-mlx-{quant_label}"


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
    collection_slug: str | None = None,
    backend: str = "auto",
    **kwargs: Any,
) -> dict[str, Any]:
    """Run the full MLX Foundry pipeline."""
    model_name = model_id.split("/")[-1]

    console.print(
        Panel(
            f"[bold]MLX Foundry Pipeline[/bold]\n\n"
            f"Model: {model_id}\n"
            f"Quantizations: {quants}\n"
            f"Backend: {backend}\n"
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
        backend=backend,
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
        repo_name = format_repo_name(author, model_id, cr.quant_bits, backend)
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
        hf_repo = format_repo_name(author, model_id, cr.quant_bits, backend)
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
            console.print(f"[yellow]⚠ Card generation failed for {cr.quant_bits}bit: {e}[/yellow]")

    # Step 4: Publish
    if not skip_publish:
        console.print("\n[bold]═══ Step 4/5: Publishing ═══[/bold]")
        for cr in conversion_results:
            hf_repo = format_repo_name(author, model_id, cr.quant_bits, backend)
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
                console.print(f"[yellow]⚠ Publish failed for {cr.quant_bits}bit: {e}[/yellow]")

        if collection_slug and results["published"]:
            try:
                from huggingface_hub import HfApi

                hf_api = HfApi()
                for pub in results["published"]:
                    try:
                        hf_api.add_collection_item(
                            collection_slug=collection_slug,
                            item_id=pub["repo"],
                            item_type="model",
                            exists_ok=True,
                        )
                    except Exception:  # noqa: BLE001, S110
                        pass
            except Exception as e:  # noqa: BLE001
                console.print(f"[yellow]⚠ Collection update notice: {e}[/yellow]")
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
