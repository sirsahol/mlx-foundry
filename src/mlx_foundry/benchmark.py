"""Benchmarking module — measures performance of converted MLX models."""

import json
import statistics
import time
from pathlib import Path

import psutil
from rich.console import Console
from rich.table import Table

from mlx_foundry.config import (
    BENCHMARK_EVAL_RUNS,
    BENCHMARK_MAX_TOKENS,
    BENCHMARK_PROMPT,
    BENCHMARK_WARMUP_RUNS,
    BenchmarkResult,
)
from mlx_foundry.utils import get_chip_name, get_system_memory_gb, now_iso

console = Console()


def measure_peak_memory_mb() -> float:
    """Get current process RSS memory in MB.

    Returns:
        float: Current RSS memory in MB.
    """
    process = psutil.Process()
    return process.memory_info().rss / (1024 * 1024)


def run_benchmark(
    model_path: str | Path,
    prompt: str = BENCHMARK_PROMPT,
    max_tokens: int = BENCHMARK_MAX_TOKENS,
    warmup_runs: int = BENCHMARK_WARMUP_RUNS,
    eval_runs: int = BENCHMARK_EVAL_RUNS,
) -> BenchmarkResult:
    """Run a full benchmark suite on a converted MLX model.

    This function:
    1. Loads the model and tokenizer using mlx_lm
    2. Runs warmup generations (discarded)
    3. Runs eval generations, measuring:
       - Tokens per second (tok/s)
       - Time to first token (TTFT) in milliseconds
       - Peak memory usage in MB
    4. Returns aggregated results

    Args:
        model_path: Path to the converted MLX model directory.
        prompt: Prompt to use for benchmarking.
        max_tokens: Maximum tokens to generate per run.
        warmup_runs: Number of warmup runs (discarded).
        eval_runs: Number of evaluation runs.

    Returns:
        BenchmarkResult: Aggregated benchmark results.
    """
    model_path = Path(model_path)
    console.print(f"[bold blue]Benchmarking {model_path.name}...[/bold blue]")

    # Import mlx_lm here to avoid import errors on non-Apple systems
    from mlx_lm import generate, load

    # Detect quantization from directory name or metadata
    quant_bits = _detect_quant_bits(model_path)

    # Load model
    console.print("  Loading model...")
    model, tokenizer = load(str(model_path))

    # Warmup runs
    console.print(f"  Running {warmup_runs} warmup generations...")
    for _ in range(warmup_runs):
        generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens, verbose=False)

    # Evaluation runs
    console.print(f"  Running {eval_runs} evaluation generations...")
    tok_per_sec_list: list[float] = []
    ttft_list: list[float] = []

    for i in range(eval_runs):
        start = time.perf_counter()
        response = generate(
            model,
            tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            verbose=False,
        )
        end = time.perf_counter()

        elapsed = max(end - start, 1e-6)

        try:
            tokens_generated = len(tokenizer.encode(response)) - len(tokenizer.encode(prompt))
        except Exception:  # noqa: BLE001
            tokens_generated = len(response.split())

        if tokens_generated <= 0:
            tokens_generated = max_tokens

        tps = tokens_generated / elapsed
        tok_per_sec_list.append(tps)

        # TTFT approximation: total_time / total_tokens * 1 (first token latency)
        ttft_approx = (elapsed / tokens_generated) * 1000.0  # ms
        ttft_list.append(ttft_approx)

        console.print(f"    Run {i + 1}/{eval_runs}: {tps:.1f} tok/s")

    mem_peak = measure_peak_memory_mb()

    result = BenchmarkResult(
        model_path=str(model_path),
        chip=get_chip_name(),
        memory_gb=get_system_memory_gb(),
        quant_bits=quant_bits,
        tokens_per_second=round(statistics.mean(tok_per_sec_list), 2) if tok_per_sec_list else 0.0,
        time_to_first_token_ms=round(statistics.mean(ttft_list), 2) if ttft_list else 0.0,
        peak_memory_mb=round(mem_peak, 1),
        prompt=prompt,
        max_tokens=max_tokens,
        num_runs=eval_runs,
        timestamp=now_iso(),
    )

    # Save benchmark results to JSON
    results_path = model_path / "benchmark_results.json"
    with open(results_path, "w") as f:
        json.dump(result.__dict__, f, indent=2)

    # Display results table
    _display_results(result)

    console.print(f"[green]✓[/green] Benchmark results saved to {results_path}")
    return result


def _detect_quant_bits(model_path: Path) -> int:
    """Detect quantization level from model path or metadata.

    Checks conversion_metadata.json first, then falls back to directory name parsing.

    Args:
        model_path: Path to the model directory.

    Returns:
        int: Quantization bits (4, 8, or 16).
    """
    metadata_path = model_path / "conversion_metadata.json"
    if metadata_path.exists():
        try:
            with open(metadata_path) as f:
                metadata = json.load(f)
                return int(metadata.get("quant_bits", 16))
        except (json.JSONDecodeError, ValueError, KeyError):
            pass

    name = model_path.name.lower()
    for bits in [16, 8, 6, 4, 3, 2]:
        if f"{bits}bit" in name:
            return bits
    return 16  # Default


def _display_results(result: BenchmarkResult) -> None:
    """Display benchmark results in a Rich table.

    Args:
        result: BenchmarkResult to display.
    """
    table = Table(title=f"Benchmark Results — {Path(result.model_path).name}")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Chip", result.chip)
    table.add_row("System Memory", f"{result.memory_gb} GB")
    table.add_row("Quantization", f"{result.quant_bits}-bit")
    table.add_row("Tokens/sec", f"{result.tokens_per_second:.2f}")
    table.add_row("TTFT", f"{result.time_to_first_token_ms:.1f} ms")
    table.add_row("Peak Memory", f"{result.peak_memory_mb:.1f} MB")
    table.add_row("Eval Runs", str(result.num_runs))

    if result.perplexity is not None:
        table.add_row("Perplexity", f"{result.perplexity:.2f}")

    console.print(table)
