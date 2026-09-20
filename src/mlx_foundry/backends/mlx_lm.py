"""MLX-LM Backend for Causal Language Models in MLX Foundry."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from mlx_foundry.backends.base import BaseBackend
from mlx_foundry.config import DEFAULT_QUANTS, BenchmarkResult, ConversionResult


class MLXLMBackend(BaseBackend):
    """Backend utilizing Apple's mlx-lm library for autoregressive LLMs."""

    name: str = "mlx_lm"

    @classmethod
    def can_handle(cls, model_id: str) -> bool:
        """Check if the model can be handled by mlx-lm."""
        from mlx_foundry.convert import check_architecture_support

        supported, _ = check_architecture_support(model_id)
        return supported

    def convert(
        self,
        model_id: str,
        quants: list[int] = DEFAULT_QUANTS,
        output_dir: Path | None = None,
        force: bool = False,
        **kwargs: Any,
    ) -> list[ConversionResult]:
        """Convert an LLM to MLX format using mlx_lm.convert."""
        from mlx_foundry.config import DEFAULT_OUTPUT_DIR
        from mlx_foundry.convert import convert_model

        target_dir = output_dir if output_dir is not None else DEFAULT_OUTPUT_DIR
        return convert_model(model_id=model_id, quants=quants, output_dir=target_dir, force=force)

    def benchmark(
        self,
        model_path: Path,
        runs: int = 5,
        **kwargs: Any,
    ) -> BenchmarkResult:
        """Benchmark an MLX language model."""
        from mlx_foundry.benchmark import run_benchmark

        return run_benchmark(model_path=model_path, eval_runs=runs, **kwargs)
