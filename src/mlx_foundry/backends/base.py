"""Base model backend interface for MLX Foundry."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from mlx_foundry.config import BenchmarkResult, ConversionResult


class BaseBackend(ABC):
    """Abstract interface for all model architecture conversion & benchmark backends."""

    name: str = "base"

    @classmethod
    @abstractmethod
    def can_handle(cls, model_id: str) -> bool:
        """Check if this backend can handle the given model ID or local directory."""

    @abstractmethod
    def convert(
        self,
        model_id: str,
        quants: list[int],
        output_dir: Path,
        force: bool = False,
        **kwargs: Any,
    ) -> list[ConversionResult]:
        """Convert a model to MLX format at specified quantization levels."""

    @abstractmethod
    def benchmark(
        self,
        model_path: Path,
        runs: int = 5,
        **kwargs: Any,
    ) -> BenchmarkResult:
        """Benchmark an MLX model's throughput and resource utilization."""
