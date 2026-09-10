"""Shared configuration and constants for MLX Foundry."""

from dataclasses import dataclass, field
from pathlib import Path

# Default quantization levels to produce
DEFAULT_QUANTS: list[int] = [4, 8, 16]

# Default output directory for conversions
DEFAULT_OUTPUT_DIR: Path = Path("./output")

# Benchmark configuration
BENCHMARK_PROMPT: str = "Write a short story about a robot learning to paint."
BENCHMARK_MAX_TOKENS: int = 256
BENCHMARK_WARMUP_RUNS: int = 2
BENCHMARK_EVAL_RUNS: int = 5

# Supported quantization bits
VALID_QUANTS: set[int] = {2, 3, 4, 6, 8, 16}


@dataclass
class ConversionResult:
    """Result of a single model conversion."""

    source_model: str
    output_path: Path
    quant_bits: int
    mlx_lm_version: str
    conversion_time_seconds: float
    output_size_bytes: int
    timestamp: str  # ISO 8601


@dataclass
class BenchmarkResult:
    """Result of benchmarking a converted model."""

    model_path: str
    chip: str  # e.g., "Apple M1"
    memory_gb: int
    quant_bits: int
    tokens_per_second: float
    time_to_first_token_ms: float
    peak_memory_mb: float
    perplexity: float | None = None  # Optional, requires eval dataset
    prompt: str = ""
    max_tokens: int = 0
    num_runs: int = 0
    timestamp: str = ""  # ISO 8601


@dataclass
class PublishConfig:
    """Configuration for publishing to HuggingFace Hub."""

    hf_repo: str  # e.g., "SirSahOl/Qwen3-0.6B-chat-mlx-4bit"
    model_path: Path
    source_model: str  # Original HF model ID
    quant_bits: int
    library_name: str = "mlx"
    pipeline_tag: str = "text-generation"
    license: str = ""  # Inherited from source model
    tags: list[str] = field(default_factory=list)
    private: bool = False


@dataclass
class ModelInfo:
    """Extracted info about the source model."""

    model_id: str
    model_name: str  # Short name, e.g., "Qwen3-0.6B"
    author: str  # e.g., "Qwen"
    license: str
    pipeline_tag: str
    tags: list[str] = field(default_factory=list)
    base_model: str | None = None
