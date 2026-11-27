"""Google LiteRT-LM (AI Edge) backend for MLX Foundry.

Enables converting HuggingFace checkpoints into Google AI Edge LiteRT FlatBuffers
(.litertlm) alongside Apple Silicon MLX weights for cross-platform deployment.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from rich.console import Console

from mlx_foundry.backends.base import BaseBackend
from mlx_foundry.config import DEFAULT_QUANTS, BenchmarkResult, ConversionResult
from mlx_foundry.utils import dir_size_bytes, now_iso

console = Console()

# Architectures natively supported by LiteRT-LM / MediaPipe GenAI
LITERT_SUPPORTED_ARCHS = {
    "gemma",
    "gemma2",
    "gemma4",
    "gemma_4",
    "llama",
    "falcon",
    "qwen",
    "qwen2",
    "qwen3",
    "phi",
    "stablelm",
}


class LiteRTBackend(BaseBackend):
    """Backend utilizing Google AI Edge LiteRT-LM for mobile & edge FlatBuffers."""

    name: str = "litert"

    @classmethod
    def can_handle(cls, model_id: str) -> bool:
        """Check if the model architecture is compatible with LiteRT-LM."""
        try:
            from huggingface_hub import HfApi

            api = HfApi()
            info = api.model_info(model_id)
            config = getattr(info, "config", {}) or {}
            model_type = config.get("model_type", "").lower()
            if any(arch in model_type for arch in LITERT_SUPPORTED_ARCHS):
                return True
            for tag in info.tags or []:
                if any(arch in tag.lower() for arch in LITERT_SUPPORTED_ARCHS):
                    return True
        except Exception:
            pass

        # Check by name heuristically
        low = model_id.lower()
        return any(k in low for k in ["gemma", "llama", "qwen", "phi", "litert"])

    def convert(
        self,
        model_id: str,
        quants: list[int] = DEFAULT_QUANTS,
        output_dir: Path | None = None,
        force: bool = False,
        **kwargs: Any,
    ) -> list[ConversionResult]:
        """Convert a model to LiteRT FlatBuffer (.litertlm) format."""
        from mlx_foundry.config import DEFAULT_OUTPUT_DIR

        target_dir = output_dir if output_dir is not None else DEFAULT_OUTPUT_DIR
        target_dir.mkdir(parents=True, exist_ok=True)

        model_name = model_id.split("/")[-1]
        results: list[ConversionResult] = []

        console.print(f"[bold cyan]Starting Google LiteRT-LM export for:[/bold cyan] {model_id}")

        for quant in quants:
            quant_label = f"int{quant}" if quant in {4, 8} else "fp16"
            export_folder = target_dir / f"{model_name}-litertlm-{quant}bit"
            output_model_file = export_folder / f"{model_name}.litertlm"

            if export_folder.exists() and not force:
                console.print(
                    f"  [yellow]Skipping {export_folder.name} (already exists, use --force to overwrite)[/yellow]"
                )
                continue

            export_folder.mkdir(parents=True, exist_ok=True)
            t0 = time.time()

            if importlib.util.find_spec("ai_edge_torch") is None:
                raise RuntimeError(
                    "LiteRT-LM conversion requires the optional ai-edge-torch compiler. "
                    "Install it with `pip install ai-edge-torch mediapipe-genai` and retry. "
                    "No placeholder weights were written."
                )

            console.print(f"  [cyan]Compiling {quant_label} LiteRT-LM flatbuffer via ai-edge-torch...[/cyan]")
            cmd = [
                sys.executable,
                "-m",
                "ai_edge_torch.generative.tools.converter",
                f"--model={model_id}",
                f"--quant={quant_label}",
                f"--output_file={output_model_file}",
            ]
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            if res.returncode != 0 or not output_model_file.exists():
                detail = (res.stderr or res.stdout or "").strip()[-500:]
                raise RuntimeError(f"LiteRT-LM conversion failed for {model_id} ({quant_label}): {detail}")

            manifest = {
                "format": "litertlm",
                "status": "compiled",
                "source_model": model_id,
                "artifact": output_model_file.name,
                "quantization": quant_label,
                "precision_bits": quant,
                "backend": "Google AI Edge LiteRT / MediaPipe GenAI",
                "schema_version": "1.0.0",
                "created_at": now_iso(),
                "runtime_requirements": {
                    "delegates": ["GPU (Metal / Vulkan)", "NPU (NNAPI / Hexagon / TPU)", "CPU (XNNPACK)"],
                    "min_sdk_version": "LiteRT 1.0.0+",
                },
            }
            with open(export_folder / "litert_manifest.json", "w", encoding="utf-8") as f:
                json.dump(manifest, f, indent=2)

            elapsed = time.time() - t0
            folder_size = dir_size_bytes(export_folder)

            results.append(
                ConversionResult(
                    source_model=model_id,
                    output_path=export_folder,
                    quant_bits=quant,
                    mlx_lm_version="litert-1.0.0",
                    conversion_time_seconds=elapsed,
                    output_size_bytes=folder_size,
                    timestamp=now_iso(),
                )
            )
            console.print(
                f"  [green]✓[/green] Successfully exported LiteRT package: {export_folder.name} ({elapsed:.1f}s)"
            )

        return results

    def benchmark(
        self,
        model_path: Path,
        runs: int = 5,
        **kwargs: Any,
    ) -> BenchmarkResult:
        """Benchmark a LiteRT FlatBuffer model.

        LiteRT has no stable Python benchmark API yet. Returning invented
        throughput or memory numbers would contaminate model cards, so this
        backend fails closed until a real runtime probe is implemented.
        """
        raise NotImplementedError(
            "LiteRT benchmarking is not implemented. Use the MLX backends for "
            "measured tokens/sec and memory results, or add a LiteRT runtime probe."
        )
