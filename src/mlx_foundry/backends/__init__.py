"""Model conversion and execution backends for MLX Foundry."""

from __future__ import annotations

from typing import Literal

from mlx_foundry.backends.base import BaseBackend
from mlx_foundry.backends.litert import LiteRTBackend
from mlx_foundry.backends.mflux import MfluxBackend
from mlx_foundry.backends.mlx_lm import MLXLMBackend
from mlx_foundry.backends.mlx_vlm import MLXVLMBackend

BackendName = Literal["auto", "mlx_lm", "mflux", "mlx_vlm", "litert"]

__all__ = [
    "BackendName",
    "BaseBackend",
    "LiteRTBackend",
    "MLXLMBackend",
    "MLXVLMBackend",
    "MfluxBackend",
    "detect_backend",
    "get_backend",
]


def detect_backend(model_id: str) -> str:
    """Inspect model characteristics and detect suitable backend name.

    Returns:
        str: "mlx_vlm" for multimodal models, "mflux" for diffusion models, otherwise "mlx_lm"
    """
    if MLXVLMBackend.can_handle(model_id):
        return "mlx_vlm"
    if MfluxBackend.can_handle(model_id):
        return "mflux"
    return "mlx_lm"


def get_backend(model_id: str, backend_name: str | None = None) -> BaseBackend:
    """Instantiate and return the appropriate backend instance.

    Args:
        model_id: HuggingFace model ID or local directory.
        backend_name: Explicit backend name override ('auto', 'mlx_lm', 'mflux', 'mlx_vlm').

    Returns:
        BaseBackend: The initialized backend instance.
    """
    if not backend_name or backend_name == "auto":
        resolved_name = detect_backend(model_id)
    else:
        resolved_name = backend_name.lower().strip()

    if resolved_name in {"mlx_vlm", "mlx-vlm", "vlm"}:
        return MLXVLMBackend()
    elif resolved_name == "mflux":
        return MfluxBackend()
    elif resolved_name in {"mlx_lm", "mlx-lm", "llm"}:
        return MLXLMBackend()
    elif resolved_name in {"litert", "lite_rt", "litertlm", "tflite"}:
        return LiteRTBackend()
    else:
        raise ValueError(
            f"Unsupported backend: '{resolved_name}'. Valid backends are: 'auto', 'mlx_lm', 'mflux', 'mlx_vlm', 'litert'."
        )
