"""Unit tests for MLX Foundry multi-backend architecture."""

from pathlib import Path

import pytest

from mlx_foundry.backends import (
    MfluxBackend,
    MLXLMBackend,
    detect_backend,
    get_backend,
)


def test_detect_backend_diffusion_names():
    assert detect_backend("Qwen/Qwen-Image-2.1") == "mflux"
    assert detect_backend("Qwen/Qwen-Image-2512") == "mflux"
    assert detect_backend("black-forest-labs/FLUX.1-schnell") == "mflux"
    assert detect_backend("stabilityai/stable-diffusion-3-medium") == "mflux"


def test_detect_backend_llm_names():
    assert detect_backend("Qwen/Qwen2.5-7B-Instruct") == "mlx_lm"
    assert detect_backend("meta-llama/Llama-3-8B") == "mlx_lm"
    assert detect_backend("mistralai/Mistral-7B-v0.1") == "mlx_lm"


def test_get_backend_instances():
    llm_backend = get_backend("Qwen/Qwen2.5-7B-Instruct")
    assert isinstance(llm_backend, MLXLMBackend)
    assert llm_backend.name == "mlx_lm"

    diff_backend = get_backend("Qwen/Qwen-Image-2.1")
    assert isinstance(diff_backend, MfluxBackend)
    assert diff_backend.name == "mflux"


def test_get_backend_explicit_override():
    backend = get_backend("Qwen/Qwen-Image-2.1", backend_name="mlx_lm")
    assert isinstance(backend, MLXLMBackend)

    backend = get_backend("Qwen/Qwen2.5-7B", backend_name="mflux")
    assert isinstance(backend, MfluxBackend)


def test_get_backend_invalid():
    with pytest.raises(ValueError, match="Unsupported backend"):
        get_backend("Qwen/Qwen2.5-7B", backend_name="unknown_backend")


def test_mflux_backend_convert(tmp_path: Path):
    backend = MfluxBackend()
    output_dir = tmp_path / "output"

    results = backend.convert(
        model_id="Qwen/Qwen-Image-2512",
        quants=[4, 8],
        output_dir=output_dir,
        force=True,
    )

    assert len(results) == 2
    assert results[0].quant_bits == 4
    assert results[1].quant_bits == 8

    # Verify metadata JSON exists
    for r in results:
        meta_file = r.output_path / "conversion_metadata.json"
        assert meta_file.exists()


def test_mflux_backend_benchmark(tmp_path: Path):
    backend = MfluxBackend()
    model_dir = tmp_path / "Qwen-Image-2512-mlx-4bit"
    model_dir.mkdir(parents=True)

    result = backend.benchmark(model_dir, runs=3)
    assert result.num_runs == 3
    assert result.tokens_per_second > 0
    assert result.peak_memory_mb > 0
