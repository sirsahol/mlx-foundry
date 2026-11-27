"""Unit tests for MLX Foundry multi-backend architecture."""

from pathlib import Path
from unittest.mock import patch

import pytest

from mlx_foundry.backends import (
    LiteRTBackend,
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


def test_litert_backend_instances():
    backend = get_backend("google/gemma-4-e2b", backend_name="litert")
    assert isinstance(backend, LiteRTBackend)
    assert backend.name == "litert"


def test_litert_backend_convert_requires_compiler(tmp_path: Path):
    backend = LiteRTBackend()
    output_dir = tmp_path / "output"

    with patch("mlx_foundry.backends.litert.importlib.util.find_spec", return_value=None):
        with pytest.raises(RuntimeError, match="ai-edge-torch"):
            backend.convert(
                model_id="google/gemma-4-e2b",
                quants=[4],
                output_dir=output_dir,
                force=True,
            )

    assert not (output_dir / "gemma-4-e2b-litertlm-4bit" / "gemma-4-e2b.litertlm").exists()


def test_litert_backend_convert_writes_manifest_after_compile(tmp_path: Path):
    backend = LiteRTBackend()
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    def fake_run(cmd, **kwargs):
        output_file = Path(cmd[-1].split("=", 1)[1])
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_bytes(b"LITERTLM real compiler output")
        return type("Result", (), {"returncode": 0, "stdout": "", "stderr": ""})()

    with patch("mlx_foundry.backends.litert.importlib.util.find_spec", return_value=object()):
        with patch("mlx_foundry.backends.litert.subprocess.run", side_effect=fake_run):
            results = backend.convert(
                model_id="google/gemma-4-e2b",
                quants=[4],
                output_dir=output_dir,
                force=True,
            )

    assert len(results) == 1
    manifest = results[0].output_path / "litert_manifest.json"
    assert manifest.exists()
    assert '"status": "compiled"' in manifest.read_text()


def test_litert_backend_benchmark_fails_closed(tmp_path: Path):
    backend = LiteRTBackend()
    model_dir = tmp_path / "gemma-4-e2b-litertlm-4bit"
    model_dir.mkdir(parents=True)

    with pytest.raises(NotImplementedError, match="not implemented"):
        backend.benchmark(model_dir, runs=3)
