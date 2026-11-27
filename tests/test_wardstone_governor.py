"""Tests for wardstone ThermalResourceGovernor, SlotLease, and cache flush integration."""

from __future__ import annotations

import json
import subprocess
import sys
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest

from mlx_foundry.benchmark import run_benchmark
from mlx_foundry.convert import convert_model


def test_convert_model_acquires_and_releases_slot_lease(tmp_path, monkeypatch):
    """convert_model inspects governor, acquires SlotLease, and releases + flushes on success."""
    wardstone_gov = pytest.importorskip("wardstone.governor")
    lease_path = tmp_path / "slot_epoch"
    monkeypatch.setenv("WARDSTONE_SLOT_LEASE_PATH", str(lease_path))

    events: list[tuple[str, object]] = []
    real_slot_lease_cls = wardstone_gov.SlotLease

    class SpySlotLease(real_slot_lease_cls):
        def acquire(self, owner: str, model: str = "", preempt_sidecar: bool = False):
            events.append(("acquire", owner, model, preempt_sidecar))
            return super().acquire(owner=owner, model=model, preempt_sidecar=preempt_sidecar)

        def release(self, owner: str, pid: int | None = None) -> bool:
            events.append(("release", owner))
            return super().release(owner=owner, pid=pid)

    def spy_flush():
        events.append(("flush",))
        return {"gc_collected": True, "mlx_cleared": False, "torch_cuda_cleared": False}

    with (
        patch("mlx_foundry.convert.SlotLease", SpySlotLease),
        patch("mlx_foundry.convert.flush_accelerator_cache", side_effect=spy_flush),
        patch(
            "mlx_foundry.convert.ThermalResourceGovernor.inspect",
            return_value={
                "profile": "mac",
                "thermal_level": "nominal",
                "free_ram_gb": 4.0,
                "total_ram_gb": 8.0,
                "vram_free_gb": 4.0,
                "immediate_unload_required": False,
                "badge": "MEM 4.0G free | NOMINAL | TTL 45s",
            },
        ) as mock_inspect,
        patch("mlx_foundry.convert.get_disk_free_gb", return_value=100.0),
        patch("mlx_foundry.convert.estimate_required_space_gb", return_value=5.0),
        patch("subprocess.run") as mock_subproc,
    ):
        mock_subproc.return_value = subprocess.CompletedProcess(
            args=["python3"], returncode=0, stdout="", stderr=""
        )
        results = convert_model(
            model_id="test-org/dummy-model",
            quants=[4],
            output_dir=tmp_path / "out",
            force=True,
        )

    assert len(results) == 1
    mock_inspect.assert_called_once()
    assert ("acquire", "mlx-foundry", "test-org/dummy-model", True) in events
    assert ("release", "mlx-foundry") in events
    assert ("flush",) in events
    assert events.index(("acquire", "mlx-foundry", "test-org/dummy-model", True)) < events.index(
        ("release", "mlx-foundry")
    )
    stored = json.loads(lease_path.read_text(encoding="utf-8"))
    assert stored["owner"] == ""
    assert stored["epoch"] == 1


def test_convert_model_releases_lease_on_error(tmp_path, monkeypatch):
    """convert_model always releases SlotLease and flushes cache if conversion fails."""
    pytest.importorskip("wardstone.governor")
    lease_path = tmp_path / "slot_epoch"
    monkeypatch.setenv("WARDSTONE_SLOT_LEASE_PATH", str(lease_path))

    mock_backend = MagicMock()
    mock_backend.name = "mflux"
    mock_backend.convert.side_effect = RuntimeError("conversion boom")

    with (
        patch("mlx_foundry.backends.get_backend", return_value=mock_backend),
        patch(
            "mlx_foundry.convert.ThermalResourceGovernor.inspect",
            return_value={
                "profile": "mac",
                "thermal_level": "nominal",
                "free_ram_gb": 4.0,
                "total_ram_gb": 8.0,
                "vram_free_gb": 4.0,
                "immediate_unload_required": False,
                "badge": "MEM 4.0G free | NOMINAL | TTL 45s",
            },
        ),
        patch("mlx_foundry.convert.flush_accelerator_cache") as mock_flush,
        pytest.raises(RuntimeError, match="conversion boom"),
    ):
        convert_model(model_id="test-org/flux-test", quants=[4], output_dir=tmp_path / "out")

    mock_flush.assert_called_once()
    stored = json.loads(lease_path.read_text(encoding="utf-8"))
    assert stored["owner"] == ""
    assert stored["epoch"] == 1


def test_run_benchmark_acquires_and_releases_slot_lease(tmp_path, monkeypatch):
    """run_benchmark acquires SlotLease before mlx_lm.load and releases + flushes in finally."""
    wardstone_gov = pytest.importorskip("wardstone.governor")
    lease_path = tmp_path / "slot_epoch"
    monkeypatch.setenv("WARDSTONE_SLOT_LEASE_PATH", str(lease_path))

    model_dir = tmp_path / "dummy-mlx-4bit"
    model_dir.mkdir()

    events: list[str] = []
    real_slot_lease_cls = wardstone_gov.SlotLease

    class SpySlotLease(real_slot_lease_cls):
        def acquire(self, owner: str, model: str = "", preempt_sidecar: bool = False):
            assert owner == "mlx-foundry"
            assert model == str(model_dir)
            assert preempt_sidecar is True
            events.append("acquire")
            return super().acquire(owner=owner, model=model, preempt_sidecar=preempt_sidecar)

        def release(self, owner: str, pid: int | None = None) -> bool:
            assert owner == "mlx-foundry"
            events.append("release")
            return super().release(owner=owner, pid=pid)

    fake_mlx_lm = ModuleType("mlx_lm")
    fake_tok = MagicMock()
    fake_tok.encode.side_effect = lambda s: s.split()
    fake_mlx_lm.load = MagicMock(return_value=(MagicMock(), fake_tok))
    fake_mlx_lm.generate = MagicMock(return_value="test prompt one two three four")

    def spy_flush():
        events.append("flush")
        return {"gc_collected": True, "mlx_cleared": False, "torch_cuda_cleared": False}

    with (
        patch.dict(sys.modules, {"mlx_lm": fake_mlx_lm}),
        patch("mlx_foundry.benchmark.SlotLease", SpySlotLease),
        patch("mlx_foundry.benchmark.flush_accelerator_cache", side_effect=spy_flush),
        patch(
            "mlx_foundry.benchmark.ThermalResourceGovernor.inspect",
            return_value={
                "profile": "mac",
                "thermal_level": "nominal",
                "free_ram_gb": 4.0,
                "total_ram_gb": 8.0,
                "vram_free_gb": 4.0,
                "immediate_unload_required": False,
                "badge": "MEM 4.0G free | NOMINAL | TTL 45s",
            },
        ),
    ):
        res = run_benchmark(model_dir, warmup_runs=1, eval_runs=2, max_tokens=16)

    assert res.num_runs == 2
    assert events == ["acquire", "release", "flush"]
    stored = json.loads(lease_path.read_text(encoding="utf-8"))
    assert stored["owner"] == ""


def test_run_benchmark_releases_lease_on_load_error(tmp_path, monkeypatch):
    """run_benchmark releases SlotLease and flushes cache even when model load fails."""
    pytest.importorskip("wardstone.governor")
    lease_path = tmp_path / "slot_epoch"
    monkeypatch.setenv("WARDSTONE_SLOT_LEASE_PATH", str(lease_path))

    model_dir = tmp_path / "broken-mlx-4bit"
    model_dir.mkdir()

    fake_mlx_lm = ModuleType("mlx_lm")
    fake_mlx_lm.load = MagicMock(side_effect=RuntimeError("weights corrupted"))
    fake_mlx_lm.generate = MagicMock()

    with (
        patch.dict(sys.modules, {"mlx_lm": fake_mlx_lm}),
        patch("mlx_foundry.benchmark.flush_accelerator_cache") as mock_flush,
        patch(
            "mlx_foundry.benchmark.ThermalResourceGovernor.inspect",
            return_value={
                "profile": "mac",
                "thermal_level": "nominal",
                "free_ram_gb": 4.0,
                "total_ram_gb": 8.0,
                "vram_free_gb": 4.0,
                "immediate_unload_required": False,
                "badge": "MEM 4.0G free | NOMINAL | TTL 45s",
            },
        ),
        pytest.raises(RuntimeError, match="weights corrupted"),
    ):
        run_benchmark(model_dir)

    mock_flush.assert_called_once()
    stored = json.loads(lease_path.read_text(encoding="utf-8"))
    assert stored["owner"] == ""


def test_run_benchmark_cached_json_skips_slot_lease(tmp_model_dir):
    """Reusing benchmark_results.json should not acquire SlotLease."""
    with patch("mlx_foundry.benchmark.SlotLease") as mock_lease_cls:
        res = run_benchmark(tmp_model_dir)
    assert res.tokens_per_second == 25.5
    mock_lease_cls.assert_not_called()


def test_graceful_fallback_without_wardstone(tmp_path):
    """Conversion and benchmarking work cleanly when optional wardstone is not installed."""
    fake_mlx_lm = ModuleType("mlx_lm")
    fake_tok = MagicMock()
    fake_tok.encode.side_effect = lambda s: s.split()
    fake_mlx_lm.load = MagicMock(return_value=(MagicMock(), fake_tok))
    fake_mlx_lm.generate = MagicMock(return_value="test prompt one two")

    model_dir = tmp_path / "fallback-mlx-4bit"
    model_dir.mkdir()

    with (
        patch("mlx_foundry.convert.SlotLease", None),
        patch("mlx_foundry.convert.ThermalResourceGovernor", None),
        patch("mlx_foundry.convert.get_disk_free_gb", return_value=100.0),
        patch("mlx_foundry.convert.estimate_required_space_gb", return_value=5.0),
        patch("subprocess.run") as mock_subproc,
    ):
        mock_subproc.return_value = subprocess.CompletedProcess(
            args=["python3"], returncode=0, stdout="", stderr=""
        )
        conv_res = convert_model(
            model_id="test-org/fallback-model",
            quants=[4],
            output_dir=tmp_path / "out",
            force=True,
        )
    assert len(conv_res) == 1

    with (
        patch.dict(sys.modules, {"mlx_lm": fake_mlx_lm}),
        patch("mlx_foundry.benchmark.SlotLease", None),
        patch("mlx_foundry.benchmark.ThermalResourceGovernor", None),
    ):
        bench_res = run_benchmark(model_dir, warmup_runs=0, eval_runs=1, max_tokens=8)
    assert bench_res.num_runs == 1
