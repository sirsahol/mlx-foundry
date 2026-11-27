# Contributing to MLX Foundry

Thank you for contributing to `mlx-foundry`.

## Core Engineering Standards

1. **Test-Driven Development (Red-Green-Refactor):**
   Every feature or bugfix requires failing `pytest` assertions written before or alongside implementation changes. Tests must run offline at `$0` external API cost with mocked network/Hugging Face calls and isolated temporary directories (`tmp_path`).
2. **Hardware-Governed Single-Slot Discipline:**
   Model conversion (`convert_model`) and benchmarking (`run_benchmark`) must acquire `wardstone.governor.SlotLease(owner="mlx-foundry", ..., preempt_sidecar=True)`, inspect `ThermalResourceGovernor`, and release the lease plus call `flush_accelerator_cache()` inside `try/finally`.
3. **Zero Fabricated Artifacts:**
   Never write placeholder weights, synthetic `.litertlm` files, or invented throughput/memory numbers. Unsupported or missing compilers (`ai-edge-torch`) must fail closed.
4. **Engineering Canon Compliance:**
   All documentation must contain zero emojis, use strict relative links, and contain zero hardcoded workstation user paths.
5. **CI-Driven Release Lifecycle:**
   All changes flow through pull requests validated by automated checks (`pytest`, `ruff check`, and Canon boundary audits). Direct production publishes from local developer shells are prohibited; PyPI publishing executes exclusively via GitHub Actions (`.github/workflows/publish.yml`). Branch promotion follows `test` -> `dev` -> `main`.

---

## Local Development Workflow

```bash
uv venv --python 3.12
uv pip install -e ".[dev]"
.venv/bin/pytest -v
.venv/bin/ruff check src tests
```

## Documentation Links

* [Architecture Specification](docs/canon/architecture.md)
* [Developer Onboarding](ONBOARDING.md)
* [Security Policy](SECURITY.md)
* [Changelog](CHANGELOG.md)
