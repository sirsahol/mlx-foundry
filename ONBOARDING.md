# Developer Onboarding Guide

Welcome to `mlx-foundry`. Follow this guide to configure a local development environment on Apple Silicon and run verification checks.

---

## 1. Prerequisites

* macOS with Apple Silicon (M1, M2, M3, or M4).
* Python 3.10 or newer (Python 3.12+ recommended).
* `uv` package manager (`uv --version`).
* Optional: `aria2c` for accelerated Hugging Face downloads and `hf auth login` for Hub publishing.

---

## 2. Environment Setup

```bash
git clone https://github.com/SirSahOl/mlx-foundry.git
cd mlx-foundry
uv venv --python 3.12
uv pip install -e ".[dev]"
```

Optional symlink for global CLI invocation:

```bash
mkdir -p ~/.local/bin
ln -sf "$(pwd)/.venv/bin/mlx-foundry" ~/.local/bin/mlx-foundry
```

---

## 3. Verification Runbook

Run unit tests (offline mocked suite, `$0` external API cost), linter, DSP graph integrity, and Engineering Canon audits:

```bash
.venv/bin/pytest -v
.venv/bin/ruff check src tests
```

---

## 4. Safe Dry-Run Execution

Execute local conversion without publishing to Hugging Face Hub:

```bash
.venv/bin/mlx-foundry pipeline \
    --model Qwen/Qwen3-0.6B \
    --quants 4,8 \
    --skip-publish \
    --no-cleanup
```

See [Architecture Specification](docs/canon/architecture.md) for backend routing and single-slot governor contracts.
