# MLX Foundry

[![PyPI version](https://img.shields.io/badge/pypi-v0.1.0-blue.svg)](https://pypi.org/project/mlx-foundry/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Apple Silicon](https://img.shields.io/badge/Apple%20Silicon-M1%20%7C%20M2%20%7C%20M3%20%7C%20M4-black.svg)]()

> A professional CLI pipeline for converting, benchmarking, and publishing HuggingFace models to Apple MLX format with publication-grade model cards.

---

## Features

- ⚡ **Multi-Quantization in One Run**: Convert any HuggingFace model to 4-bit, 8-bit, and 16-bit MLX format.
- 📊 **Automated Benchmarking**: Profile tokens/second, time to first token (TTFT), and peak memory usage on Apple Silicon unified memory.
- 📝 **Professional Model Cards**: Generate publication-ready HuggingFace model cards featuring reproducible commands, benchmark tables, and hardware guides.
- 🤗 **One-Command Hub Publishing**: Create repositories, assign appropriate tags (`mlx`, `safetensors`, `conversational`, etc.), and upload with automatic cleanup.
- 📣 **Social Post Generation**: Instantly create ready-to-share summaries formatted for X/Twitter and Reddit (`r/LocalLLaMA`).

---

## Requirements

- **macOS** with Apple Silicon (M1/M2/M3/M4)
- **Python >= 3.10**
- `mlx-lm` and HuggingFace credentials (`hf auth login`)

---

## Installation

```bash
pip install mlx-foundry
```

Or install from source for development:

```bash
git clone https://github.com/SirSahOl/mlx-foundry.git
cd mlx-foundry
pip install -e ".[dev]"
```

---

## Quick Start

Run the end-to-end pipeline to convert, benchmark, generate model cards, and publish:

```bash
mlx-foundry pipeline \
    --model Qwen/Qwen3-0.6B \
    --author SirSahOl \
    --quants 4,8,16
```

### Dry Run (Local only, skip publishing)

```bash
mlx-foundry pipeline \
    --model Qwen/Qwen3-0.6B \
    --author SirSahOl \
    --quants 4,8 \
    --skip-publish \
    --no-cleanup
```

---

## Commands Reference

### 1. `pipeline`
Runs the complete workflow: conversion, benchmark, card generation, publish, and social posts.

```bash
mlx-foundry pipeline --model <HF_MODEL_ID> [OPTIONS]
```

Options:
- `-m, --model`: HuggingFace model ID (required).
- `-a, --author`: HuggingFace username/org (default: `SirSahOl`).
- `-q, --quants`: Comma-separated quantizations (default: `4,8,16`).
- `-o, --output`: Base output directory (default: `./output`).
- `--skip-benchmark`: Skip performance benchmarking.
- `--skip-publish`: Skip HuggingFace upload.
- `--skip-social`: Skip social post generation.
- `--no-cleanup`: Keep local converted weights after upload.
- `--private`: Mark HuggingFace repository as private.
- `-f, --force`: Overwrite existing output directories.

### 2. `convert`
Convert models to specified quantization levels.

```bash
mlx-foundry convert --model Qwen/Qwen3-0.6B --quants 4,8,16 --output ./output
```

### 3. `benchmark`
Measure throughput (tokens/sec), first-token latency, and memory footprint.

```bash
mlx-foundry benchmark --model ./output/Qwen3-0.6B-mlx-4bit --runs 5
```

### 4. `card`
Generate a rich `README.md` model card using Jinja2 templates.

```bash
mlx-foundry card \
    --model ./output/Qwen3-0.6B-mlx-4bit \
    --source Qwen/Qwen3-0.6B \
    --repo SirSahOl/Qwen3-0.6B-chat-mlx-4bit
```

### 5. `publish`
Upload a converted model folder to HuggingFace Hub with complete tags and metadata.

```bash
mlx-foundry publish \
    --model ./output/Qwen3-0.6B-mlx-4bit \
    --repo SirSahOl/Qwen3-0.6B-chat-mlx-4bit \
    --source Qwen/Qwen3-0.6B \
    --quant 4
```

### 6. `social`
Generate formatted announcements for Twitter and Reddit.

```bash
mlx-foundry social \
    --model-name Qwen3-0.6B \
    --repo SirSahOl/Qwen3-0.6B-chat-mlx-4bit \
    --source Qwen/Qwen3-0.6B \
    --quant 4
```

---

## Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the [issues page](https://github.com/SirSahOl/mlx-foundry/issues).

---

## License

Distributed under the MIT License. See [LICENSE](LICENSE) for more information.
