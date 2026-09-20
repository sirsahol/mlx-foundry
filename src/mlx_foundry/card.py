"""Model card generator — creates professional README.md files for HuggingFace repos."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from jinja2 import Environment, PackageLoader
from rich.console import Console

from mlx_foundry.config import BenchmarkResult
from mlx_foundry.utils import fetch_model_info, format_size

console = Console()


def generate_model_card(
    model_path: Path,
    source_model: str,
    hf_repo: str,
    author: str = "SirSahOl",
    benchmark_results: list[BenchmarkResult] | None = None,
    all_quant_repos: list[dict[str, Any]] | None = None,
) -> str:
    """Generate a professional model card README.md.

    This function:
    1. Loads conversion metadata and benchmark results from the model directory
    2. Loads model architecture config (config.json) if available
    3. Fetches source model info from HuggingFace
    4. Computes rich architectural details, hardware sizing, and multi-quant matrix
    5. Renders the Jinja2 template with all data
    6. Writes the README.md to the model directory
    7. Returns the rendered markdown string

    Args:
        model_path: Path to the converted model directory.
        source_model: Original HuggingFace model ID.
        hf_repo: Target HuggingFace repo ID (e.g., 'SirSahOl/Qwen2.5-7B-Instruct-chat-mlx-4bit').
        author: HuggingFace username.
        benchmark_results: List of BenchmarkResult objects (can include results from
            other quant levels for comparison).
        all_quant_repos: List of dicts with keys 'repo', 'quant_bits', 'url' for cross-linking.

    Returns:
        str: The rendered model card markdown.
    """
    model_path = Path(model_path)

    # Load conversion metadata
    conversion_meta: dict[str, Any] = {}
    meta_path = model_path / "conversion_metadata.json"
    if meta_path.exists():
        try:
            with open(meta_path) as f:
                conversion_meta = json.load(f)
        except (json.JSONDecodeError, OSError):
            conversion_meta = {}

    # Load model config if present
    config: dict[str, Any] = {}
    config_path = model_path / "config.json"
    if config_path.exists():
        try:
            with open(config_path) as f:
                config = json.load(f)
        except (json.JSONDecodeError, OSError):
            config = {}

    # Load benchmark results from file if not provided
    if benchmark_results is None:
        benchmark_results = []
        bench_path = model_path / "benchmark_results.json"
        if bench_path.exists():
            try:
                with open(bench_path) as f:
                    data = json.load(f)
                    benchmark_results = [BenchmarkResult(**data)]
            except (json.JSONDecodeError, TypeError, OSError):
                benchmark_results = []

    # Fetch source model info
    try:
        source_info = fetch_model_info(source_model)
    except Exception as e:  # noqa: BLE001
        console.print(f"[yellow]Warning: Could not fetch source model info: {e}[/yellow]")
        source_info = {
            "model_id": source_model,
            "author": source_model.split("/")[0],
            "license": "unknown",
            "pipeline_tag": "text-generation",
            "tags": [],
            "base_model": source_model,
        }

    # Determine quant bits from metadata or path
    quant_bits = conversion_meta.get("quant_bits", _detect_quant_from_path(model_path))

    # Architectural and sizing computations
    params_b, param_str = _extract_param_info(model_path, source_model, config, conversion_meta)
    model_details = _get_model_details(
        model_path=model_path,
        source_model=source_model,
        quant_bits=quant_bits,
        config=config,
        conversion_meta=conversion_meta,
        params_b=params_b,
        param_str=param_str,
    )
    hardware_matrix = _get_hardware_sizing_matrix(params_b=params_b, quant_bits=quant_bits)
    quant_comparison = _get_quant_comparison(
        params_b=params_b,
        quant_bits=quant_bits,
        hf_repo=hf_repo,
        all_quant_repos=all_quant_repos,
    )
    chat_config = _get_chat_config(source_model=source_model, hf_repo=hf_repo)

    # Extra tags
    extra_tags: list[str] = []
    source_lower = source_model.lower()
    if "qwen2.5" in source_lower:
        extra_tags.extend(["qwen", "qwen2.5"])
    elif "qwen" in source_lower:
        extra_tags.append("qwen")
    elif "glm" in source_lower:
        extra_tags.extend(["glm", "edge"])
    elif "deepseek" in source_lower:
        extra_tags.extend(["deepseek", "reasoning"])

    for tag in source_info.get("tags", []):
        if (
            tag not in extra_tags
            and tag
            not in ["mlx", "safetensors", "apple-silicon", "conversational", f"{quant_bits}-bit"]
            and not tag.startswith("base_model:")
            and tag != "region:us"
        ):
            extra_tags.append(tag)

    is_diffusion = (
        conversion_meta.get("backend") == "mflux"
        or "diffusers" in source_info.get("tags", [])
        or any(
            h in source_lower or h in model_path.name.lower()
            for h in ["image", "flux", "diffusion", "dit"]
        )
    )

    # Build template context
    context = {
        "model_name": model_path.name,
        "hf_repo": hf_repo,
        "source_model": source_model,
        "source_info": source_info,
        "author": author,
        "quant_bits": quant_bits,
        "is_diffusion": is_diffusion,
        "conversion_meta": conversion_meta,
        "benchmark_results": [
            r.__dict__ if hasattr(r, "__dict__") else r for r in benchmark_results
        ],
        "model_size": format_size(conversion_meta.get("output_size_bytes", 0)),
        "all_quant_repos": all_quant_repos or [],
        "mlx_lm_version": conversion_meta.get("mlx_lm_version", "unknown"),
        "model_details": model_details,
        "hardware_matrix": hardware_matrix,
        "quant_comparison": quant_comparison,
        "chat_config": chat_config,
        "extra_tags": extra_tags,
    }

    # Render template
    env = Environment(
        loader=PackageLoader("mlx_foundry", "templates"),
        keep_trailing_newline=True,
        trim_blocks=True,
        lstrip_blocks=True,
    )
    template = env.get_template("model_card.md.j2")
    rendered = template.render(**context)

    # Write to model directory
    readme_path = model_path / "README.md"
    with open(readme_path, "w") as f:
        f.write(rendered)

    console.print(f"[green]✓[/green] Model card written to {readme_path}")
    return rendered


def _detect_quant_from_path(path: Path) -> int:
    """Detect quant bits from directory name."""
    name = path.name.lower()
    for bits in [16, 8, 6, 4, 3, 2]:
        if f"{bits}bit" in name:
            return bits
    return 16


def _extract_param_info(
    model_path: Path,
    source_model: str,
    config: dict[str, Any],
    conversion_meta: dict[str, Any],
) -> tuple[float, str]:
    """Extract parameter count in billions and display string."""
    source_lower = source_model.lower()
    name_lower = model_path.name.lower()

    if "qwen2.5-7b" in source_lower or "qwen2.5-7b" in name_lower:
        return 7.0, "7.61B (7.07B non-embedding)"
    if "qwen3-0.6b" in source_lower or "qwen3-0.6b" in name_lower:
        return 0.6, "0.59B"
    if "qwen2.5-1.5b" in source_lower or "qwen2.5-1.5b" in name_lower:
        return 1.5, "1.54B"
    if "qwen2.5-3b" in source_lower or "qwen2.5-3b" in name_lower:
        return 3.0, "3.09B"
    if "glm-edge-4b" in source_lower or "glm-edge-4b" in name_lower:
        return 4.0, "4.15B"
    if "glm-edge-1.5b" in source_lower or "glm-edge-1.5b" in name_lower:
        return 1.5, "1.54B"
    if "k2-horizon-0.9b" in source_lower or "k2-horizon-0.9b" in name_lower:
        return 0.9, "0.9B"
    if "neohorse-1-9b" in source_lower or "neohorse-1-9b" in name_lower:
        return 9.0, "9.0B"

    # Regex search for parameter pattern without matching quant suffixes like 4bit
    pattern = re.compile(r"(?<![a-zA-Z0-9.])(\d+(?:\.\d+)?)\s*[bB](?:illion)?(?![a-zA-Z])")
    match = pattern.search(source_model) or pattern.search(model_path.name)
    if match:
        val = float(match.group(1))
        return val, f"{val:g}B"

    # Fallback from output size
    output_size = conversion_meta.get("output_size_bytes", 0)
    if output_size > 0:
        quant_bits = conversion_meta.get("quant_bits", 4)
        bytes_per_param = (quant_bits / 8.0) * 1.15
        est_params_b = round(output_size / (bytes_per_param * (1024**3)), 1)
        if est_params_b > 0.1:
            return est_params_b, f"~{est_params_b:g}B"

    return 7.0, "7B"


def _get_model_details(
    model_path: Path,
    source_model: str,
    quant_bits: int,
    config: dict[str, Any],
    conversion_meta: dict[str, Any],
    params_b: float,
    param_str: str,
) -> dict[str, Any]:
    """Build architectural and hardware detail dictionary."""
    source_lower = source_model.lower()

    # Architecture class
    if config.get("architectures"):
        arch = config["architectures"][0]
    elif config.get("model_type"):
        arch = f"{config['model_type'].capitalize()}ForCausalLM"
    elif "qwen" in source_lower:
        arch = "Qwen2ForCausalLM"
    elif "glm" in source_lower:
        arch = "GLM-Edge"
    else:
        arch = model_path.name.split("-mlx")[0] or "Transformer Causal LM"

    # Context Length
    max_pos = config.get("max_position_embeddings")
    if max_pos:
        if "qwen" in source_lower and max_pos == 32768:
            context_length = "32,768 tokens (native, extensible up to 131,072 with YaRN)"
        else:
            context_length = f"{max_pos:,} tokens"
    else:
        if "qwen" in source_lower:
            context_length = "32,768 tokens (native, up to 131,072 with YaRN / context extension)"
        elif "glm" in source_lower:
            context_length = "8,192 tokens"
        else:
            context_length = "32,768 tokens"

    # Average bits per weight
    if quant_bits == 4:
        avg_bits = "4.50"
    elif quant_bits == 8:
        avg_bits = "8.25"
    elif quant_bits == 16:
        avg_bits = "16.00 (unquantized bfloat16)"
    else:
        avg_bits = f"{quant_bits}.00"

    # VRAM footprint and minimum recommended RAM
    if 6.0 <= params_b <= 8.5:
        vram_map = {4: "4.2 GB", 8: "7.8 GB", 16: "15.2 GB"}
        vram_footprint = vram_map.get(
            quant_bits, f"{round(params_b * (quant_bits / 8.0) * 1.1, 1)} GB"
        )
        min_ram_map = {
            4: "8 GB Unified Memory",
            8: "16 GB Unified Memory",
            16: "24 GB – 32 GB Unified Memory",
        }
        min_ram = min_ram_map.get(quant_bits, "16 GB Unified Memory")
    else:
        weight_gb = round(params_b * (quant_bits / 8.0) * 1.12 + 0.35, 1)
        vram_footprint = f"{weight_gb} GB"
        if weight_gb <= 5.0:
            min_ram = "8 GB Unified Memory"
        elif weight_gb <= 12.0:
            min_ram = "16 GB Unified Memory"
        else:
            min_ram = "32 GB Unified Memory"

    return {
        "architecture": arch,
        "parameters": param_str,
        "context_length": context_length,
        "avg_bits_per_weight": avg_bits,
        "vram_footprint": vram_footprint,
        "min_ram": min_ram,
    }


def _get_hardware_sizing_matrix(params_b: float, quant_bits: int) -> list[dict[str, Any]]:
    """Build estimated hardware sizing and performance matrix for Apple Silicon."""
    if 6.0 <= params_b <= 8.5:
        # Standard 7B-class models
        if quant_bits == 4:
            return [
                {
                    "tier": "M1 / M2 / M3 / M4 (Base)",
                    "ram": "8 GB – 16 GB",
                    "vram": "~4.2 GB",
                    "speed": "~35 tokens/sec",
                    "ttft": "~110 ms",
                    "use_case": "Fast everyday local assistant, coding sidecar, single-agent interactions",
                },
                {
                    "tier": "M1 / M2 / M3 / M4 Pro",
                    "ram": "18 GB – 36 GB",
                    "vram": "~4.2 GB",
                    "speed": "~52 tokens/sec",
                    "ttft": "~75 ms",
                    "use_case": "High-throughput agentic workflows, complex tool use, long contexts",
                },
                {
                    "tier": "M1 / M2 / M3 / M4 Max",
                    "ram": "36 GB – 128 GB",
                    "vram": "~4.2 GB",
                    "speed": "~75 tokens/sec",
                    "ttft": "~45 ms",
                    "use_case": "Ultra-low-latency generation, parallel multi-turn evaluation, RAG pipelines",
                },
                {
                    "tier": "M1 / M2 / M3 Ultra",
                    "ram": "64 GB – 192 GB",
                    "vram": "~4.2 GB",
                    "speed": "~105 tokens/sec",
                    "ttft": "~30 ms",
                    "use_case": "Enterprise local serving, multi-user concurrency, batch document synthesis",
                },
            ]
        if quant_bits == 8:
            return [
                {
                    "tier": "M1 / M2 / M3 / M4 (Base)",
                    "ram": "16 GB (min. required)",
                    "vram": "~7.8 GB",
                    "speed": "~22 tokens/sec",
                    "ttft": "~160 ms",
                    "use_case": "High-precision instruction following on 16GB Apple Silicon Macs",
                },
                {
                    "tier": "M1 / M2 / M3 / M4 Pro",
                    "ram": "18 GB – 36 GB",
                    "vram": "~7.8 GB",
                    "speed": "~34 tokens/sec",
                    "ttft": "~110 ms",
                    "use_case": "Balanced daily driver for complex math, structured reasoning, and code",
                },
                {
                    "tier": "M1 / M2 / M3 / M4 Max",
                    "ram": "36 GB – 128 GB",
                    "vram": "~7.8 GB",
                    "speed": "~48 tokens/sec",
                    "ttft": "~70 ms",
                    "use_case": "Low-latency multi-step reasoning, swarm agents, data analysis",
                },
                {
                    "tier": "M1 / M2 / M3 Ultra",
                    "ram": "64 GB – 192 GB",
                    "vram": "~7.8 GB",
                    "speed": "~72 tokens/sec",
                    "ttft": "~45 ms",
                    "use_case": "High-throughput serving with near-lossless precision",
                },
            ]
        # 16-bit
        return [
            {
                "tier": "M1 / M2 / M3 / M4 (Base)",
                "ram": "24 GB (min. required)",
                "vram": "~15.2 GB",
                "speed": "~12 tokens/sec",
                "ttft": "~240 ms",
                "use_case": "Full-precision unquantized evaluation on 24GB Macs (M2/M3/M4)",
            },
            {
                "tier": "M1 / M2 / M3 / M4 Pro",
                "ram": "36 GB – 48 GB",
                "vram": "~15.2 GB",
                "speed": "~18 tokens/sec",
                "ttft": "~160 ms",
                "use_case": "Development, prompt engineering, and ground-truth model comparison",
            },
            {
                "tier": "M1 / M2 / M3 / M4 Max",
                "ram": "36 GB – 128 GB",
                "vram": "~15.2 GB",
                "speed": "~28 tokens/sec",
                "ttft": "~100 ms",
                "use_case": "Unquantized reference inference with zero perplexity penalty",
            },
            {
                "tier": "M1 / M2 / M3 Ultra",
                "ram": "64 GB – 192 GB",
                "vram": "~15.2 GB",
                "speed": "~42 tokens/sec",
                "ttft": "~65 ms",
                "use_case": "Enterprise workstation deployment, zero-compromise reference serving",
            },
        ]

    # Dynamic scaling for other parameter sizes
    weight_gb = max(0.4, params_b * (quant_bits / 8.0) * 1.12)
    vram_str = f"~{round(weight_gb + 0.4, 1)} GB"
    base_spd = max(12, round(35 * (7.0 / params_b) ** 0.85 * (4.0 / quant_bits) ** 0.6))
    pro_spd = max(18, round(base_spd * 1.5))
    max_spd = max(26, round(base_spd * 2.15))
    ultra_spd = max(38, round(base_spd * 3.0))

    base_ttft = max(25, round(110 * (params_b / 7.0) * (quant_bits / 4.0) ** 0.5))
    pro_ttft = max(18, round(base_ttft * 0.68))
    max_ttft = max(12, round(base_ttft * 0.42))
    ultra_ttft = max(8, round(base_ttft * 0.28))

    min_ram_base = "8 GB" if weight_gb < 5.0 else ("16 GB" if weight_gb < 11.0 else "24 GB+")
    min_ram_pro = "18 GB – 36 GB" if weight_gb < 14.0 else "36 GB – 48 GB"

    return [
        {
            "tier": "M1 / M2 / M3 / M4 (Base)",
            "ram": f"{min_ram_base} Unified Memory",
            "vram": vram_str,
            "speed": f"~{base_spd} tokens/sec",
            "ttft": f"~{base_ttft} ms",
            "use_case": "Everyday interactive assistant & fast local completions",
        },
        {
            "tier": "M1 / M2 / M3 / M4 Pro",
            "ram": min_ram_pro,
            "vram": vram_str,
            "speed": f"~{pro_spd} tokens/sec",
            "ttft": f"~{pro_ttft} ms",
            "use_case": "Balanced daily driver for coding, tool invocation, and multi-turn chat",
        },
        {
            "tier": "M1 / M2 / M3 / M4 Max",
            "ram": "36 GB – 128 GB",
            "vram": vram_str,
            "speed": f"~{max_spd} tokens/sec",
            "ttft": f"~{max_ttft} ms",
            "use_case": "High-throughput generation, low latency, agent orchestration",
        },
        {
            "tier": "M1 / M2 / M3 Ultra",
            "ram": "64 GB – 192 GB",
            "vram": vram_str,
            "speed": f"~{ultra_spd} tokens/sec",
            "ttft": f"~{ultra_ttft} ms",
            "use_case": "Peak concurrency, batch document extraction, production serving",
        },
    ]


def _get_quant_comparison(
    params_b: float,
    quant_bits: int,
    hf_repo: str,
    all_quant_repos: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    """Build multi-quantization comparison table for 4-bit, 8-bit, and 16-bit variants."""
    repo_map: dict[int, str] = {}
    if all_quant_repos:
        for qr in all_quant_repos:
            q = qr.get("quant_bits")
            url = qr.get("url") or f"https://huggingface.co/{qr.get('repo')}"
            if q:
                repo_map[int(q)] = url

    # Infer URLs if missing
    for q in [4, 8, 16]:
        if q not in repo_map:
            sibling_repo = re.sub(r"-\d+bit$", f"-{q}bit", hf_repo)
            if sibling_repo == hf_repo and f"-{quant_bits}bit" not in hf_repo:
                sibling_repo = f"{hf_repo}-{q}bit"
            repo_map[q] = f"https://huggingface.co/{sibling_repo}"

    # Calculate sizes and benefits
    if 6.0 <= params_b <= 8.5:
        sizes = {
            4: {
                "disk": "~4.3 GB",
                "vram": "~4.2 GB",
                "hw": "M1 / M2 / M3 / M4 (8GB+ Unified Memory)",
                "adv": "Maximum generation speed, lowest memory pressure; ideal for multitasking and everyday local chat alongside IDEs.",
            },
            8: {
                "disk": "~8.1 GB",
                "vram": "~7.8 GB",
                "hw": "M1 / M2 / M3 / M4 Pro/Max (16GB+ Unified Memory)",
                "adv": "Near-lossless precision, high-fidelity reasoning, and stable complex instruction following.",
            },
            16: {
                "disk": "~15.2 GB",
                "vram": "~15.2 GB",
                "hw": "M2 / M3 / M4 Max/Ultra (32GB+ Unified Memory)",
                "adv": "Full unquantized bfloat16 precision; zero perplexity loss, ideal for evaluation and reference output.",
            },
        }
    else:
        sizes = {
            4: {
                "disk": f"~{round(params_b * 0.55 + 0.35, 1)} GB",
                "vram": f"~{round(params_b * 0.55 + 0.35, 1)} GB",
                "hw": "M1 / M2 / M3 / M4 (8GB+)",
                "adv": "Maximum generation speed and lowest RAM overhead.",
            },
            8: {
                "disk": f"~{round(params_b * 1.05 + 0.45, 1)} GB",
                "vram": f"~{round(params_b * 1.05 + 0.45, 1)} GB",
                "hw": "M1 / M2 / M3 / M4 Pro/Max (16GB+)",
                "adv": "Balanced accuracy and generation speed; near-lossless reasoning.",
            },
            16: {
                "disk": f"~{round(params_b * 2.0 + 0.8, 1)} GB",
                "vram": f"~{round(params_b * 2.0 + 0.8, 1)} GB",
                "hw": "M2 / M3 / M4 Max/Ultra (32GB+)",
                "adv": "Full unquantized precision; reference evaluation quality.",
            },
        }

    rows = []
    for q in [4, 8, 16]:
        info = sizes[q]
        url = repo_map[q]
        repo_name = url.split("https://huggingface.co/")[-1]
        is_current = q == quant_bits
        if is_current:
            label = f"**{q}-bit MLX** (This Repository)"
        else:
            label = f"**[{q}-bit MLX]({url})**"
        rows.append(
            {
                "bits": q,
                "label": label,
                "repo": repo_name,
                "url": url,
                "disk_size": info["disk"],
                "vram": info["vram"],
                "target_hardware": info["hw"],
                "advantage": info["adv"],
                "is_current": is_current,
            }
        )
    return rows


def _get_chat_config(source_model: str, hf_repo: str) -> dict[str, Any]:
    """Determine stop tokens and prompt template formatting for local inference engines."""
    source_lower = source_model.lower()
    repo_lower = hf_repo.lower()

    if "glm" in source_lower or "glm" in repo_lower:
        return {
            "preset_name": "GLM-Edge",
            "ollama_name": "glm-edge",
            "stop_tokens": ["<|user|>", "<|observation|>", "<|endoftext|>"],
            "user_prefix": "<|user|>\\n",
            "user_suffix": "",
            "system_prefix": "<|system|>\\n",
            "system_suffix": "\\n",
            "assistant_prefix": "<|assistant|>\\n",
            "assistant_suffix": "\\n<|assistant|>\\n",
        }

    # Default to Qwen / ChatML syntax
    preset_name = "Qwen2.5" if "qwen" in source_lower or "qwen" in repo_lower else "ChatML"
    model_slug = hf_repo.split("/")[-1].lower()
    ollama_name = (
        "qwen2.5-7b" if "qwen2.5-7b" in source_lower or "qwen2.5-7b" in repo_lower else model_slug
    )

    return {
        "preset_name": preset_name,
        "ollama_name": ollama_name,
        "stop_tokens": ["<|im_start|>", "<|im_end|>", "<|endoftext|>"],
        "user_prefix": "<|im_start|>user\\n",
        "user_suffix": "<|im_end|>\\n",
        "system_prefix": "<|im_start|>system\\n",
        "system_suffix": "<|im_end|>\\n",
        "assistant_prefix": "<|im_start|>assistant\\n",
        "assistant_suffix": "<|im_end|>\\n<|im_start|>assistant\\n",
    }
