#!/usr/bin/env python3
"""Convert, benchmark, and publish facebook/opt-125m with custom MLX architecture."""

import io
import json
import os
from pathlib import Path
import pickle
import shutil
import subprocess
import sys
import time

from huggingface_hub import HfApi, hf_hub_download
import mlx.core as mx
import numpy as np
from rich.console import Console
from rich.panel import Panel
from safetensors.numpy import save_file

# Add src to sys.path
SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from mlx_foundry.benchmark import run_benchmark
from mlx_foundry.card import generate_model_card
from mlx_foundry.models import opt
from mlx_foundry.publish import publish_model

console = Console()

AUTHOR = "SirSahOl"
MODEL_ID = "facebook/opt-125m"
COLLECTION_SLUG = "SirSahOl/mlx-models-by-sirsahol-optimized-for-apple-silicon-6aa2b239913bcab23b1ed59a"
RAW_DIR = Path("output/opt_raw")
QUANTS = [4, 8, 16]


def synthesize_safetensors():
    """Extract raw weights from pytorch_model.bin and save as model.safetensors."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    safetensors_path = RAW_DIR / "model.safetensors"
    if safetensors_path.exists():
        console.print("[green]✓[/green] model.safetensors already synthesized.")
        return

    console.print("[bold blue]Downloading and unpacking pytorch_model.bin...[/bold blue]")
    bin_path = hf_hub_download(MODEL_ID, "pytorch_model.bin")

    import zipfile
    z = zipfile.ZipFile(bin_path, "r")

    class PyTorchUnpickler(pickle.Unpickler):
        def persistent_load(self, pid):
            typename, key, location, numel = pid[1], pid[2], pid[3], pid[4]
            data = z.read(f"archive/data/{key}")
            return np.frombuffer(data, dtype=np.float16)

        def find_class(self, module, name):
            if module == "torch" and name == "HalfStorage":
                return lambda *args: None
            if module == "torch._utils" and name == "_rebuild_tensor_v2":
                def rebuild(storage, storage_offset, size, stride, requires_grad, backward_hooks):
                    return np.lib.stride_tricks.as_strided(
                        storage[storage_offset:],
                        shape=size,
                        strides=[s * 2 for s in stride],
                    ).copy()
                return rebuild
            return super().find_class(module, name)

    pkl_bytes = z.read("archive/data.pkl")
    state_dict = PyTorchUnpickler(io.BytesIO(pkl_bytes)).load()

    save_file(state_dict, str(safetensors_path))
    console.print(f"[green]✓[/green] Synthesized {safetensors_path} ({len(state_dict)} tensors)")


def prepare_raw_dir():
    """Copy config and tokenizer files into RAW_DIR."""
    synthesize_safetensors()

    files_to_copy = [
        "config.json",
        "vocab.json",
        "merges.txt",
        "tokenizer_config.json",
        "special_tokens_map.json",
    ]
    for fname in files_to_copy:
        dest = RAW_DIR / fname
        if not dest.exists():
            p = hf_hub_download(MODEL_ID, fname)
            shutil.copy(p, dest)
            console.print(f"  [green]✓[/green] Cached {fname}")


STANDALONE_GENERATE_PY = '''#!/usr/bin/env python3
"""Standalone MLX inference script for OPT models."""
import argparse
import sys
from pathlib import Path
import mlx_lm
import mlx_lm.models

try:
    import modeling_opt_mlx as opt
except ImportError:
    from . import modeling_opt_mlx as opt

# Register opt architecture dynamically
sys.modules["mlx_lm.models.opt"] = opt

def main():
    parser = argparse.ArgumentParser(description="Run inference on MLX OPT model")
    parser.add_argument("--model", type=str, default=str(Path(__file__).parent), help="Model directory")
    parser.add_argument("--prompt", type=str, default="Why is the sky blue?", help="Input prompt")
    parser.add_argument("--max-tokens", type=int, default=100, help="Maximum tokens to generate")
    parser.add_argument("--temp", type=float, default=0.7, help="Sampling temperature")
    args = parser.parse_args()

    print(f"Loading {args.model}...")
    model, tokenizer = mlx_lm.load(args.model)
    print(f"Prompt: {args.prompt}\\n")
    response = mlx_lm.generate(
        model, tokenizer, prompt=args.prompt, max_tokens=args.max_tokens, temp=args.temp, verbose=True
    )
    print("\\nDone!")

if __name__ == "__main__":
    main()
'''


def convert_and_package():
    """Run 3-quant conversion and inject custom architecture files."""
    prepare_raw_dir()

    # Register opt in mlx_lm
    sys.modules["mlx_lm.models.opt"] = opt

    opt_py_source = Path("src/mlx_foundry/models/opt.py").read_text()

    conversion_dirs = []

    for q in QUANTS:
        label = f"{q}bit"
        out_path = Path("output") / f"opt-125m-mlx-{label}"
        if out_path.exists():
            shutil.rmtree(out_path)

        console.print(f"\\n[bold blue]Converting OPT-125M to {label}...[/bold blue]")

        # Run conversion script directly in Python process with opt registered
        from mlx_lm.convert import convert
        convert_kwargs = {
            "hf_path": str(RAW_DIR),
            "mlx_path": str(out_path),
            "quantize": (q < 16),
            "q_bits": q if q < 16 else 16,
            "q_group_size": 64,
        }
        convert(**convert_kwargs)
        console.print(f"[green]✓[/green] Converted to {out_path}")

        # Inject custom modeling and generate files into the output directory
        (out_path / "modeling_opt_mlx.py").write_text(opt_py_source)
        (out_path / "generate.py").write_text(STANDALONE_GENERATE_PY)
        os.chmod(out_path / "generate.py", 0o755)
        console.print(f"  [green]✓[/green] Injected modeling_opt_mlx.py and generate.py")

        conversion_dirs.append((q, out_path))

    # Benchmark
    console.print("\\n[bold]═══ Benchmarking OPT-125M ═══[/bold]")
    benchmark_results = []
    for q, out_path in conversion_dirs:
        br = run_benchmark(out_path)
        benchmark_results.append(br)

    all_quant_repos = [
        {
            "repo": f"{AUTHOR}/opt-125m-chat-mlx-{q}bit",
            "quant_bits": q,
            "url": f"https://huggingface.co/{AUTHOR}/opt-125m-chat-mlx-{q}bit",
        }
        for q in QUANTS
    ]

    # Generate Model Cards
    console.print("\\n[bold]═══ Generating Model Cards ═══[/bold]")
    for q, out_path in conversion_dirs:
        repo_name = f"{AUTHOR}/opt-125m-chat-mlx-{q}bit"
        generate_model_card(
            model_path=out_path,
            source_model=MODEL_ID,
            hf_repo=repo_name,
            author=AUTHOR,
            benchmark_results=benchmark_results,
            all_quant_repos=all_quant_repos,
        )
        console.print(f"  [green]✓[/green] Generated card for {repo_name}")

    # Publish
    console.print("\\n[bold]═══ Publishing to Hugging Face Hub ═══[/bold]")
    published = []
    for q, out_path in conversion_dirs:
        repo_name = f"{AUTHOR}/opt-125m-chat-mlx-{q}bit"
        url = publish_model(
            model_path=out_path,
            hf_repo=repo_name,
            source_model=MODEL_ID,
            quant_bits=q,
            cleanup=False,  # Keep local copy
        )
        published.append(repo_name)
        console.print(f"  [green]✓[/green] Published: {url}")

    # Add to Collection
    hf_api = HfApi()
    for repo in published:
        try:
            hf_api.add_collection_item(
                collection_slug=COLLECTION_SLUG,
                item_id=repo,
                item_type="model",
            )
            console.print(f"  [green]✓[/green] Enrolled in collection: {repo}")
        except Exception as e:
            console.print(f"  [yellow]⚠ Collection enrollment note: {e}[/yellow]")

    console.print("\\n[bold green]🎉 All 3 quants of OPT-125M converted, benchmarked, and published![/bold green]")


if __name__ == "__main__":
    convert_and_package()
