# Graph Report - mlx-foundry  (2026-11-27)

## Corpus Check
- 43 files · ~21,746 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 74 file(s) not represented in the graph (top: (none) 74)

## Summary
- 495 nodes · 1066 edges · 15 communities (13 shown, 2 thin omitted)
- Extraction: 96% EXTRACTED · 4% INFERRED · 0% AMBIGUOUS · INFERRED: 38 edges (avg confidence: 0.92)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `78446bac`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- pathlib
- convert_opt.py
- json
- generate_model_card
- run_benchmark
- convert_model
- .convert
- cli.py
- get_backend
- opt.py
- pipeline.py
- publish_model
- MLX Foundry
- mlx-foundry

## God Nodes (most connected - your core abstractions)
1. `convert_model()` - 33 edges
2. `generate_model_card()` - 31 edges
3. `run_benchmark()` - 28 edges
4. `BenchmarkResult` - 25 edges
5. `ConversionResult` - 20 edges
6. `download_model_snapshot()` - 20 edges
7. `BaseBackend` - 18 edges
8. `get_backend()` - 16 edges
9. `run_batch()` - 16 edges
10. `run_pipeline()` - 16 edges

## Surprising Connections (you probably didn't know these)
- `Core Engineering Standards` --references--> `run_benchmark()`  [INFERRED]
  CONTRIBUTING.md → src/mlx_foundry/benchmark.py
- `A. Single-Slot Hardware Governance (`convert.py`, `benchmark.py`)` --references--> `run_benchmark()`  [INFERRED]
  docs/canon/architecture.md → src/mlx_foundry/benchmark.py
- `D. Model Card Generation and Hub Publishing (`card.py`, `publish.py`)` --references--> `generate_model_card()`  [INFERRED]
  docs/canon/architecture.md → src/mlx_foundry/card.py
- `4. `card`` --references--> `card()`  [INFERRED]
  README.md → src/mlx_foundry/cli.py
- `5. `publish`` --references--> `publish()`  [INFERRED]
  README.md → src/mlx_foundry/cli.py

## Import Cycles
- None detected.

## Communities (15 total, 2 thin omitted)

### Community 0 - "pathlib"
Cohesion: 0.05
Nodes (14): BaseBackend, LiteRTBackend, MLXLMBackend, MLXVLMBackend, BenchmarkResult, ConversionResult, ModelInfo, PublishConfig (+6 more)

### Community 1 - "convert_opt.py"
Cohesion: 0.06
Nodes (15): patch_models(), convert_and_package(), prepare_raw_dir(), synthesize_safetensors(), find_class(), download_model_snapshot(), find_aria2c(), get_hf_token() (+7 more)

### Community 2 - "json"
Cohesion: 0.06
Nodes (17): cleanup_model_cache(), load_progress(), main(), run_batch(), save_progress(), start_prefetch(), _worker(), check_architecture_support() (+9 more)

### Community 3 - "generate_model_card"
Cohesion: 0.08
Nodes (18): _detect_quant_from_path(), _extract_param_info(), _format_benchmark_table(), generate_model_card(), _get_chat_config(), _get_hardware_sizing_matrix(), _get_model_details(), _get_quant_comparison() (+10 more)

### Community 4 - "run_benchmark"
Cohesion: 0.07
Nodes (17): _detect_quant_bits(), _display_results(), flush_accelerator_cache(), measure_peak_memory_mb(), run_benchmark(), test_detect_quant_bits(), test_measure_peak_memory_mb(), test_convert_model_acquires_and_releases_slot_lease() (+9 more)

### Community 5 - "convert_model"
Cohesion: 0.09
Nodes (13): sync_to_mlx_community(), convert_model(), ensure_safetensors_symlinks(), estimate_required_space_gb(), flush_accelerator_cache(), validate_quants(), get_mlx_lm_version(), test_convert_model() (+5 more)

### Community 6 - ".convert"
Cohesion: 0.06
Nodes (25): [0.2.0] - 2026-09-21, [0.3.0] - 2026-11-27, Added, Added, Changed, Changelog, Contributing to MLX Foundry, Core Engineering Standards (+17 more)

### Community 7 - "cli.py"
Cohesion: 0.10
Nodes (16): 1. `pipeline`, 2. `convert`, 3. `benchmark`, 4. `card`, 5. `publish`, 6. `social`, Commands Reference, benchmark() (+8 more)

### Community 8 - "get_backend"
Cohesion: 0.09
Nodes (14): detect_backend(), get_backend(), MfluxBackend, test_detect_backend_diffusion_names(), test_detect_backend_llm_names(), test_get_backend_explicit_override(), test_get_backend_instances(), test_get_backend_invalid() (+6 more)

### Community 9 - "opt.py"
Cohesion: 0.13
Nodes (5): Attention, Model, ModelArgs, OPTDecoder, TransformerBlock

### Community 10 - "pipeline.py"
Cohesion: 0.11
Nodes (6): main(), format_repo_name(), run_pipeline(), generate_social_posts(), test_format_repo_name(), test_pipeline_dry_run()

### Community 11 - "publish_model"
Cohesion: 0.14
Nodes (7): _ensure_frontmatter(), publish_model(), test_ensure_frontmatter_creates_file(), test_ensure_frontmatter_prepends(), test_publish_model(), test_publish_model_16bit_relation(), test_publish_model_explicit_relation()

### Community 12 - "MLX Foundry"
Cohesion: 0.20
Nodes (9): Backends, Contributing, Dry Run (Local only, skip publishing), Features, Installation, License, MLX Foundry, Quick Start (+1 more)

## Knowledge Gaps
- **1 isolated node(s):** `mlx-foundry`
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 213 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **2 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `convert_model()` connect `convert_model` to `pathlib`, `convert_opt.py`, `json`, `run_benchmark`, `.convert`, `cli.py`, `get_backend`, `pipeline.py`?**
  _High betweenness centrality (0.141) - this node is a cross-community bridge._
- **Are the 2 inferred relationships involving `convert_model()` (e.g. with `Core Engineering Standards` and `A. Single-Slot Hardware Governance (`convert.py`, `benchmark.py`)`) actually correct?**
  _`convert_model()` has 2 INFERRED edges - model-reasoned connections that need verification._
- **What connects `mlx-foundry` to the rest of the system?**
  _1 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `pathlib` be split into smaller, more focused modules?**
  _Cohesion score 0.05450165612767239 - nodes in this community are weakly interconnected._
- **Why does `generate_model_card()` connect `generate_model_card` to `pathlib`, `convert_opt.py`, `json`, `.convert`, `cli.py`, `pipeline.py`?**
  _High betweenness centrality (0.106) - this node is a cross-community bridge._
- **Are the 2 inferred relationships involving `run_benchmark()` (e.g. with `Core Engineering Standards` and `A. Single-Slot Hardware Governance (`convert.py`, `benchmark.py`)`) actually correct?**
  _`run_benchmark()` has 2 INFERRED edges - model-reasoned connections that need verification._
- **Should `convert_opt.py` be split into smaller, more focused modules?**
  _Cohesion score 0.057329462989840346 - nodes in this community are weakly interconnected._