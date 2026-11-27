# Changelog

All notable changes to `mlx-foundry` will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0] - 2026-11-27

### Added
* **Wardstone Governance Integration:** Integrated `wardstone` (`v0.1.0`) into `mlx_foundry.convert` and `mlx_foundry.benchmark` using `ThermalResourceGovernor`, `SlotLease` (`owner="mlx-foundry"`, `preempt_sidecar=True`), and `flush_accelerator_cache()` in `try/finally` blocks.
* **Engineering Canon Suite:** Added `CONTRIBUTING.md`, `CHANGELOG.md`, `SECURITY.md`, `ONBOARDING.md`, `docs/canon/config.yaml`, and `docs/canon/architecture.md`.
* **Governor Test Suite:** Added `tests/test_wardstone_governor.py` covering `SlotLease` acquisition, error-path release, accelerator cache flushing, and cached benchmark fast-path behavior.

### Changed
* **Model Card Frontmatter:** Parameterized `base_model_relation` across card generation, publishing, and pipeline workflows, adding `quantized` relation support for 16-bit exports.
* **LiteRT Export Backend:** Added explicit `litert` backend (`src/mlx_foundry/backends/litert.py`) failing closed when `ai-edge-torch` is absent.
* **Documentation Hygiene:** Stripped emojis and workstation-specific paths from `README.md` and Engineering Canon documentation.

## [0.2.0] - 2026-09-21

### Added
* Modular multi-backend architecture supporting `mlx_lm`, `mlx_vlm`, and `mflux`.
* Multi-connection `aria2c` snapshot downloader with source cache pruning and asynchronous prefetching.
* Automated Hugging Face Hub publishing and collection cross-linking.
