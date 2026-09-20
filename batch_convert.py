#!/usr/bin/env python3
"""Batch model conversion script for MLX Foundry (Tier 1 and Tier 2)."""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure src is in sys.path
SRC_DIR = Path(__file__).resolve().parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from mlx_foundry.batch_convert import (  # noqa: F401
    AUTHOR,
    COLLECTION_SLUG,
    MIN_DISK_GB,
    OUTPUT_DIR,
    PROGRESS_FILE,
    QUANTS,
    TIER_1_MODELS,
    TIER_2_MODELS,
    cleanup_model_cache,
    load_progress,
    main,
    run_batch,
    save_progress,
)

if __name__ == "__main__":
    main()
