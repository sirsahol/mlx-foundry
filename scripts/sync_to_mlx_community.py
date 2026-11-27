#!/usr/bin/env python3
"""Sync unique models from SirSahOl namespace to mlx-community on Hugging Face.

Features:
- Enforces strict duplicate detection (skips any model already existing in mlx-community).
- Uses Hugging Face server-side repo duplication (zero local disk/bandwidth consumption).
- Handles Hugging Face's 15 repos/day rate limit gracefully with clear progress reporting.
"""

import re
import sys
import time
from huggingface_hub import HfApi

def sync_to_mlx_community():
    api = HfApi()

    print("Fetching SirSahOl models and mlx-community catalog...")
    user_models = [m.id for m in api.list_models(author="SirSahOl")]
    comm_models = set(m.id.lower() for m in api.list_models(author="mlx-community"))

    def is_duplicate(repo_id: str, comm_set: set) -> tuple[bool, str | None]:
        base = repo_id.split("/")[-1].lower()
        target = f"mlx-community/{base}".lower()
        if target in comm_set:
            return True, target
        simplified = re.sub(r"[-_]chat[-_]mlx[-_]", "-", base)
        simplified = re.sub(r"[-_]mlx[-_]", "-", simplified)
        if f"mlx-community/{simplified}".lower() in comm_set:
            return True, f"mlx-community/{simplified}"
        return False, None

    success_count = 0
    skip_count = 0
    rate_limited = False

    print(f"\nAnalyzing {len(user_models)} models for mlx-community cross-publishing...\n")

    for um in sorted(user_models):
        if um == "SirSahOl/SirSahOl":
            continue

        dup, reason = is_duplicate(um, comm_models)
        if dup:
            skip_count += 1
            continue

        if rate_limited:
            continue

        base = um.split("/")[-1]
        target_repo = f"mlx-community/{base}"
        print(f"Cross-publishing: {um} -> {target_repo}...")
        try:
            api.duplicate_repo(from_id=um, to_id=target_repo, repo_type="model")
            print(f"  ✓ Success: {target_repo}")
            success_count += 1
            comm_models.add(target_repo.lower())
            time.sleep(1.0)
        except Exception as e:
            if "exceeded the rate limit" in str(e) or "429" in str(e):
                print(f"\n[!] Hugging Face Daily Duplication Rate Limit Reached (15 repos/day).")
                print(f"    {e}")
                print(f"    Stopping run for today. Remaining models can be synced tomorrow.")
                rate_limited = True
            else:
                print(f"  ✗ Error duplicating {um}: {e}")

    print(f"\n=== Sync Summary ===")
    print(f"Successfully cross-published this run: {success_count}")
    print(f"Duplicates safely skipped: {skip_count}")
    print(f"Status: {'Daily rate limit reached. Re-run tomorrow for next batch.' if rate_limited else 'All models synced!'}")

if __name__ == "__main__":
    sync_to_mlx_community()
