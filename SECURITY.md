# Security Policy

## Threat Model and Operational Invariants

1. **Zero Local Path Leakage:**
   Generated model cards, reproduction commands, logs, and documentation must never embed local user home directories or private workspace paths. Use relative paths (`output/...`) or `$HOME/...`.
2. **Credential Hygiene:**
   Hugging Face authentication relies on `HF_TOKEN` or `huggingface_hub` credential stores (`hf auth login`). Never commit `.env` files, API tokens, or private keys to version control.
3. **Restricted Lease State Permissions:**
   Cross-process `SlotLease` lock and epoch files managed via `wardstone.governor` (`~/.cache/wardstone/slot_epoch` or `WARDSTONE_SLOT_LEASE_PATH`) are written atomically with `0o600` owner-only permissions.
4. **Fail-Closed Conversion and Benchmarking:**
   Missing compilers (`ai-edge-torch` for LiteRT) or unimplemented benchmark probes raise explicit errors rather than producing dummy weights or fabricated telemetry.
5. **Subprocess and Remote Code Controls:**
   Conversion commands use explicit argument arrays (`subprocess.run([...])`) without shell interpolation.

---

## Reporting Vulnerabilities

If you discover a security vulnerability in `mlx-foundry`, please report it privately via GitHub Security Advisories on the repository rather than opening a public issue. Include reproduction steps, Python and macOS/MLX versions, and impact scope.
