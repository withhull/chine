# Baseline: Qwen3.5-0.8B-Q4_K_M.gguf on Apple M3 (8gb)

Label: **locally-calibrated** (measured on this machine; thermals and background load uncontrolled). Not a claim; see `docs/claims.md`.

- macOS 27.0.1 (26A434), power: Battery Power, memory 8589934592 B
- chine `d50f60914a67` (dirty tree: False), llama.cpp b11695 (pinned, scripts/fetch-llama.sh)
- artifact `Qwen3.5-0.8B-Q4_K_M.gguf`, 549698976 B, sha256 `ac7c9d7a1b3e3695bb3bd50f8ceaa97f9c93e99ccc3d3d1a620301b6dd6d3d86`
- shape: 512 in / 256 out, 3 repetitions per invocation, context 2048, 3 ABBA rounds; frozen synthetic token ids (standard-v1), identical for both harness engines
- warm = repetitions 2..n of an invocation; cold-load = repetition 1 (includes worker spawn + model load; page cache uncontrolled)

## Runs (ABBA order)

| round | slot | engine | status | reclaimable before | swap delta | warm TTFT | warm e2e tok/s | warm decode tok/s | cold-load TTFT | footprint peak |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | A | chinecore-metal | ok | 3451 MiB | 0 MiB | 388 ms | 84.08 | 95.99 | 821 ms | 817 MiB |
| 1 | B | llama-metal | ok | 3410 MiB | 0 MiB | 280 ms | 99.08 | 110.72 | 922 ms | 240 MiB |
| 2 | A | llama-metal | ok | 3366 MiB | 0 MiB | 279 ms | 99.20 | 110.78 | 586 ms | 242 MiB |
| 2 | B | chinecore-metal | ok | 3369 MiB | 0 MiB | 388 ms | 84.04 | 95.93 | 800 ms | 817 MiB |
| 3 | A | chinecore-metal | ok | 3489 MiB | 0 MiB | 387 ms | 84.06 | 95.93 | 817 ms | 817 MiB |
| 3 | B | llama-metal | ok | 3513 MiB | 0 MiB | 279 ms | 99.64 | 111.36 | 591 ms | 241 MiB |

## Medians across valid rounds

| engine | rounds | warm TTFT | warm e2e tok/s | warm decode tok/s | footprint peak |
| --- | --- | --- | --- | --- | --- |
| chinecore-metal | 3 | 388 ms | 84.06 | 95.93 | 817 MiB |
| llama-metal | 3 | 279 ms | 99.20 | 110.78 | 241 MiB |

## Paired ratios (ChineCore / llama.cpp, same round)

- warm decode tok/s ratio: median 0.866 over 3 paired rounds (0.867, 0.866, 0.861)
- warm TTFT ratio (higher = slower prefill): median 1.39 over 3 paired rounds

## Vendor tool: pinned llama-bench (own prompt tokens; not identity-matched)

not run

## MLX

available: False — mlx 0.32.4, mlx-lm 0.32.2; mlx-lm cannot load the identical GGUF artifact (NotADirectoryError: [Errno 20] Not a directory: 'artifacts-local/Qwen3.5-0.8B-Q4_K_M.gguf/config.json'); an MLX-format conversion would be a different artifact, so no MLX row is recorded
