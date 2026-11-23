# Baseline: Qwen3.5-0.8B-Q4_K_M.gguf on Apple M3 (8gb)

Label: **locally-calibrated** (measured on this machine; thermals and background load uncontrolled). Not a claim; see `docs/claims.md`.

- macOS 27.0.1 (26A434), power: Battery Power, memory 8589934592 B
- chine `e0fa4ca3673c` (dirty tree: False), llama.cpp b11695 (pinned, scripts/fetch-llama.sh)
- artifact `Qwen3.5-0.8B-Q4_K_M.gguf`, 549698976 B, sha256 `ac7c9d7a1b3e3695bb3bd50f8ceaa97f9c93e99ccc3d3d1a620301b6dd6d3d86`
- shape: 512 in / 256 out, 3 repetitions per invocation, context 2048, 3 ABBA rounds; frozen synthetic token ids (standard-v1), identical for both harness engines
- warm = repetitions 2..n of an invocation; cold-load = repetition 1 (includes worker spawn + model load; page cache uncontrolled)

## Runs (ABBA order)

| round | slot | engine | status | reclaimable before | swap delta | warm TTFT | warm e2e tok/s | warm decode tok/s | cold-load TTFT | footprint peak |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | A | chinecore-metal | ok | 4068 MiB | 0 MiB | 385 ms | 96.64 | 112.63 | 797 ms | 818 MiB |
| 1 | B | llama-metal | ok | 4171 MiB | 0 MiB | 278 ms | 99.59 | 111.25 | 675 ms | 241 MiB |
| 2 | A | llama-metal | ok | 3709 MiB | 0 MiB | 279 ms | 99.50 | 111.16 | 589 ms | 241 MiB |
| 2 | B | chinecore-metal | ok | 3795 MiB | 0 MiB | 388 ms | 96.90 | 113.15 | 800 ms | 818 MiB |
| 3 | A | chinecore-metal | ok | 4096 MiB | 0 MiB | 387 ms | 97.07 | 113.32 | 809 ms | 818 MiB |
| 3 | B | llama-metal | ok | 4107 MiB | 0 MiB | 279 ms | 99.62 | 111.33 | 589 ms | 242 MiB |

## Medians across valid rounds

| engine | rounds | warm TTFT | warm e2e tok/s | warm decode tok/s | footprint peak |
| --- | --- | --- | --- | --- | --- |
| chinecore-metal | 3 | 387 ms | 96.90 | 113.15 | 818 MiB |
| llama-metal | 3 | 279 ms | 99.59 | 111.25 | 241 MiB |

## Paired ratios (ChineCore / llama.cpp, same round)

- warm decode tok/s ratio: median 1.018 over 3 paired rounds (1.012, 1.018, 1.018)
- warm TTFT ratio (higher = slower prefill): median 1.39 over 3 paired rounds

## Vendor tool: pinned llama-bench (own prompt tokens; not identity-matched)

not run

## MLX

available: False — mlx 0.32.4, mlx-lm 0.32.2; mlx-lm cannot load the identical GGUF artifact (NotADirectoryError: [Errno 20] Not a directory: 'artifacts-local/Qwen3.5-0.8B-Q4_K_M.gguf/config.json'); an MLX-format conversion would be a different artifact, so no MLX row is recorded
