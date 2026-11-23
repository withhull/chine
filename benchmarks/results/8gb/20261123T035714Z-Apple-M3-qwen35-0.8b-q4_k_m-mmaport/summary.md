# Baseline: Qwen3.5-0.8B-Q4_K_M.gguf on Apple M3 (8gb)

Label: **locally-calibrated** (measured on this machine; thermals and background load uncontrolled). Not a claim; see `docs/claims.md`.

- macOS 27.0.1 (26A434), power: Battery Power, memory 8589934592 B
- chine `8dd124cdd87c` (dirty tree: False), llama.cpp b11695 (pinned, scripts/fetch-llama.sh)
- artifact `Qwen3.5-0.8B-Q4_K_M.gguf`, 549698976 B, sha256 `ac7c9d7a1b3e3695bb3bd50f8ceaa97f9c93e99ccc3d3d1a620301b6dd6d3d86`
- shape: 512 in / 256 out, 3 repetitions per invocation, context 2048, 3 ABBA rounds; frozen synthetic token ids (standard-v1), identical for both harness engines
- warm = repetitions 2..n of an invocation; cold-load = repetition 1 (includes worker spawn + model load; page cache uncontrolled)

## Runs (ABBA order)

| round | slot | engine | status | reclaimable before | swap delta | warm TTFT | warm e2e tok/s | warm decode tok/s | cold-load TTFT | footprint peak |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | A | chinecore-metal | ok | 4347 MiB | 0 MiB | 384 ms | 97.00 | 113.06 | 808 ms | 818 MiB |
| 1 | B | llama-metal | ok | 4387 MiB | 0 MiB | 279 ms | 99.49 | 111.17 | 648 ms | 241 MiB |
| 2 | A | llama-metal | ok | 3835 MiB | 0 MiB | 279 ms | 99.60 | 111.28 | 589 ms | 241 MiB |
| 2 | B | chinecore-metal | ok | 3903 MiB | 0 MiB | 384 ms | 97.09 | 113.19 | 795 ms | 818 MiB |
| 3 | A | chinecore-metal | ok | 4394 MiB | 0 MiB | 385 ms | 97.09 | 113.25 | 793 ms | 818 MiB |
| 3 | B | llama-metal | ok | 4285 MiB | 0 MiB | 279 ms | 99.50 | 111.16 | 590 ms | 243 MiB |

## Medians across valid rounds

| engine | rounds | warm TTFT | warm e2e tok/s | warm decode tok/s | footprint peak |
| --- | --- | --- | --- | --- | --- |
| chinecore-metal | 3 | 384 ms | 97.09 | 113.19 | 818 MiB |
| llama-metal | 3 | 279 ms | 99.50 | 111.17 | 241 MiB |

## Paired ratios (ChineCore / llama.cpp, same round)

- warm decode tok/s ratio: median 1.017 over 3 paired rounds (1.017, 1.017, 1.019)
- warm TTFT ratio (higher = slower prefill): median 1.38 over 3 paired rounds

## Vendor tool: pinned llama-bench (own prompt tokens; not identity-matched)

not run

## MLX

available: False — mlx 0.32.4, mlx-lm 0.32.2; mlx-lm cannot load the identical GGUF artifact (NotADirectoryError: [Errno 20] Not a directory: 'artifacts-local/Qwen3.5-0.8B-Q4_K_M.gguf/config.json'); an MLX-format conversion would be a different artifact, so no MLX row is recorded
