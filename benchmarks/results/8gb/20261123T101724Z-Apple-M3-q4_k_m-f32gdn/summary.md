# Baseline: Qwen3.5-0.8B-Q4_K_M.gguf on Apple M3 (8gb)

Label: **locally-calibrated** (measured on this machine; thermals and background load uncontrolled). Not a claim; see `docs/claims.md`.

- macOS 27.0.1 (26A434), power: AC Power, memory 8589934592 B
- chine `9f878266d9a9` (dirty tree: False), llama.cpp b11695 (pinned, scripts/fetch-llama.sh)
- artifact `Qwen3.5-0.8B-Q4_K_M.gguf`, 549698976 B, sha256 `ac7c9d7a1b3e3695bb3bd50f8ceaa97f9c93e99ccc3d3d1a620301b6dd6d3d86`
- shape: 512 in / 256 out, 3 repetitions per invocation, context 2048, 3 ABBA rounds; frozen synthetic token ids (standard-v1), identical for both harness engines
- warm = repetitions 2..n of an invocation; cold-load = repetition 1 (includes worker spawn + model load; page cache uncontrolled)

## Runs (ABBA order)

| round | slot | engine | status | reclaimable before | swap delta | warm TTFT | warm e2e tok/s | warm decode tok/s | cold-load TTFT | footprint peak |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | A | chinecore-metal | ok | 3301 MiB | 0 MiB | 268 ms | 101.16 | 112.71 | 834 ms | 813 MiB |
| 1 | B | llama-metal | ok | 3496 MiB | 0 MiB | 282 ms | 98.15 | 109.62 | 708 ms | 241 MiB |
| 2 | A | llama-metal | ok | 3497 MiB | 0 MiB | 279 ms | 99.19 | 110.79 | 586 ms | 242 MiB |
| 2 | B | chinecore-metal | ok | 3470 MiB | 0 MiB | 267 ms | 101.45 | 113.03 | 711 ms | 813 MiB |
| 3 | A | chinecore-metal | ok | 3565 MiB | 0 MiB | 268 ms | 100.42 | 111.81 | 692 ms | 813 MiB |
| 3 | B | llama-metal | ok | 3431 MiB | 0 MiB | 279 ms | 99.33 | 110.94 | 601 ms | 242 MiB |

## Medians across valid rounds

| engine | rounds | warm TTFT | warm e2e tok/s | warm decode tok/s | footprint peak |
| --- | --- | --- | --- | --- | --- |
| chinecore-metal | 3 | 268 ms | 101.16 | 112.71 | 813 MiB |
| llama-metal | 3 | 279 ms | 99.19 | 110.79 | 242 MiB |

## Paired ratios (ChineCore / llama.cpp, same round)

- warm decode tok/s ratio: median 1.020 over 3 paired rounds (1.028, 1.020, 1.008)
- warm TTFT ratio (higher = slower prefill): median 0.96 over 3 paired rounds

## Vendor tool: pinned llama-bench (own prompt tokens; not identity-matched)

| test | tokens/s (avg ± stddev) | backend |
| --- | --- | --- |
| pp512 | 1911.32 ± 0.67 | MTL,BLAS |
| tg256 | 114.70 ± 0.07 | MTL,BLAS |
- `/usr/bin/time -l`: 761462784  maximum resident set size
- `/usr/bin/time -l`: 238749064  peak memory footprint

## MLX

available: False — mlx 0.32.4, mlx-lm 0.32.2; mlx-lm cannot load the identical GGUF artifact (NotADirectoryError: [Errno 20] Not a directory: 'artifacts-local/Qwen3.5-0.8B-Q4_K_M.gguf/config.json'); an MLX-format conversion would be a different artifact, so no MLX row is recorded
