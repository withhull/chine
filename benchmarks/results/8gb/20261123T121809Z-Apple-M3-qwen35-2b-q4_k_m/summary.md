# Baseline: Qwen3.5-2B-Q4_K_M.gguf on Apple M3 (8gb)

Label: **locally-calibrated** (measured on this machine; thermals and background load uncontrolled). Not a claim; see `docs/claims.md`.

- macOS 27.0.1 (26A434), power: Battery Power, memory 8589934592 B
- chine `39589609eba8` (dirty tree: False), llama.cpp b11695 (pinned, scripts/fetch-llama.sh)
- artifact `Qwen3.5-2B-Q4_K_M.gguf`, 1280835840 B, sha256 `aaf42c8b7c3cab2bf3d69c355048d4a0ee9973d48f16c731c0520ee914699223`
- shape: 512 in / 256 out, 3 repetitions per invocation, context 2048, 3 ABBA rounds; frozen synthetic token ids (standard-v1), identical for both harness engines
- warm = repetitions 2..n of an invocation; cold-load = repetition 1 (includes worker spawn + model load; page cache uncontrolled)

## Runs (ABBA order)

| round | slot | engine | status | reclaimable before | swap delta | warm TTFT | warm e2e tok/s | warm decode tok/s | cold-load TTFT | footprint peak |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | A | chinecore-metal | ok | 4176 MiB | 0 MiB | 622 ms | 45.62 | 51.11 | 1418 ms | 1554 MiB |
| 1 | B | llama-metal | ok | 3941 MiB | 0 MiB | 629 ms | 50.72 | 57.71 | 1068 ms | 247 MiB |
| 2 | A | llama-metal | ok | 3940 MiB | 0 MiB | 668 ms | 50.31 | 57.68 | 974 ms | 248 MiB |
| 2 | B | chinecore-metal | ok | 3845 MiB | 0 MiB | 697 ms | 44.72 | 50.72 | 1241 ms | 1554 MiB |
| 3 | A | chinecore-metal | ok | 4368 MiB | 0 MiB | 752 ms | 40.59 | 45.90 | 1249 ms | 1554 MiB |
| 3 | B | llama-metal | ok | 4359 MiB | 0 MiB | 972 ms | 39.68 | 46.55 | 992 ms | 248 MiB |

## Medians across valid rounds

| engine | rounds | warm TTFT | warm e2e tok/s | warm decode tok/s | footprint peak |
| --- | --- | --- | --- | --- | --- |
| chinecore-metal | 3 | 697 ms | 44.72 | 50.72 | 1554 MiB |
| llama-metal | 3 | 668 ms | 50.31 | 57.68 | 248 MiB |

## Paired ratios (ChineCore / llama.cpp, same round)

- warm decode tok/s ratio: median 0.886 over 3 paired rounds (0.886, 0.879, 0.986)
- warm TTFT ratio (higher = slower prefill): median 0.99 over 3 paired rounds

## Vendor tool: pinned llama-bench (own prompt tokens; not identity-matched)

| test | tokens/s (avg ± stddev) | backend |
| --- | --- | --- |
| pp512 | 828.18 ± 2.38 | MTL,BLAS |
| tg256 | 39.43 ± 1.50 | MTL,BLAS |
- `/usr/bin/time -l`: 1506787328  maximum resident set size
- `/usr/bin/time -l`: 252250056  peak memory footprint

## MLX

available: False — mlx 0.32.4, mlx-lm 0.32.2; mlx-lm cannot load the identical GGUF artifact (NotADirectoryError: [Errno 20] Not a directory: 'artifacts-local/Qwen3.5-2B-Q4_K_M.gguf/config.json'); an MLX-format conversion would be a different artifact, so no MLX row is recorded
