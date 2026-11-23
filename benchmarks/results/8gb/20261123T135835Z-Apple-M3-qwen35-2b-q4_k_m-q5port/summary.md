# Baseline: Qwen3.5-2B-Q4_K_M.gguf on Apple M3 (8gb)

Label: **locally-calibrated** (measured on this machine; thermals and background load uncontrolled). Not a claim; see `docs/claims.md`.

- macOS 27.0.1 (26A434), power: Battery Power, memory 8589934592 B
- chine `7a577b1d04b4` (dirty tree: False), llama.cpp b11695 (pinned, scripts/fetch-llama.sh)
- artifact `Qwen3.5-2B-Q4_K_M.gguf`, 1280835840 B, sha256 `aaf42c8b7c3cab2bf3d69c355048d4a0ee9973d48f16c731c0520ee914699223`
- shape: 512 in / 256 out, 3 repetitions per invocation, context 2048, 3 ABBA rounds; frozen synthetic token ids (standard-v1), identical for both harness engines
- warm = repetitions 2..n of an invocation; cold-load = repetition 1 (includes worker spawn + model load; page cache uncontrolled)

## Runs (ABBA order)

| round | slot | engine | status | reclaimable before | swap delta | warm TTFT | warm e2e tok/s | warm decode tok/s | cold-load TTFT | footprint peak |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | A | chinecore-metal | ok | 2951 MiB | 0 MiB | 622 ms | 51.35 | 58.44 | 1896 ms | 1556 MiB |
| 1 | B | llama-metal | ok | 3885 MiB | -8 MiB | 621 ms | 50.72 | 57.62 | 1054 ms | 248 MiB |
| 2 | A | llama-metal | ok | 3466 MiB | 0 MiB | 619 ms | 50.83 | 57.74 | 965 ms | 249 MiB |
| 2 | B | chinecore-metal | ok | 3484 MiB | 0 MiB | 621 ms | 51.64 | 58.81 | 1258 ms | 1554 MiB |
| 3 | A | chinecore-metal | ok | 3763 MiB | -8 MiB | 657 ms | 50.47 | 57.75 | 1228 ms | 1554 MiB |
| 3 | B | llama-metal | ok | 3803 MiB | 0 MiB | 668 ms | 49.61 | 56.78 | 1128 ms | 248 MiB |

## Medians across valid rounds

| engine | rounds | warm TTFT | warm e2e tok/s | warm decode tok/s | footprint peak |
| --- | --- | --- | --- | --- | --- |
| chinecore-metal | 3 | 622 ms | 51.35 | 58.44 | 1554 MiB |
| llama-metal | 3 | 621 ms | 50.72 | 57.62 | 248 MiB |

## Paired ratios (ChineCore / llama.cpp, same round)

- warm decode tok/s ratio: median 1.017 over 3 paired rounds (1.014, 1.018, 1.017)
- warm TTFT ratio (higher = slower prefill): median 1.00 over 3 paired rounds

## Vendor tool: pinned llama-bench (own prompt tokens; not identity-matched)

| test | tokens/s (avg ± stddev) | backend |
| --- | --- | --- |
| pp512 | 824.33 ± 22.21 | MTL,BLAS |
| tg256 | 57.94 ± 1.12 | MTL,BLAS |
- `/usr/bin/time -l`: 1457635328  maximum resident set size
- `/usr/bin/time -l`: 252053424  peak memory footprint

## MLX

available: False — mlx 0.32.4, mlx-lm 0.32.2; mlx-lm cannot load the identical GGUF artifact (NotADirectoryError: [Errno 20] Not a directory: 'artifacts-local/Qwen3.5-2B-Q4_K_M.gguf/config.json'); an MLX-format conversion would be a different artifact, so no MLX row is recorded
