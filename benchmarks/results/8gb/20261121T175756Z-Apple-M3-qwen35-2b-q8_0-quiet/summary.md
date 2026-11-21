# Baseline: Qwen3.5-2B-Q8_0.gguf on Apple M3 (8gb)

Label: **locally-calibrated** (measured on this machine; thermals and background load uncontrolled). Not a claim; see `docs/claims.md`.

- macOS 27.0.1 (26A434), power: Battery Power, memory 8589934592 B
- tuckmill `546531a324d4` (dirty tree: False), llama.cpp b11695 (pinned, scripts/fetch-llama.sh)
- artifact `Qwen3.5-2B-Q8_0.gguf`, 2012012800 B, sha256 `1b04acba824817554f4ce23639bc8495ff70453b8fcb047900c731521021f2c1`
- shape: 512 in / 256 out, 3 repetitions per invocation, context 2048, 3 ABBA rounds; frozen synthetic token ids (standard-v1), identical for both harness engines
- warm = repetitions 2..n of an invocation; cold-load = repetition 1 (includes worker spawn + model load; page cache uncontrolled)

## Runs (ABBA order)

| round | slot | engine | status | reclaimable before | swap delta | warm TTFT | warm e2e tok/s | warm decode tok/s | cold-load TTFT | footprint peak |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | A | tuckcore-metal | ok | 2573 MiB | 0 MiB | 655 ms | 35.14 | 38.46 | 2268 ms | 2148 MiB |
| 1 | B | llama-metal | ok | 3772 MiB | 0 MiB | 610 ms | 36.85 | 40.24 | 2185 ms | 249 MiB |
| 2 | A | llama-metal | ok | 3607 MiB | 0 MiB | 603 ms | 37.04 | 40.43 | 982 ms | 249 MiB |
| 2 | B | tuckcore-metal | ok | 3599 MiB | 0 MiB | 829 ms | 34.30 | 38.35 | 1750 ms | 2148 MiB |
| 3 | A | tuckcore-metal | ok | 3824 MiB | 0 MiB | 1257 ms | 27.97 | 32.30 | 1754 ms | 2148 MiB |
| 3 | B | llama-metal | ok | 3880 MiB | 0 MiB | 1086 ms | 28.01 | 31.66 | 1469 ms | 248 MiB |

## Medians across valid rounds

| engine | rounds | warm TTFT | warm e2e tok/s | warm decode tok/s | footprint peak |
| --- | --- | --- | --- | --- | --- |
| llama-metal | 3 | 610 ms | 36.85 | 40.24 | 249 MiB |
| tuckcore-metal | 3 | 829 ms | 34.30 | 38.35 | 2148 MiB |

## Paired ratios (TuckCore / llama.cpp, same round)

- warm decode tok/s ratio: median 0.956 over 3 paired rounds (0.956, 0.949, 1.020)
- warm TTFT ratio (higher = slower prefill): median 1.16 over 3 paired rounds

## Vendor tool: pinned llama-bench (own prompt tokens; not identity-matched)

not run

## MLX

available: False — mlx 0.32.4, mlx-lm 0.32.2; mlx-lm cannot load the identical GGUF artifact (NotADirectoryError: [Errno 20] Not a directory: 'artifacts-local/Qwen3.5-2B-Q8_0.gguf/config.json'); an MLX-format conversion would be a different artifact, so no MLX row is recorded
