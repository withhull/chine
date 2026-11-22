# Baseline: gemma-3-1b-it-Q8_0.gguf on Apple M3 (8gb)

Label: **locally-calibrated** (measured on this machine; thermals and background load uncontrolled). Not a claim; see `docs/claims.md`.

- macOS 27.0.1 (26A434), power: AC Power, memory 8589934592 B
- tuckmill `183773721287` (dirty tree: False), llama.cpp b11695 (pinned, scripts/fetch-llama.sh)
- artifact `gemma-3-1b-it-Q8_0.gguf`, 1069306368 B, sha256 `b205840c5dcef55078e37d344677869a714ffd42a4ae448c48dcfb52e4bb10d5`
- shape: 512 in / 256 out, 3 repetitions per invocation, context 2048, 3 ABBA rounds; frozen synthetic token ids (standard-v1), identical for both harness engines
- warm = repetitions 2..n of an invocation; cold-load = repetition 1 (includes worker spawn + model load; page cache uncontrolled)

## Runs (ABBA order)

| round | slot | engine | status | reclaimable before | swap delta | warm TTFT | warm e2e tok/s | warm decode tok/s | cold-load TTFT | footprint peak |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | A | tuckcore-metal | ok | 4096 MiB | 0 MiB | 303 ms | 68.25 | 73.97 | 778 ms | 1277 MiB |
| 1 | B | llama-metal | ok | 4063 MiB | 0 MiB | 306 ms | 57.90 | 61.99 | 665 ms | 198 MiB |
| 2 | A | llama-metal | ok | 3824 MiB | 0 MiB | 289 ms | 70.07 | 75.81 | 592 ms | 200 MiB |
| 2 | B | tuckcore-metal | ok | 3985 MiB | 0 MiB | 301 ms | 68.40 | 74.09 | 743 ms | 1277 MiB |
| 3 | A | tuckcore-metal | ok | 4329 MiB | 0 MiB | 301 ms | 68.51 | 74.21 | 738 ms | 1277 MiB |
| 3 | B | llama-metal | ok | 4294 MiB | 0 MiB | 289 ms | 70.06 | 75.79 | 585 ms | 200 MiB |

## Medians across valid rounds

| engine | rounds | warm TTFT | warm e2e tok/s | warm decode tok/s | footprint peak |
| --- | --- | --- | --- | --- | --- |
| llama-metal | 3 | 289 ms | 70.06 | 75.79 | 200 MiB |
| tuckcore-metal | 3 | 301 ms | 68.40 | 74.09 | 1277 MiB |

## Paired ratios (TuckCore / llama.cpp, same round)

- warm decode tok/s ratio: median 0.979 over 3 paired rounds (1.193, 0.977, 0.979)
- warm TTFT ratio (higher = slower prefill): median 1.04 over 3 paired rounds

## Vendor tool: pinned llama-bench (own prompt tokens; not identity-matched)

not run

## MLX

available: False — mlx 0.32.4, mlx-lm 0.32.2; mlx-lm cannot load the identical GGUF artifact (NotADirectoryError: [Errno 20] Not a directory: 'artifacts-local/gemma-3-1b-it-Q8_0.gguf/config.json'); an MLX-format conversion would be a different artifact, so no MLX row is recorded
