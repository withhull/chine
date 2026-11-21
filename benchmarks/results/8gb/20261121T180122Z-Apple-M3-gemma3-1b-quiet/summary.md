# Baseline: gemma-3-1b-it-Q8_0.gguf on Apple M3 (8gb)

Label: **locally-calibrated** (measured on this machine; thermals and background load uncontrolled). Not a claim; see `docs/claims.md`.

- macOS 27.0.1 (26A434), power: Battery Power, memory 8589934592 B
- tuckmill `546531a324d4` (dirty tree: False), llama.cpp b11695 (pinned, scripts/fetch-llama.sh)
- artifact `gemma-3-1b-it-Q8_0.gguf`, 1069306368 B, sha256 `b205840c5dcef55078e37d344677869a714ffd42a4ae448c48dcfb52e4bb10d5`
- shape: 512 in / 256 out, 3 repetitions per invocation, context 2048, 3 ABBA rounds; frozen synthetic token ids (standard-v1), identical for both harness engines
- warm = repetitions 2..n of an invocation; cold-load = repetition 1 (includes worker spawn + model load; page cache uncontrolled)

## Runs (ABBA order)

| round | slot | engine | status | reclaimable before | swap delta | warm TTFT | warm e2e tok/s | warm decode tok/s | cold-load TTFT | footprint peak |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | A | tuckcore-metal | ok | 2803 MiB | 0 MiB | 349 ms | 59.10 | 64.04 | 930 ms | 1226 MiB |
| 1 | B | llama-metal | ok | 3448 MiB | 0 MiB | 305 ms | 57.41 | 61.41 | 669 ms | 200 MiB |
| 2 | A | llama-metal | ok | 3383 MiB | 0 MiB | 291 ms | 63.68 | 68.44 | 586 ms | 199 MiB |
| 2 | B | tuckcore-metal | ok | 3185 MiB | 0 MiB | 341 ms | 60.07 | 65.04 | 811 ms | 1226 MiB |
| 3 | A | tuckcore-metal | ok | 3667 MiB | 0 MiB | 341 ms | 60.09 | 65.07 | 779 ms | 1226 MiB |
| 3 | B | llama-metal | ok | 3784 MiB | 0 MiB | 289 ms | 69.87 | 75.58 | 608 ms | 199 MiB |

## Medians across valid rounds

| engine | rounds | warm TTFT | warm e2e tok/s | warm decode tok/s | footprint peak |
| --- | --- | --- | --- | --- | --- |
| llama-metal | 3 | 291 ms | 63.68 | 68.44 | 199 MiB |
| tuckcore-metal | 3 | 341 ms | 60.07 | 65.04 | 1226 MiB |

## Paired ratios (TuckCore / llama.cpp, same round)

- warm decode tok/s ratio: median 0.950 over 3 paired rounds (1.043, 0.950, 0.861)
- warm TTFT ratio (higher = slower prefill): median 1.17 over 3 paired rounds

## Vendor tool: pinned llama-bench (own prompt tokens; not identity-matched)

not run

## MLX

available: False — mlx 0.32.4, mlx-lm 0.32.2; mlx-lm cannot load the identical GGUF artifact (NotADirectoryError: [Errno 20] Not a directory: 'artifacts-local/gemma-3-1b-it-Q8_0.gguf/config.json'); an MLX-format conversion would be a different artifact, so no MLX row is recorded
