# Baseline: gemma-3-1b-it-Q8_0.gguf on Apple M3 (8gb)

Label: **locally-calibrated** (measured on this machine; thermals and background load uncontrolled). Not a claim; see `docs/claims.md`.

- macOS 27.0.1 (26A434), power: AC Power, memory 8589934592 B
- tuckmill `118a7119eab2` (dirty tree: False), llama.cpp b11695 (pinned, scripts/fetch-llama.sh)
- artifact `gemma-3-1b-it-Q8_0.gguf`, 1069306368 B, sha256 `b205840c5dcef55078e37d344677869a714ffd42a4ae448c48dcfb52e4bb10d5`
- shape: 512 in / 256 out, 3 repetitions per invocation, context 2048, 3 ABBA rounds; frozen synthetic token ids (standard-v1), identical for both harness engines
- warm = repetitions 2..n of an invocation; cold-load = repetition 1 (includes worker spawn + model load; page cache uncontrolled)

## Runs (ABBA order)

| round | slot | engine | status | reclaimable before | swap delta | warm TTFT | warm e2e tok/s | warm decode tok/s | cold-load TTFT | footprint peak |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | A | tuckcore-metal | ok | 3735 MiB | -40 MiB | 387 ms | 52.78 | 57.15 | 951 ms | 1229 MiB |
| 1 | B | llama-metal | ok | 3414 MiB | -24 MiB | 351 ms | 52.78 | 56.69 | 823 ms | 198 MiB |
| 2 | A | llama-metal | ok | 3183 MiB | 0 MiB | 352 ms | 52.78 | 56.69 | 713 ms | 200 MiB |
| 2 | B | tuckcore-metal | ok | 3219 MiB | 0 MiB | 385 ms | 53.91 | 58.45 | 971 ms | 1229 MiB |
| 3 | A | tuckcore-metal | ok | 3389 MiB | -16 MiB | 391 ms | 53.78 | 58.37 | 946 ms | 1231 MiB |
| 3 | B | llama-metal | ok | 3382 MiB | 0 MiB | 436 ms | 45.83 | 49.48 | 747 ms | 199 MiB |

## Medians across valid rounds

| engine | rounds | warm TTFT | warm e2e tok/s | warm decode tok/s | footprint peak |
| --- | --- | --- | --- | --- | --- |
| llama-metal | 3 | 352 ms | 52.78 | 56.69 | 199 MiB |
| tuckcore-metal | 3 | 387 ms | 53.78 | 58.37 | 1229 MiB |

## Paired ratios (TuckCore / llama.cpp, same round)

- warm decode tok/s ratio: median 1.031 over 3 paired rounds (1.008, 1.031, 1.180)
- warm TTFT ratio (higher = slower prefill): median 1.09 over 3 paired rounds

## Vendor tool: pinned llama-bench (own prompt tokens; not identity-matched)

not run

## MLX

available: False — mlx 0.32.4, mlx-lm 0.32.2; mlx-lm cannot load the identical GGUF artifact (NotADirectoryError: [Errno 20] Not a directory: 'artifacts-local/gemma-3-1b-it-Q8_0.gguf/config.json'); an MLX-format conversion would be a different artifact, so no MLX row is recorded
