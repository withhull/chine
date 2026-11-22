# Baseline: gemma-3-1b-it-Q8_0.gguf on Apple M3 (8gb)

Label: **locally-calibrated** (measured on this machine; thermals and background load uncontrolled). Not a claim; see `docs/claims.md`.

- macOS 27.0.1 (26A434), power: AC Power, memory 8589934592 B
- tuckmill `c47b764862f3` (dirty tree: True), llama.cpp b11695 (pinned, scripts/fetch-llama.sh)
- artifact `gemma-3-1b-it-Q8_0.gguf`, 1069306368 B, sha256 `b205840c5dcef55078e37d344677869a714ffd42a4ae448c48dcfb52e4bb10d5`
- shape: 512 in / 256 out, 3 repetitions per invocation, context 2048, 3 ABBA rounds; frozen synthetic token ids (standard-v1), identical for both harness engines
- warm = repetitions 2..n of an invocation; cold-load = repetition 1 (includes worker spawn + model load; page cache uncontrolled)

## Runs (ABBA order)

| round | slot | engine | status | reclaimable before | swap delta | warm TTFT | warm e2e tok/s | warm decode tok/s | cold-load TTFT | footprint peak |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | A | tuckcore-metal | ok | 3564 MiB | -8 MiB | 389 ms | 53.20 | 57.65 | 1453 ms | 1232 MiB |
| 1 | B | llama-metal | ok | 3380 MiB | 0 MiB | 357 ms | 51.96 | 55.81 | 857 ms | 198 MiB |
| 2 | A | llama-metal | ok | 3382 MiB | -24 MiB | 359 ms | 52.62 | 56.61 | 735 ms | 204 MiB |
| 2 | B | tuckcore-metal | ok | 3239 MiB | 0 MiB | 385 ms | 53.84 | 58.36 | 916 ms | 1230 MiB |
| 3 | A | tuckcore-metal | ok | 3569 MiB | -16 MiB | 444 ms | 51.72 | 56.63 | 892 ms | 1230 MiB |
| 3 | B | llama-metal | ok | 3529 MiB | 0 MiB | 449 ms | 47.38 | 51.45 | 717 ms | 201 MiB |

## Medians across valid rounds

| engine | rounds | warm TTFT | warm e2e tok/s | warm decode tok/s | footprint peak |
| --- | --- | --- | --- | --- | --- |
| llama-metal | 3 | 359 ms | 51.96 | 55.81 | 201 MiB |
| tuckcore-metal | 3 | 389 ms | 53.20 | 57.65 | 1230 MiB |

## Paired ratios (TuckCore / llama.cpp, same round)

- warm decode tok/s ratio: median 1.033 over 3 paired rounds (1.033, 1.031, 1.101)
- warm TTFT ratio (higher = slower prefill): median 1.07 over 3 paired rounds

## Vendor tool: pinned llama-bench (own prompt tokens; not identity-matched)

not run

## MLX

available: False — mlx 0.32.4, mlx-lm 0.32.2; mlx-lm cannot load the identical GGUF artifact (NotADirectoryError: [Errno 20] Not a directory: 'artifacts-local/gemma-3-1b-it-Q8_0.gguf/config.json'); an MLX-format conversion would be a different artifact, so no MLX row is recorded
