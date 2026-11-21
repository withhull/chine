# Baseline: gemma-3-1b-it-Q8_0.gguf on Apple M3 (8gb)

Label: **locally-calibrated** (measured on this machine; thermals and background load uncontrolled). Not a claim; see `docs/claims.md`.

- macOS 27.0.1 (26A434), power: AC Power, memory 8589934592 B
- tuckmill `2d03884353f6` (dirty tree: False), llama.cpp b11695 (pinned, scripts/fetch-llama.sh)
- artifact `gemma-3-1b-it-Q8_0.gguf`, 1069306368 B, sha256 `b205840c5dcef55078e37d344677869a714ffd42a4ae448c48dcfb52e4bb10d5`
- shape: 512 in / 256 out, 3 repetitions per invocation, context 2048, 3 ABBA rounds; frozen synthetic token ids (standard-v1), identical for both harness engines
- warm = repetitions 2..n of an invocation; cold-load = repetition 1 (includes worker spawn + model load; page cache uncontrolled)

## Runs (ABBA order)

| round | slot | engine | status | reclaimable before | swap delta | warm TTFT | warm e2e tok/s | warm decode tok/s | cold-load TTFT | footprint peak |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | A | tuckcore-metal | ok | 4296 MiB | 0 MiB | 330 ms | 60.22 | 65.04 | 796 ms | 1227 MiB |
| 1 | B | llama-metal | ok | 3647 MiB | 0 MiB | 290 ms | 70.14 | 75.89 | 653 ms | 197 MiB |
| 2 | A | llama-metal | ok | 3688 MiB | 0 MiB | 291 ms | 70.08 | 75.85 | 591 ms | 199 MiB |
| 2 | B | tuckcore-metal | ok | 3699 MiB | 0 MiB | 331 ms | 60.09 | 64.90 | 772 ms | 1226 MiB |
| 3 | A | tuckcore-metal | ok | 3938 MiB | 0 MiB | 330 ms | 60.35 | 65.19 | 759 ms | 1227 MiB |
| 3 | B | llama-metal | ok | 3901 MiB | 0 MiB | 289 ms | 70.31 | 76.09 | 586 ms | 200 MiB |

## Medians across valid rounds

| engine | rounds | warm TTFT | warm e2e tok/s | warm decode tok/s | footprint peak |
| --- | --- | --- | --- | --- | --- |
| llama-metal | 3 | 290 ms | 70.14 | 75.89 | 199 MiB |
| tuckcore-metal | 3 | 330 ms | 60.22 | 65.04 | 1227 MiB |

## Paired ratios (TuckCore / llama.cpp, same round)

- warm decode tok/s ratio: median 0.857 over 3 paired rounds (0.857, 0.856, 0.857)
- warm TTFT ratio (higher = slower prefill): median 1.14 over 3 paired rounds

## Vendor tool: pinned llama-bench (own prompt tokens; not identity-matched)

not run

## MLX

available: False — mlx 0.32.4, mlx-lm 0.32.2; mlx-lm cannot load the identical GGUF artifact (NotADirectoryError: [Errno 20] Not a directory: 'artifacts-local/gemma-3-1b-it-Q8_0.gguf/config.json'); an MLX-format conversion would be a different artifact, so no MLX row is recorded
