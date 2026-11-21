# Baseline: gemma-3-1b-it-Q8_0.gguf on Apple M3 (8gb)

Label: **locally-calibrated** (measured on this machine; thermals and background load uncontrolled). Not a claim; see `docs/claims.md`.

- macOS 27.0.1 (26A434), power: Battery Power, memory 8589934592 B
- tuckmill `5f242d00badd` (dirty tree: False), llama.cpp b11695 (pinned, scripts/fetch-llama.sh)
- artifact `gemma-3-1b-it-Q8_0.gguf`, 1069306368 B, sha256 `b205840c5dcef55078e37d344677869a714ffd42a4ae448c48dcfb52e4bb10d5`
- shape: 512 in / 256 out, 3 repetitions per invocation, context 2048, 3 ABBA rounds; frozen synthetic token ids (standard-v1), identical for both harness engines
- warm = repetitions 2..n of an invocation; cold-load = repetition 1 (includes worker spawn + model load; page cache uncontrolled)

## Runs (ABBA order)

| round | slot | engine | status | reclaimable before | swap delta | warm TTFT | warm e2e tok/s | warm decode tok/s | cold-load TTFT | footprint peak |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | A | tuckcore-metal | ok | 4281 MiB | 0 MiB | 326 ms | 60.01 | 64.73 | 991 ms | 1227 MiB |
| 1 | B | llama-metal | ok | 3944 MiB | 0 MiB | 289 ms | 70.18 | 75.93 | 660 ms | 199 MiB |
| 2 | A | llama-metal | ok | 3844 MiB | 0 MiB | 289 ms | 69.93 | 75.64 | 587 ms | 200 MiB |
| 2 | B | tuckcore-metal | ok | 3530 MiB | 0 MiB | 326 ms | 60.29 | 65.06 | 784 ms | 1227 MiB |
| 3 | A | tuckcore-metal | ok | 4192 MiB | 0 MiB | 325 ms | 60.37 | 65.13 | 767 ms | 1227 MiB |
| 3 | B | llama-metal | ok | 4209 MiB | 0 MiB | 306 ms | 58.36 | 62.52 | 584 ms | 200 MiB |

## Medians across valid rounds

| engine | rounds | warm TTFT | warm e2e tok/s | warm decode tok/s | footprint peak |
| --- | --- | --- | --- | --- | --- |
| llama-metal | 3 | 289 ms | 69.93 | 75.64 | 200 MiB |
| tuckcore-metal | 3 | 326 ms | 60.29 | 65.06 | 1227 MiB |

## Paired ratios (TuckCore / llama.cpp, same round)

- warm decode tok/s ratio: median 0.860 over 3 paired rounds (0.852, 0.860, 1.042)
- warm TTFT ratio (higher = slower prefill): median 1.13 over 3 paired rounds

## Vendor tool: pinned llama-bench (own prompt tokens; not identity-matched)

not run

## MLX

available: False — mlx 0.32.4, mlx-lm 0.32.2; mlx-lm cannot load the identical GGUF artifact (NotADirectoryError: [Errno 20] Not a directory: 'artifacts-local/gemma-3-1b-it-Q8_0.gguf/config.json'); an MLX-format conversion would be a different artifact, so no MLX row is recorded
