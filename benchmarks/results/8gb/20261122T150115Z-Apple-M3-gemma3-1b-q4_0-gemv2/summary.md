# Baseline: gemma-3-1b-it-Q4_0.gguf on Apple M3 (8gb)

Label: **locally-calibrated** (measured on this machine; thermals and background load uncontrolled). Not a claim; see `docs/claims.md`.

- macOS 27.0.1 (26A434), power: AC Power, memory 8589934592 B
- chine `9aff57657881` (dirty tree: False), llama.cpp b11695 (pinned, scripts/fetch-llama.sh)
- artifact `gemma-3-1b-it-Q4_0.gguf`, 721918496 B, sha256 `27ee88e03be02e9ba73def9a819d570d8ad73716e50769e87f374ae394b0276e`
- shape: 512 in / 256 out, 3 repetitions per invocation, context 2048, 3 ABBA rounds; frozen synthetic token ids (standard-v1), identical for both harness engines
- warm = repetitions 2..n of an invocation; cold-load = repetition 1 (includes worker spawn + model load; page cache uncontrolled)

## Runs (ABBA order)

| round | slot | engine | status | reclaimable before | swap delta | warm TTFT | warm e2e tok/s | warm decode tok/s | cold-load TTFT | footprint peak |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | A | chinecore-metal | ok | 3094 MiB | 0 MiB | 313 ms | 84.78 | 94.22 | 806 ms | 938 MiB |
| 1 | B | llama-metal | ok | 3356 MiB | 0 MiB | 287 ms | 91.18 | 101.17 | 701 ms | 199 MiB |
| 2 | A | llama-metal | ok | 3259 MiB | 0 MiB | 286 ms | 92.02 | 102.19 | 568 ms | 200 MiB |
| 2 | B | chinecore-metal | ok | 3422 MiB | 0 MiB | 312 ms | 85.65 | 95.26 | 708 ms | 938 MiB |
| 3 | A | chinecore-metal | ok | 3712 MiB | 0 MiB | 312 ms | 85.97 | 95.65 | 725 ms | 938 MiB |
| 3 | B | llama-metal | ok | 3764 MiB | 0 MiB | 287 ms | 91.86 | 102.01 | 570 ms | 200 MiB |

## Medians across valid rounds

| engine | rounds | warm TTFT | warm e2e tok/s | warm decode tok/s | footprint peak |
| --- | --- | --- | --- | --- | --- |
| chinecore-metal | 3 | 312 ms | 85.65 | 95.26 | 938 MiB |
| llama-metal | 3 | 287 ms | 91.86 | 102.01 | 200 MiB |

## Paired ratios (ChineCore / llama.cpp, same round)

- warm decode tok/s ratio: median 0.932 over 3 paired rounds (0.931, 0.932, 0.938)
- warm TTFT ratio (higher = slower prefill): median 1.09 over 3 paired rounds

## Vendor tool: pinned llama-bench (own prompt tokens; not identity-matched)

not run

## MLX

available: False — mlx 0.32.4, mlx-lm 0.32.2; mlx-lm cannot load the identical GGUF artifact (NotADirectoryError: [Errno 20] Not a directory: 'artifacts-local/gemma-3-1b-it-Q4_0.gguf/config.json'); an MLX-format conversion would be a different artifact, so no MLX row is recorded
