# Baseline: gemma-3-1b-it-Q4_0.gguf on Apple M3 (8gb)

Label: **locally-calibrated** (measured on this machine; thermals and background load uncontrolled). Not a claim; see `docs/claims.md`.

- macOS 27.0.1 (26A434), power: Battery Power, memory 8589934592 B
- chine `483c6a158059` (dirty tree: False), llama.cpp b11695 (pinned, scripts/fetch-llama.sh)
- artifact `gemma-3-1b-it-Q4_0.gguf`, 721918496 B, sha256 `27ee88e03be02e9ba73def9a819d570d8ad73716e50769e87f374ae394b0276e`
- shape: 512 in / 256 out, 3 repetitions per invocation, context 2048, 3 ABBA rounds; frozen synthetic token ids (standard-v1), identical for both harness engines
- warm = repetitions 2..n of an invocation; cold-load = repetition 1 (includes worker spawn + model load; page cache uncontrolled)

## Runs (ABBA order)

| round | slot | engine | status | reclaimable before | swap delta | warm TTFT | warm e2e tok/s | warm decode tok/s | cold-load TTFT | footprint peak |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | A | chinecore-metal | ok | 4406 MiB | 0 MiB | 311 ms | 82.45 | 91.28 | 696 ms | 938 MiB |
| 1 | B | llama-metal | ok | 4278 MiB | 0 MiB | 286 ms | 92.24 | 102.45 | 609 ms | 200 MiB |
| 2 | A | llama-metal | ok | 3650 MiB | 0 MiB | 287 ms | 79.91 | 87.68 | 559 ms | 199 MiB |
| 2 | B | chinecore-metal | ok | 3684 MiB | 0 MiB | 311 ms | 82.45 | 91.29 | 701 ms | 938 MiB |
| 3 | A | chinecore-metal | ok | 4223 MiB | 0 MiB | 311 ms | 82.48 | 91.33 | 701 ms | 938 MiB |
| 3 | B | llama-metal | ok | 4345 MiB | 0 MiB | 287 ms | 79.69 | 87.42 | 575 ms | 199 MiB |

## Medians across valid rounds

| engine | rounds | warm TTFT | warm e2e tok/s | warm decode tok/s | footprint peak |
| --- | --- | --- | --- | --- | --- |
| chinecore-metal | 3 | 311 ms | 82.45 | 91.29 | 938 MiB |
| llama-metal | 3 | 287 ms | 79.91 | 87.68 | 199 MiB |

## Paired ratios (ChineCore / llama.cpp, same round)

- warm decode tok/s ratio: median 1.041 over 3 paired rounds (0.891, 1.041, 1.045)
- warm TTFT ratio (higher = slower prefill): median 1.08 over 3 paired rounds

## Vendor tool: pinned llama-bench (own prompt tokens; not identity-matched)

not run

## MLX

available: False — mlx 0.32.4, mlx-lm 0.32.2; mlx-lm cannot load the identical GGUF artifact (NotADirectoryError: [Errno 20] Not a directory: 'artifacts-local/gemma-3-1b-it-Q4_0.gguf/config.json'); an MLX-format conversion would be a different artifact, so no MLX row is recorded
