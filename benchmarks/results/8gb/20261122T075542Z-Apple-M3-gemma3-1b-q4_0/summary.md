# Baseline: gemma-3-1b-it-Q4_0.gguf on Apple M3 (8gb)

Label: **locally-calibrated** (measured on this machine; thermals and background load uncontrolled). Not a claim; see `docs/claims.md`.

- macOS 27.0.1 (26A434), power: Battery Power, memory 8589934592 B
- chine `6ac210d03385` (dirty tree: False), llama.cpp b11695 (pinned, scripts/fetch-llama.sh)
- artifact `gemma-3-1b-it-Q4_0.gguf`, 721918496 B, sha256 `27ee88e03be02e9ba73def9a819d570d8ad73716e50769e87f374ae394b0276e`
- shape: 512 in / 256 out, 3 repetitions per invocation, context 2048, 3 ABBA rounds; frozen synthetic token ids (standard-v1), identical for both harness engines
- warm = repetitions 2..n of an invocation; cold-load = repetition 1 (includes worker spawn + model load; page cache uncontrolled)

## Runs (ABBA order)

| round | slot | engine | status | reclaimable before | swap delta | warm TTFT | warm e2e tok/s | warm decode tok/s | cold-load TTFT | footprint peak |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | A | chinecore-metal | ok | 3037 MiB | 0 MiB | 313 ms | 78.27 | 86.21 | 786 ms | 938 MiB |
| 1 | B | llama-metal | ok | 3410 MiB | 0 MiB | 287 ms | 91.93 | 102.11 | 666 ms | 198 MiB |
| 2 | A | llama-metal | ok | 3484 MiB | 0 MiB | 286 ms | 92.29 | 102.53 | 561 ms | 200 MiB |
| 2 | B | chinecore-metal | ok | 3473 MiB | 0 MiB | 312 ms | 78.44 | 86.39 | 713 ms | 938 MiB |
| 3 | A | chinecore-metal | ok | 3733 MiB | 0 MiB | 312 ms | 78.47 | 86.43 | 709 ms | 938 MiB |
| 3 | B | llama-metal | ok | 3767 MiB | 0 MiB | 286 ms | 91.98 | 102.15 | 570 ms | 199 MiB |

## Medians across valid rounds

| engine | rounds | warm TTFT | warm e2e tok/s | warm decode tok/s | footprint peak |
| --- | --- | --- | --- | --- | --- |
| chinecore-metal | 3 | 312 ms | 78.44 | 86.39 | 938 MiB |
| llama-metal | 3 | 286 ms | 91.98 | 102.15 | 199 MiB |

## Paired ratios (ChineCore / llama.cpp, same round)

- warm decode tok/s ratio: median 0.844 over 3 paired rounds (0.844, 0.843, 0.846)
- warm TTFT ratio (higher = slower prefill): median 1.09 over 3 paired rounds

## Vendor tool: pinned llama-bench (own prompt tokens; not identity-matched)

not run

## MLX

available: False — mlx 0.32.4, mlx-lm 0.32.2; mlx-lm cannot load the identical GGUF artifact (NotADirectoryError: [Errno 20] Not a directory: 'artifacts-local/gemma-3-1b-it-Q4_0.gguf/config.json'); an MLX-format conversion would be a different artifact, so no MLX row is recorded
