# Baseline: gemma-3-1b-it-Q4_0.gguf on Apple M3 (8gb)

Label: **locally-calibrated** (measured on this machine; thermals and background load uncontrolled). Not a claim; see `docs/claims.md`.

- macOS 27.0.1 (26A434), power: Battery Power, memory 8589934592 B
- chine `12e52d494837` (dirty tree: False), llama.cpp b11695 (pinned, scripts/fetch-llama.sh)
- artifact `gemma-3-1b-it-Q4_0.gguf`, 721918496 B, sha256 `27ee88e03be02e9ba73def9a819d570d8ad73716e50769e87f374ae394b0276e`
- shape: 512 in / 256 out, 3 repetitions per invocation, context 2048, 3 ABBA rounds; frozen synthetic token ids (standard-v1), identical for both harness engines
- warm = repetitions 2..n of an invocation; cold-load = repetition 1 (includes worker spawn + model load; page cache uncontrolled)

## Runs (ABBA order)

| round | slot | engine | status | reclaimable before | swap delta | warm TTFT | warm e2e tok/s | warm decode tok/s | cold-load TTFT | footprint peak |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | A | chinecore-metal | ok | 3720 MiB | 0 MiB | 312 ms | 82.59 | 91.48 | 959 ms | 940 MiB |
| 1 | B | llama-metal | ok | 4221 MiB | 0 MiB | 286 ms | 91.79 | 101.90 | 619 ms | 199 MiB |
| 2 | A | llama-metal | ok | 3716 MiB | 0 MiB | 286 ms | 91.68 | 101.77 | 561 ms | 200 MiB |
| 2 | B | chinecore-metal | ok | 3601 MiB | 0 MiB | 311 ms | 82.85 | 91.79 | 702 ms | 938 MiB |
| 3 | A | chinecore-metal | ok | 4192 MiB | 0 MiB | 311 ms | 82.80 | 91.73 | 710 ms | 938 MiB |
| 3 | B | llama-metal | ok | 4205 MiB | 0 MiB | 288 ms | 79.26 | 86.93 | 569 ms | 200 MiB |

## Medians across valid rounds

| engine | rounds | warm TTFT | warm e2e tok/s | warm decode tok/s | footprint peak |
| --- | --- | --- | --- | --- | --- |
| chinecore-metal | 3 | 311 ms | 82.80 | 91.73 | 938 MiB |
| llama-metal | 3 | 286 ms | 91.68 | 101.77 | 200 MiB |

## Paired ratios (ChineCore / llama.cpp, same round)

- warm decode tok/s ratio: median 0.902 over 3 paired rounds (0.898, 0.902, 1.055)
- warm TTFT ratio (higher = slower prefill): median 1.09 over 3 paired rounds

## Vendor tool: pinned llama-bench (own prompt tokens; not identity-matched)

not run

## MLX

available: False — mlx 0.32.4, mlx-lm 0.32.2; mlx-lm cannot load the identical GGUF artifact (NotADirectoryError: [Errno 20] Not a directory: 'artifacts-local/gemma-3-1b-it-Q4_0.gguf/config.json'); an MLX-format conversion would be a different artifact, so no MLX row is recorded
