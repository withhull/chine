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
| 1 | A | chinecore-metal | ok | 3063 MiB | 0 MiB | 1770 ms | 14.85 | 16.48 | 3738 ms | 940 MiB |
| 1 | B | llama-metal | ok | 3434 MiB | 0 MiB | 1522 ms | 16.94 | 18.77 | 1606 ms | 198 MiB |
| 2 | A | llama-metal | ok | 3310 MiB | 0 MiB | 1576 ms | 16.94 | 18.84 | 1140 ms | 199 MiB |
| 2 | B | chinecore-metal | ok | 3284 MiB | 0 MiB | 1771 ms | 14.92 | 16.57 | 1849 ms | 938 MiB |
| 3 | A | chinecore-metal | ok | 3521 MiB | 0 MiB | 1786 ms | 14.38 | 15.92 | 1182 ms | 938 MiB |
| 3 | B | llama-metal | ok | 3583 MiB | 0 MiB | 1567 ms | 16.87 | 18.74 | 1146 ms | 199 MiB |

## Medians across valid rounds

| engine | rounds | warm TTFT | warm e2e tok/s | warm decode tok/s | footprint peak |
| --- | --- | --- | --- | --- | --- |
| chinecore-metal | 3 | 1771 ms | 14.85 | 16.48 | 938 MiB |
| llama-metal | 3 | 1567 ms | 16.94 | 18.77 | 199 MiB |

## Paired ratios (ChineCore / llama.cpp, same round)

- warm decode tok/s ratio: median 0.878 over 3 paired rounds (0.878, 0.879, 0.850)
- warm TTFT ratio (higher = slower prefill): median 1.14 over 3 paired rounds

## Vendor tool: pinned llama-bench (own prompt tokens; not identity-matched)

not run

## MLX

available: False — mlx 0.32.4, mlx-lm 0.32.2; mlx-lm cannot load the identical GGUF artifact (NotADirectoryError: [Errno 20] Not a directory: 'artifacts-local/gemma-3-1b-it-Q4_0.gguf/config.json'); an MLX-format conversion would be a different artifact, so no MLX row is recorded
