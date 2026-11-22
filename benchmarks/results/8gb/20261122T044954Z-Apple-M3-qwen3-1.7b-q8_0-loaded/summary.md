# Baseline: Qwen3-1.7B-Q8_0.gguf on Apple M3 (8gb)

Label: **locally-calibrated** (measured on this machine; thermals and background load uncontrolled). Not a claim; see `docs/claims.md`.

- macOS 27.0.1 (26A434), power: AC Power, memory 8589934592 B
- chine `dcce36f3e6bb` (dirty tree: False), llama.cpp b11695 (pinned, scripts/fetch-llama.sh)
- artifact `Qwen3-1.7B-Q8_0.gguf`, 1834426016 B, sha256 `061b54daade076b5d3362dac252678d17da8c68f07560be70818cace6590cb1a`
- shape: 512 in / 256 out, 3 repetitions per invocation, context 2048, 3 ABBA rounds; frozen synthetic token ids (standard-v1), identical for both harness engines
- warm = repetitions 2..n of an invocation; cold-load = repetition 1 (includes worker spawn + model load; page cache uncontrolled)

## Runs (ABBA order)

| round | slot | engine | status | reclaimable before | swap delta | warm TTFT | warm e2e tok/s | warm decode tok/s | cold-load TTFT | footprint peak |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | A | chinecore-metal | ok | 2337 MiB | -32 MiB | 786 ms | 31.14 | 34.30 | 2665 ms | 2233 MiB |
| 1 | B | llama-metal | ok | 3147 MiB | -24 MiB | 855 ms | 28.54 | 31.43 | 2293 ms | 382 MiB |
| 2 | A | llama-metal | ok | 2896 MiB | -32 MiB | 1349 ms | 20.55 | 22.94 | 1307 ms | 381 MiB |
| 2 | B | chinecore-metal | discarded-swap | 3023 MiB | 275 MiB | n/a | n/a | n/a | n/a | n/a |
| 2 | B | chinecore-metal | ok | 2958 MiB | -24 MiB | 1225 ms | 24.98 | 28.26 | 2258 ms | 2233 MiB |
| 3 | A | chinecore-metal | ok | 3187 MiB | -32 MiB | 1387 ms | 24.40 | 28.02 | 2659 ms | 2233 MiB |
| 3 | B | llama-metal | ok | 3273 MiB | -8 MiB | 1277 ms | 23.11 | 26.02 | 2173 ms | 381 MiB |

## Medians across valid rounds

| engine | rounds | warm TTFT | warm e2e tok/s | warm decode tok/s | footprint peak |
| --- | --- | --- | --- | --- | --- |
| chinecore-metal | 3 | 1225 ms | 24.98 | 28.26 | 2233 MiB |
| llama-metal | 3 | 1277 ms | 23.11 | 26.02 | 381 MiB |

## Paired ratios (ChineCore / llama.cpp, same round)

- warm decode tok/s ratio: median 1.091 over 3 paired rounds (1.091, 1.232, 1.077)
- warm TTFT ratio (higher = slower prefill): median 0.92 over 3 paired rounds

## Vendor tool: pinned llama-bench (own prompt tokens; not identity-matched)

not run

## MLX

available: False — mlx 0.32.4, mlx-lm 0.32.2; mlx-lm cannot load the identical GGUF artifact (NotADirectoryError: [Errno 20] Not a directory: 'artifacts-local/Qwen3-1.7B-Q8_0.gguf/config.json'); an MLX-format conversion would be a different artifact, so no MLX row is recorded
