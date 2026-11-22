# Baseline: Qwen3-1.7B-Q8_0.gguf on Apple M3 (8gb)

Label: **locally-calibrated** (measured on this machine; thermals and background load uncontrolled). Not a claim; see `docs/claims.md`.

- macOS 27.0.1 (26A434), power: AC Power, memory 8589934592 B
- chine `2a21431b5f6d` (dirty tree: False), llama.cpp b11695 (pinned, scripts/fetch-llama.sh)
- artifact `Qwen3-1.7B-Q8_0.gguf`, 1834426016 B, sha256 `061b54daade076b5d3362dac252678d17da8c68f07560be70818cace6590cb1a`
- shape: 512 in / 256 out, 3 repetitions per invocation, context 2048, 3 ABBA rounds; frozen synthetic token ids (standard-v1), identical for both harness engines
- warm = repetitions 2..n of an invocation; cold-load = repetition 1 (includes worker spawn + model load; page cache uncontrolled)

## Runs (ABBA order)

| round | slot | engine | status | reclaimable before | swap delta | warm TTFT | warm e2e tok/s | warm decode tok/s | cold-load TTFT | footprint peak |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | A | chinecore-metal | ok | 3019 MiB | 0 MiB | 589 ms | 40.89 | 44.96 | 1698 ms | 2233 MiB |
| 1 | B | llama-metal | ok | 4093 MiB | 0 MiB | 559 ms | 41.44 | 45.38 | 1450 ms | 381 MiB |
| 2 | A | llama-metal | ok | 3776 MiB | 0 MiB | 559 ms | 41.44 | 45.39 | 843 ms | 380 MiB |
| 2 | B | chinecore-metal | ok | 3803 MiB | 0 MiB | 588 ms | 40.83 | 44.88 | 1624 ms | 2233 MiB |
| 3 | A | chinecore-metal | ok | 4009 MiB | 0 MiB | 585 ms | 40.89 | 44.93 | 1361 ms | 2233 MiB |
| 3 | B | llama-metal | ok | 4054 MiB | 0 MiB | 559 ms | 41.43 | 45.37 | 999 ms | 380 MiB |

## Medians across valid rounds

| engine | rounds | warm TTFT | warm e2e tok/s | warm decode tok/s | footprint peak |
| --- | --- | --- | --- | --- | --- |
| chinecore-metal | 3 | 588 ms | 40.89 | 44.93 | 2233 MiB |
| llama-metal | 3 | 559 ms | 41.44 | 45.38 | 380 MiB |

## Paired ratios (ChineCore / llama.cpp, same round)

- warm decode tok/s ratio: median 0.990 over 3 paired rounds (0.991, 0.989, 0.990)
- warm TTFT ratio (higher = slower prefill): median 1.05 over 3 paired rounds

## Vendor tool: pinned llama-bench (own prompt tokens; not identity-matched)

not run

## MLX

available: False — mlx 0.32.4, mlx-lm 0.32.2; mlx-lm cannot load the identical GGUF artifact (NotADirectoryError: [Errno 20] Not a directory: 'artifacts-local/Qwen3-1.7B-Q8_0.gguf/config.json'); an MLX-format conversion would be a different artifact, so no MLX row is recorded
