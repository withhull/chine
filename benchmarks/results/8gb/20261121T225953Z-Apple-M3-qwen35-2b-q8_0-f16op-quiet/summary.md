# Baseline: Qwen3.5-2B-Q8_0.gguf on Apple M3 (8gb)

Label: **locally-calibrated** (measured on this machine; thermals and background load uncontrolled). Not a claim; see `docs/claims.md`.

- macOS 27.0.1 (26A434), power: AC Power, memory 8589934592 B
- tuckmill `2d03884353f6` (dirty tree: False), llama.cpp b11695 (pinned, scripts/fetch-llama.sh)
- artifact `Qwen3.5-2B-Q8_0.gguf`, 2012012800 B, sha256 `1b04acba824817554f4ce23639bc8495ff70453b8fcb047900c731521021f2c1`
- shape: 512 in / 256 out, 3 repetitions per invocation, context 2048, 3 ABBA rounds; frozen synthetic token ids (standard-v1), identical for both harness engines
- warm = repetitions 2..n of an invocation; cold-load = repetition 1 (includes worker spawn + model load; page cache uncontrolled)

## Runs (ABBA order)

| round | slot | engine | status | reclaimable before | swap delta | warm TTFT | warm e2e tok/s | warm decode tok/s | cold-load TTFT | footprint peak |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | A | tuckcore-metal | ok | 3087 MiB | 0 MiB | 602 ms | 36.54 | 39.82 | 1830 ms | 2148 MiB |
| 1 | B | llama-metal | ok | 4051 MiB | 0 MiB | 587 ms | 37.22 | 40.53 | 1570 ms | 247 MiB |
| 2 | A | llama-metal | ok | 3854 MiB | 0 MiB | 588 ms | 37.33 | 40.68 | 958 ms | 248 MiB |
| 2 | B | tuckcore-metal | ok | 4145 MiB | 0 MiB | 601 ms | 36.57 | 39.85 | 1817 ms | 2148 MiB |
| 3 | A | tuckcore-metal | ok | 3903 MiB | 0 MiB | 600 ms | 36.52 | 39.79 | 1746 ms | 2148 MiB |
| 3 | B | llama-metal | ok | 3815 MiB | 0 MiB | 587 ms | 37.28 | 40.60 | 1179 ms | 248 MiB |

## Medians across valid rounds

| engine | rounds | warm TTFT | warm e2e tok/s | warm decode tok/s | footprint peak |
| --- | --- | --- | --- | --- | --- |
| llama-metal | 3 | 587 ms | 37.28 | 40.60 | 248 MiB |
| tuckcore-metal | 3 | 601 ms | 36.54 | 39.82 | 2148 MiB |

## Paired ratios (TuckCore / llama.cpp, same round)

- warm decode tok/s ratio: median 0.980 over 3 paired rounds (0.982, 0.980, 0.980)
- warm TTFT ratio (higher = slower prefill): median 1.02 over 3 paired rounds

## Vendor tool: pinned llama-bench (own prompt tokens; not identity-matched)

not run

## MLX

available: False — mlx 0.32.4, mlx-lm 0.32.2; mlx-lm cannot load the identical GGUF artifact (NotADirectoryError: [Errno 20] Not a directory: 'artifacts-local/Qwen3.5-2B-Q8_0.gguf/config.json'); an MLX-format conversion would be a different artifact, so no MLX row is recorded
