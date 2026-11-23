# Baseline: Qwen3.5-2B-Q8_0.gguf on Apple M3 (8gb)

Label: **locally-calibrated** (measured on this machine; thermals and background load uncontrolled). Not a claim; see `docs/claims.md`.

- macOS 27.0.1 (26A434), power: Battery Power, memory 8589934592 B
- chine `805cbc935a00` (dirty tree: False), llama.cpp b11695 (pinned, scripts/fetch-llama.sh)
- artifact `Qwen3.5-2B-Q8_0.gguf`, 2012012800 B, sha256 `1b04acba824817554f4ce23639bc8495ff70453b8fcb047900c731521021f2c1`
- shape: 512 in / 256 out, 3 repetitions per invocation, context 2048, 3 ABBA rounds; frozen synthetic token ids (standard-v1), identical for both harness engines
- warm = repetitions 2..n of an invocation; cold-load = repetition 1 (includes worker spawn + model load; page cache uncontrolled)

## Runs (ABBA order)

| round | slot | engine | status | reclaimable before | swap delta | warm TTFT | warm e2e tok/s | warm decode tok/s | cold-load TTFT | footprint peak |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | A | chinecore-metal | ok | 3510 MiB | 0 MiB | 585 ms | 37.22 | 40.52 | 1855 ms | 2251 MiB |
| 1 | B | llama-metal | ok | 3993 MiB | 0 MiB | 589 ms | 37.29 | 40.63 | 1607 ms | 248 MiB |
| 2 | A | llama-metal | ok | 4145 MiB | 0 MiB | 588 ms | 37.29 | 40.62 | 954 ms | 248 MiB |
| 2 | B | chinecore-metal | ok | 4179 MiB | 0 MiB | 592 ms | 37.15 | 40.48 | 1804 ms | 2251 MiB |
| 3 | A | chinecore-metal | ok | 4073 MiB | 0 MiB | 583 ms | 37.15 | 40.42 | 1446 ms | 2251 MiB |
| 3 | B | llama-metal | ok | 3965 MiB | 0 MiB | 588 ms | 37.28 | 40.62 | 1208 ms | 248 MiB |

## Medians across valid rounds

| engine | rounds | warm TTFT | warm e2e tok/s | warm decode tok/s | footprint peak |
| --- | --- | --- | --- | --- | --- |
| chinecore-metal | 3 | 585 ms | 37.15 | 40.48 | 2251 MiB |
| llama-metal | 3 | 588 ms | 37.29 | 40.62 | 248 MiB |

## Paired ratios (ChineCore / llama.cpp, same round)

- warm decode tok/s ratio: median 0.996 over 3 paired rounds (0.997, 0.996, 0.995)
- warm TTFT ratio (higher = slower prefill): median 0.99 over 3 paired rounds

## Vendor tool: pinned llama-bench (own prompt tokens; not identity-matched)

| test | tokens/s (avg ± stddev) | backend |
| --- | --- | --- |
| pp512 | 884.04 ± 19.48 | MTL,BLAS |
| tg256 | 40.85 ± 0.09 | MTL,BLAS |
- `/usr/bin/time -l`: 2219769856  maximum resident set size
- `/usr/bin/time -l`: 252037496  peak memory footprint

## MLX

available: False — mlx 0.32.4, mlx-lm 0.32.2; mlx-lm cannot load the identical GGUF artifact (NotADirectoryError: [Errno 20] Not a directory: 'artifacts-local/Qwen3.5-2B-Q8_0.gguf/config.json'); an MLX-format conversion would be a different artifact, so no MLX row is recorded
