# Vital render latency

Vital already renders offline. The measured delay was dominated by work around synthesis, including repeated process/plugin startup and Python conversion of JUCE patch-state data.

The state encoder/decoder now uses standard-library base64 and byte translation tables, reversing bit order to preserve JUCE's exact format. It keeps declared-size padding, truncation and invalid-character rejection. Native state readback, per-render plugin hashing, QA and process timeouts still run.

Across the same five training patches, total fresh-process render time fell from **25.04s to 19.80s (20.9% less wall time)**. Mean patch-state conversion fell from **1.450s to 0.0068s**. Every requested patch, read-back native state, renderer provenance and float32 audio hash matched before/after exactly.

| Family | Audio duration | Native synthesis after | Whole call before | Whole call after |
|---|---:|---:|---:|---:|
| wavetable | 4.0s | 0.078s | 4.67s | 4.21s |
| layered | 4.0s | 0.072s | 5.13s | 3.61s |
| filtered | 4.0s | 0.087s | 5.40s | 3.81s |
| modulated | 4.0s | 0.092s | 5.06s | 4.34s |
| chorus | 4.0s | 0.142s | 4.79s | 3.83s |

All 10 native attempts passed. The full data-generation suite plus training stream checks passed **104 local tests**. Dedicated compatibility tests cover every single-byte value, bit boundaries, payloads up to 256 KiB, short/excess data and malformed encodings. GitHub Actions remains disabled.

This is a five-case local Windows CPU measurement, performed serially before/after after other test jobs finished. Individual calls vary, and this does not establish broad throughput or model quality. The small profiler used to find the bottleneck is excluded from these unprofiled phase measurements. No final-test states or audio archive were used.

The remaining substantial work is process/plugin startup and native state loading. Reusing an isolated worker is a separate follow-up: it needs crash/timeout recovery, bounded lifetime and tests for patch-to-patch contamination before becoming the default. Increasing batch or queue size alone cannot remove this cost.

[Aggregate evidence](data/vital_render_latency.json) · [Measurement script](scripts/benchmark_vital_latency.py) · [Summarizer](scripts/summarize_vital_latency.py)
