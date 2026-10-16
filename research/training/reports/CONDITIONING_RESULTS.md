# Performance-conditioning comparison

The attention variant changed mean native sound error by **+1.96%** against the original feature concatenation model. This single-seed ten-patch alpha check does not demonstrate a benefit, so concatenation remains the default. Both architectures remain available for broader coverage work; no release model is promoted.

Both models used the same frozen 126-view paired feature cache, labels, shared initial weights, initial predictions, minibatches and per-example draw counts for 1,000 updates. No new training audio was rendered. The attention path adds 70,465 parameters and begins with zero residual gain; this compares practical architectures, not attention independently of extra capacity.

| Architecture | Parameters | Mean reference/transfer MR-STFT |
|---|---:|---:|
| concat | 359676 | 1.7098 |
| cross_attention | 430141 | 1.7433 |

Attention-minus-concatenation paired mean: +0.0335; descriptive 95% state-bootstrap interval: [-0.060470486246049504, 0.11409592588121671]. Wins: 3/10; ties: 0. Reference and reserved transfer performances remain together in each resampling unit. These development states have been reused and are not a final test.

All 120/120 native validation attempts passed. The learned residual gain was 0.000429. The full local renderer/training suite passed 264 tests, including common initialization, temporal feature layout, gradients, architecture guards, exact recovery and audio-only export/load behavior. Ruff/diff checks passed. GitHub Actions stays disabled.

Compute was local CPU float32 with four threads; independent checks overlapped training and validation, so timings are not a dedicated throughput benchmark. Raw training audio was not archived, and final-test states remain reserved.

This completes the planned initial architecture comparison. Further narrow tuning is deferred while canonical assets, sampler/voice behavior, LFOs, effect order and routing coverage are expanded, followed by measured data scaling and combined training.

[Architecture and protocol](../CONDITIONING.md) · [Frozen plan](data/conditioning_plan.json) · [Aggregate evidence](data/conditioning_summary.json)
