# Local A1 correction: learn control-change effects

Declared 2026-10-20 before collection. The [first helper](BASIC_SPECTRAL_HELPER_RESULTS.md) predicted held-out spectra better than simple baselines but suggested misleading improvements in 21/48 guided trials. This experiment targets its learned response to control changes. It keeps the 14-output patch vocabulary, 1,413,033-parameter helper architecture and native admission gates. It is a feasibility correction, not an inverse-model training run.

## Frozen change

Train on aligned original/changed pairs and explicitly penalize incorrect signed sound changes. The prior data contained paired performances but changed all continuous controls together and sampled different times for each recording. It therefore did not directly supervise individual control responses at matching audio moments.

Keep the same 128 source training states, source ancestry and four generated MIDI performances per state. Pair each original with one alternative: two isolated active continuous controls, one unison change and one filter-enable change. Cycle the continuous coordinate across the nine controls, skipping inactive controls and clamping ±0.05 changes. This gives 512 planned pairs / 1,024 native training requests. Isolated coverage is 29 level, 29 pan, 21 detune, 23 cutoff, 14 resonance, 56 attack, 28 decay, 28 sustain and 28 release changes. Native measurements determine the effect; parameter-distance accuracy is not the objective.

Both members of a pair use identical MIDI and identical sampled absolute times: 16 frames at each full-frequency FFT resolution and 16 L/R/side RMS frames. Pair ancestry, performance, lengths and times are checked before each update. Both records must pass native QA; incomplete pairs are logged and excluded together, with at least 90% of the 512 planned pairs required before training. No replacement requests are added automatically.

Use four pairs (eight recordings) per update. Retain the existing absolute spectral/envelope objective and add **0.25 × paired-change loss**. For each measurement, compute `log1p(value / 0.0001)` and subtract original from changed. Compare predicted and native signed differences by mean absolute error, divided by the native difference's mean absolute size with floor 0.05. Average the three spectral resolutions; add 0.1 each for L/R envelope and side-envelope terms. This distinguishes the direction of a change while retaining absolute-spectrum supervision. The weight and floors are fixed; no sweep or validation-based tuning.

The model is reinitialized with seed zero and trained for 12,000 updates, Adam 0.001, gradient-norm cap five, float32 CPU/two threads. Checkpoint every 250 updates, retaining optimizer and pair-selection RNG for exact recovery. Use the final checkpoint. The signed-change objective and independent/aligned pair construction are one declared learning intervention; this is not a loss-only ablation against identical old data.

## Identity, qualification and budget

Canonical plan hash: `3ab7769e6b9c51b606005eb7b06609715f1da642844928e970915901b15a11bf`. The plan includes all source states/targets/MIDI, source-module hashes, paired-objective specification, codec and runtime. It additionally pins the renderer source containing the [exact-frame repair](EXACT_RENDER_FRAMES.md). Historical receipts/checkpoints are not rewritten; reproducing the original A1 implementation uses commit `1963a0a`.

Keep the prior 12 validation states and reference/transfer recipes for **development qualification**, with all ten native methods per state/performance: 240 planned requests, 16 recovery attempts reserved. Full recordings and full temporal targets are scored. All original thresholds remain unchanged: baseline prediction margins, ≥70% ranking agreement for continuous/category/pan/active-detune probes, ≥60% real improvements and negative mean native/spectral changes for both continuous-only and joint guidance, separately on reference and transfer. All qualification requests must pass QA and all 24 phase-fixed repeats must be exact. A passing result would still need independent confirmation and listening later; this reused panel is not a fresh final test.

This is a new local run with its own unchanged ceilings: 2,048 training attempts, 256 qualification attempts, 12,000 committed helper updates, four hours elapsed, 8 GiB process-tree RAM, replay ≤512 records and ≤512 MiB, and feature/evaluation/checkpoint archive ≤2 GiB. Failed attempts and recovery count within these limits. No helper refresh, bigger model, broader patch scope or Colab work. No permission to bypass a failed qualification gate.

Five paired-objective tests pass, alongside the seven existing helper tests. They check signed-response sensitivity and gradients, exact pair timing/ancestry, single active-control perturbations, refusal of insufficient complete pairs, and exact interruption/recovery. The collection test verifies stored pair times through replay. Ruff passes. The renderer repair separately passes 33 focused tests and all 19 native post-fix checks.

```powershell
python research/training/reports/scripts/run_basic_spectral_helper.py --paired-changes --pool research/training/outputs/authored-expanded-20261013/pool.json --output research/training/outputs/paired-spectral-helper-20261020
```

Use `--resume --paired-changes` only with the same output and frozen source. If qualification fails again, preserve and diagnose it before selecting another mechanism; do not automatically continue increasing capacity, updates or budgets.
