# Fixed score scaling restores the measured category-learning signal

Completed 2026-10-21 under the [frozen diagnosis card](CATEGORY_SIGNAL_CARD.md). The two existing category heads receive weak derivatives because their raw scores are often highly separated. Dividing each head by a fixed training-derived scale restores its gradient without changing any initial hard patch. This passes the bounded derivative check, **not** the model-quality gate.

[Reconstructed evidence](data/category_signal_summary.json) · Frozen source `03661ec` · Plan `39f32dd27c91794b566a6cf8658ca38e0515760660e747c1d1e94fb679f5f199`

| Existing head | Fixed scale | Original median score gradient | Scaled median gradient | Median gain | Responsive exact-zero steps |
|---|---:|---:|---:|---:|---:|
| Unison voices | 20.31772 | 8.53e-7 | 9.09e-4 | **1,066×** | 1 → 0 |
| Filter on/off | 20.33763 | 1.11e-16 | 2.06e-3 | **18.56 trillion×** | 8 → 0 |

Both heads exceed the predeclared 10× median-gain threshold, with finite measurements at all 128 committed points. The median raw top-two score gaps are 9.92 and 26.94. A recorded native alternative has lower loss on 88/128 voice-count points and 25/128 filter points, so weak gradients cannot be explained solely by the current choice always being best. These are fixed-control counterfactuals, not a source-category accuracy score or a guarantee that simultaneous changes help.

Each constant uses only the frozen starting model's outputs on the 32 training-reference feature records: float64 RMS after removing each row's common class offset, floored at one. Source labels and validation scores do not enter calibration. All 32 initial hard targets and decoded preset bytes stay identical; continuous predictions are untouched. There are still **14 dense outputs**. The deployed selection remains raw-score argmax.

The observer reproduces all 256 original Gumbel choices from separate local generators, with the same noise and temperatures for the derivative comparisons. The measured hard points and conditional native losses stay fixed. It consumes none of the actual sampler RNG. The complete historical 128-update sequence per arm, both optimizer states, Gumbel RNG, archived interruption checkpoint and all native audit results reproduce the published parent report unchanged. The publication audit independently checks source/input hashes, reconstructs every saved derivative and recomputes scales, denominators and gates.

The check takes **37.60 seconds**, peaks at **809,132,032 bytes (0.75 GiB)** process-tree RAM and uses **zero new native requests or fitting updates**. It performs 256 replayed optimizer updates and reads the parent's previously costed measurements. New evidence fits below the 32-MiB cap. Three focused scaling/gradient/contract tests pass; the full observed historical replay is the real-data integration check. Ruff and whitespace checks pass.

Next, integrate the fixed scales into both matched training arms, retain raw-score production export and save calibration identity in recovery metadata. Freeze a fresh costed comparison from the same historical initial weights with new optimizers. Only matching, independently verified settled-renderer measurements may be reused. The held-out 10% improvement, tail, confirmation, seed, listening and bundle requirements remain unchanged. No full training run is launched by this diagnosis, no model is promoted, and wider controls remain deferred.
