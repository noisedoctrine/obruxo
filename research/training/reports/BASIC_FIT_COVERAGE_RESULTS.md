# Sound fine-tuning covers a small, uneven part of the existing training pool

Examined 2026-10-21 using the [completed calibrated comparison](CALIBRATED_NATIVE_INVERSE_RESULTS.md). Its sound-training updates use **32 of 404 training states (7.92%)**, with five or six examples per unison/filter combination. The historical starting model was parameter-trained on the 404-state pool; this diagnosis concerns subsequent sound fine-tuning, not everything the model has previously seen.

[Reproducible evidence](data/basic_fit_coverage_summary.json) · Parent plan `b93183419006684feaa0dd593842a1b27511528bda2f9c6c21a4dd18dddb891b`

The saved measurements show a generalization gap and uneven coverage. They **do not prove that coverage is the sole cause**, or that increasing it will pass the quality gate. No new audio, optimizer updates or final-test scoring is performed.

## Coverage

The diagnosis verifies parent file hashes and reconstructs all 88 published native cases from actual receipts. Pool/codec identities, source targets, split membership, complete reference/transfer pairs, exact repeats and reuse of one predicted patch across performances are checked before statistics. Only training targets and the already-used development population enter the analysis; test assignments remain reserved.

Category balancing does not ensure continuous-control coverage. In equal-width fifths of each normalized range:

| Control region | Fitted sound-training states | All existing training states |
|---|---:|---:|
| Attack, highest fifth | 1 / 32 (3.13%) | 84 / 404 (20.79%) |
| Attack, fourth fifth | 14 / 32 (43.75%) | 90 / 404 (22.28%) |
| Sustain, highest fifth | 13 / 32 (40.63%) | 89 / 404 (22.03%) |
| Release, second fifth | 2 / 32 (6.25%) | 88 / 404 (21.78%) |

The evidence includes all nine controls and all five bins, not just these examples. Inactive detune/filter controls are excluded using source activity masks.

For the 12 development states, mean distance to the nearest fitted training target is **0.30243**, versus **0.21750** using the complete training pool. Distance is normalized control RMSE over source-active controls within identical source categories. It is **not audio similarity, a deployable retrieval method or a perceptual score**. Development targets must not guide the next training-state selection.

## Held-out sound gains vary across categories

Each row has only two development states. Positive values mean lower mean native MRSTFT than the historical starting model; negative values mean worse.

| Source voices | Filter | Reference gain | Transfer gain |
|---|---|---:|---:|
| 1 | Off | +3.21% | −2.78% |
| 1 | On | −20.96% | −10.63% |
| 2 | Off | +7.00% | +4.74% |
| 2 | On | +16.32% | +15.46% |
| 4 | Off | +17.71% | +10.98% |
| 4 | On | +3.73% | −5.15% |

Every category improves in mean on the fitted population under both performances. Small development groups do not support a reliable ranking of category difficulty or a new acceptance rule. The aggregate result remains **7.23% reference / 4.24% transfer**, below 10%.

## Parameter differences are supporting diagnostics

These mean absolute errors use normalized stored-control units, source-active values and one prediction per state. They do not measure audible importance or require recovery of a unique source patch.

| Control | Development states | Initial model | Sound-trained model |
|---|---:|---:|---:|
| Oscillator level | 12 | 0.1936 | 0.2085 |
| Pan | 12 | 0.0500 | 0.0406 |
| Unison detune | 8 | 0.3138 | 0.2726 |
| Filter cutoff | 6 | 0.0706 | 0.0960 |
| Filter resonance | 6 | 0.3104 | 0.3026 |
| Attack | 12 | 0.2450 | 0.2660 |
| Decay | 12 | 0.4583 | 0.4950 |
| Sustain | 12 | 0.2472 | 0.2662 |
| Release | 12 | 0.3021 | 0.3328 |

The sound arm lowers these errors on fitted states for every listed control, but many rise on development states. Sound errors still determine success; this does not justify returning to parameter loss or adding outputs.

## Next controlled comparison

Freeze a broader **64-state** sound-training selection from the same training pool: retain the original 32 states and add 32 using training-only category balance and coverage of active controls. Preserve all split assignments. Select no state using development errors or target distances. Keep the same 14 outputs, historical initial weights, fixed scales, optimizer and **128 updates per arm**; two passes over 64 states replace four passes over 32.

Before execution, pin the selection rule, actual states/MIDI, increased target/audit request allocation and archive bound in a new card. Keep original improvement thresholds and separately check the retained 32-state population so new cases cannot dilute its fit requirement. Render targets on demand. This is a new controlled comparison, not an extension or second seed of the failed-quality run.

This analysis takes **0.087 seconds** after imports, with **zero new renders or optimizer updates**. Four tests cover reserved membership, complete pairs, fixed cross-performance predictions and inactive-control masking. Ruff and whitespace checks pass. This is descriptive evidence, not a quality qualification. A3 confirmation, actual listening, bundle checks and later bounded capability stages remain pending. Colab stays deferred and GitHub Actions disabled.
