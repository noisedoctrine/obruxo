# Issue 187 — Stage 1 interim results

**Interim finding:** On the shared retrieval panel, SAME-S currently has higher Recall@1 than Open Small by 1.3 percentage points in the common gallery and 1.6 points in the mixed gallery. Both point estimates trail the carried-forward CLAP and CIR baselines. This is descriptive only: new-encoder confidence intervals have not been computed, and SAME-L retrieval is still running.

Recall@1 is the fraction of queries whose exact target preset ranks first; higher is better. Each condition has 6,450 queries across 129 preset families and 50 distinct MIDI performances, with 129 candidates per gallery. The two conditions total 12,900 episodes, but the episodes share presets and performances and are not 12,900 independent samples.

## Retrieval results

| Encoder | Common gallery Recall@1 | Mixed gallery Recall@1 |
|---|---:|---:|
| CIR — carried-forward baseline | 81.0% [74.7, 86.6] | 80.9% [77.1, 84.4] |
| CLAP — carried-forward baseline | 59.4% [50.8, 67.7] | 55.0% [50.2, 59.8] |
| Vital JEPA V7 — carried-forward baseline | 29.1% [23.0, 35.6] | 29.5% [25.2, 33.8] |
| Stable Audio Open Small | 39.1% (interval pending) | 39.0% (interval pending) |
| SAME-S | 40.4% (interval pending) | 40.6% (interval pending) |
| SAME-L | Pending | Pending |

The baseline intervals are existing 95% intervals clustered over preset family and MIDI performance. The new-encoder cells are point estimates from the same episode panel; paired clustered intervals are not yet available, so the small SAME-S versus Open Small gap is not evidence of a reliable difference. The full per-episode results remain in the active run output and are not committed; the compact metrics and provenance snapshot are in [STAGE1_INTERIM.json](STAGE1_INTERIM.json).

## Codec reconstruction checks

These checks measure codec audio reconstruction on 256 clips from 64 patch roots and 16 MIDI performances. MR-STFT and spectral convergence are lower-is-better reconstruction errors; they do not measure preset retrieval or downstream control recovery.

| Codec | MR-STFT | Spectral convergence |
|---|---:|---:|
| Open Small | 1.055 | 0.223 |
| SAME-S | 2.994 | 0.184 |
| SAME-L | 2.224 | 0.126 |

Stage 1 uses frozen representations and codec reconstruction only; no Vital controls are predicted or edited. The later adapter and inversion stages have not run, so these results do not establish Vital-control recovery or end-to-end quality.

## Execution and reproducibility

At the report snapshot (`2026-11-28 02:54:00 MYT`), the worker was running SAME-L retrieval at 183/6,450 with zero failures. Open Small and SAME-S retrieval and all three codec reconstruction checks were complete. The worker uses the existing Conda `py312` environment and cached per-clip features; the environment was not modified. The full Stage 1 comparison, then Stages 2 and 3, continue automatically. The carried-forward baseline is [Issue 167 v7](../../../modelling/frozen_encoder_retrieval/reports/issue167_v7_results.json).
