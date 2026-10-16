# Audio-only candidate ranking

Selecting a patch using only the input recording changed mean native reference/transfer error by **-10.64%** against ordinary prediction in this fixed ten-patch alpha check. The saved candidate bundle is **qualified for this bounded optional path**. This does not promote a finished release model.

## What was measured

The public ranker inferred notes from the input audio, proposed at most four legal patches and rendered them under the same inferred performance. Ground-truth MIDI and source settings never entered selection. Evaluation then rendered the chosen patch under known reference and reserved transfer performances. Evaluator-only best-of-four is an upper bound, reported separately from the deployed choice.

| Method | Scored/planned states | Mean reference/transfer MR-STFT |
|---|---:|---:|
| deployed | 10/10 | 1.6002 |
| model | 10/10 | 1.7908 |
| oracle_best_four | 10/10 | 1.5686 |
| source_repeat | 10/10 | 0.0000 |
| train_retrieval | 10/10 | 2.1386 |

Deployed-minus-model paired mean: -0.1905; descriptive 95% state-bootstrap interval: [-0.3298890538917233, -0.06266083953281244]. Wins: 6/10; ties: 2. The two performances of each patch stay together. These are reused development states, not a fresh final test.

Ranked coverage: 100.0%. Model-warm ranking p95: 15.99s; bundle cold load: 0.72s. Each candidate render still starts a fresh native process. Accepted native attempts: 159/160. Sampled process-tree peak: 848.0 MiB. Elapsed check time excludes initial bundle creation and includes partial recovery when present.

Independent local checks overlapped this CPU run, so these times describe that workload rather than an idle-machine throughput benchmark. No audio archive, test-split evaluation or paid compute was used. The ordinary top-1 path remains available. Inferred notes and heuristic velocities can limit ranking quality; diagnostic best-of-four must not be mistaken for deployed performance.

Only 9/10 evaluator candidate panels completed every proposed patch under both performances. The one failed attempt was a transfer evaluation of a categorical alternative: a RenderTimeout after a long host scheduling pause, with Windows Modern Standby observed. It remains counted. The chosen deployed patch and ordinary-model comparison were fully scored on all ten states. The diagnostic oracle uses accepted candidates, so it is not a complete four-way ceiling for the affected case.

Validation: the training suite passed 182 tests; the renderer/stream suite passed 104 tests. Counts describe separate overlapping suites. Dedicated checks cover audio-only note estimation, four-render caps, invalid/no-note fallbacks, mono WAV handling, frontend equivalence, state/identity rejection, qualification guards and recovery caps. Ruff/diff checks pass; GitHub Actions stays disabled.

[Usage and gates](../AUDIO_CANDIDATES.md) · [Frozen plan](data/audio_ranking_plan.json) · [Aggregate evidence](data/audio_ranking_summary.json)
