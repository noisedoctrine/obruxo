# Verified expansion of the authored training pool

Completed 2026-10-13. The pipeline compares models trained on different-sized authored pools while keeping the original validation cases and training-retrieval baseline fixed. The expanded pool has **404 training states**, up from 205, within the same nine-control/two-category vocabulary. The contract and pool verification below preserves its original scope; the subsequent [native state-coverage experiment](STATE_COVERAGE_RESULTS.md) trained the expanded pool and completed the fixed-case comparison. Its selected 16K checkpoint improved mean validation distance over the incumbent, but both paired intervals include zero and it wins on only 8/21 cases.

The [machine-readable verification](data/pool_expansion_readiness.json) records pool and manifest identities, preserved evaluation cases, checkpoint checksums, source hashes and memory estimates. The [cached-feature learning curve](CACHED_CURVE_RESULTS.md) remains a historical 205-state control; its selected model is unchanged.

## Population and split preservation

`expand-pilot` regenerates a larger independent authored population using the parent's seed and codec. It verifies that the entire original ordered population, including target values, remains its prefix. Every existing state must retain its exact lineage and split assignment. The expanded manifest must still obey the independent-root split rule. A changed seed, reordered/replaced prefix, moved split, duplicated state, smaller count or count above 4,096 fails validation.

| Population | Training | Validation | Final test | Total |
|---|---:|---:|---:|---:|
| Original | 205 | 21 | 30 | 256 |
| Added | 199 | 23 | 34 | 256 |
| Expanded | 404 | 44 | 64 | 512 |

All 51 original held-out states retain their assignments. The new 23 validation and 34 test states also remain excluded from training. Comparisons continue to use the **original 21 validation cases**, with their original reference and transfer performances. None of the 64 test states has been evaluated in this increment.

The [expanded pool](../outputs/authored-expanded-20261013/pool.json) has ID `40eaadfabd24790025f99ee39bcb40caae1fa611b8c3a8cbab940b52a6acbe86`. Its [expansion declaration](../outputs/authored-expanded-20261013/expansion.json) binds it to parent pool `ffb7b52c9c0a9fb40c7493767b012530da7668c9ca49d0057fdf9ba46798961f`. The parent file is preserved. The ordinary pool format is unchanged, so existing same-pool training, inference bundles and checkpoints remain usable.

Reproduce into a new output directory:

```sh
python research/training/run.py expand-pilot \
  --pool research/training/outputs/authored-pilot-v2-20261005/pool.json \
  --count 512 --output research/training/outputs/expanded-pilot
```

The command writes `pool.json`, `summary.json` and `expansion.json`. It needs no PyTorch, native plugin, audio or GPU. Comparison recomputes the expansion proof from the actual loaded pools; an edited declaration file cannot authorize an incompatible candidate.

## Fixed validation across training pools

`compare --pool` defines the evaluation pool and the pool that must match the retrieval checkpoint. Optional `--checkpoint-pools NAME=PATH` entries identify the actual training pool for each expanded candidate. Every supplied name must match a candidate. Omitted names default to the evaluation pool, and a mismatched checkpoint fails before output creation or rendering.

Each candidate's checkpoint, sampler, replay records, targets and acknowledged cursor are validated against its own training pool. An expanded pool must prove a strict prefix extension of the evaluation pool, preserving all its held-out assignments. All candidates still require identical codec, render, frontend, model and precision contracts, plus the same native renderer identity. Only then can their predictions be scored on shared parent-pool source buffers.

After a fresh expanded-pool training run, compare it using:

```sh
python research/training/run.py compare \
  --pool research/training/outputs/authored-pilot-v2-20261005/pool.json \
  --checkpoints incumbent=research/training/outputs/two-view-train-20261010/checkpoint.pt \
    expanded=research/training/outputs/expanded-training/checkpoint.pt \
  --checkpoint-pools expanded=research/training/outputs/expanded-pilot/pool.json \
  --retrieval-checkpoint research/training/outputs/full-pilot-step4000-20261009/checkpoint.pt \
  --output research/training/outputs/expanded-validation --count 21 --timeout 30
```

The comparison plan records each training pool ID and expansion proof, the explicit candidate order, and every validation state's reference/transfer performance IDs. Resume rejects changed candidate order, pool binding, checkpoints, cases or retrieval data. The selected inference bundle carries the winning checkpoint's **training pool ID**, which may differ from the evaluation pool ID.

New comparisons use `pilot_real_render_comparison_v2`. Historical v1 comparison reports and exports remain valid historical artifacts. Their saved plans lack the new bindings and explicit tie order, so they cannot be resumed with v2; use the corresponding earlier implementation for such a recovery, or create a new comparison output. Existing learning-curve evaluation contracts are unchanged.

## Feature reuse and compute limits

Changing the pool changes its sampler, manifest and recipe identities. Cross-pool `--resume` and `--seed-replay` remain rejected. The first expanded run must initialize fresh learning state and render its training examples on demand. Same-pool feature seeding and interrupted recovery remain supported. This avoids relabeling old feature records under unrelated recipe IDs or treating a new data distribution as an ordinary resume.

The static recipe preflight covers all 404 training states exactly once in the first `state_round_robin` epoch. Every recipe uses a training state and reference performance; none uses held-out membership or a transfer performance. These are planned recipes, not accepted-audio counts. Native QA and actual distinct-state coverage remain mandatory during training.

One feature record per state needs **15,618,640 bytes, about 14.90 MiB**, using the measured 38,660-byte frontend payload per record. This fits a 404-example replay and the existing **32 MiB** feature budget; the implementation's 512-example cap also remains unchanged. Two retained performances for all 404 states would require 808 records and does not fit that cap. The feature estimate excludes model, optimizer, native worker and application memory.

No expanded-pool native training or evaluation run was started. No paid compute, Colab session or audio archive was used. Local CPU is the next execution target; Linux/Colab native qualification remains open.

## Verification and experiment outcome

**108 distinct training/contract/renderer unit tests passed**, across the focused pool/trainer checks and the remaining contract/renderer suite. One new CLI integration test needed a correction to its synthetic renderer fixture and passed on rerun. Ruff passed. The tests exercise preserved population and split assignments, rejected invalid expansion, strict cross-pool seeding/resume, CLI candidate-to-pool bindings, fixed parent validation and retrieval, selected export provenance, unchanged same-pool evaluation, and candidate-order rejection on resume. These integration checks use a synthetic renderer and do not claim native acoustic quality.

Actual-artifact checks also passed: the expanded pool loads and preserves the parent prefix; all 404 first-epoch training recipes are eligible train/reference combinations; all 21 validation recipe IDs and performance pairs match the previous native comparison; the original 205-record retrieval digest is unchanged; and the incumbent and retrieval checkpoints load under their original pool contracts. Their weights and files were not modified.

The declared 205-versus-404-state coverage experiment has now completed. Its [result report](STATE_COVERAGE_RESULTS.md) records the immutable plan, 404/404 accepted training-state/reference coverage, 4K/8K/16K checkpoints, complete 21-case native comparison and selected export. The expanded 16K model leads on mean validation distance, but both paired intervals include zero and it beats the incumbent on only 8 of 21 states. All 64 expanded-pool test states remain reserved. A multi-seed replication is the next step; this single-seed result does not establish broad generalization.
