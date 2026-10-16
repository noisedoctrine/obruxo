# Learn from candidates checked by Vital

This implements Stage D's direct-render fallback in the [training plan](../modelling/TRAINING_PLAN.md). The forward helper's input gradients failed native qualification; audio-based training remains necessary. Here, actual Vital renders decide whether a proposed patch is a better match.

The pipeline starts from a trained audio-only alpha checkpoint. It selects 32 new training patches outside that checkpoint's retained training states, balanced across the authored synthesis families. Each source uses one of the four reference performances. Validation patches and reserved performances never enter training or target selection.

For each accepted training recording, the frozen source model proposes at most four distinct, valid patches:

- Its ordinary single prediction.
- The nearest patch in the original training-only feature bank.
- One legal alternative from its least decisive active categorical head.
- A bounded change to three active continuous controls.

Proposals use audio-derived features and the model's outputs. They receive no source patch parameters, source IDs or known notes. A fixed case-index seed controls continuous variation. Known training notes enter only native rendering and scoring.

Each candidate is rendered under the source performance. The best measured candidate and the ordinary prediction are rendered a second time. A candidate becomes an auxiliary teaching target only when **its worse score across both renders is at least 5% better than the ordinary prediction's better score across both renders**. A failed draw cannot pass. The ground-truth patch is never inserted into the proposal set. All candidates, scores, duplicates, QA rejections, confirmations and recipe identities remain auditable.

## Matched learning and validation

Two inverse models start from identical source weights with fresh optimizer state. Both receive exactly the same minibatches: half old retained examples and half new source recordings. Both retain source-parameter supervision. The candidate arm adds a weight-0.5 loss toward confirmed alternative targets, scaled by the proportion of confirmed examples in the new half-batch. The control has weight zero. No candidate sound proxy is used.

The fixed first check uses 1,500 updates, batch size 16, learning rate 0.0001 and four CPU threads. Per-record draw counts and minibatch RNG must match between arms. The imported Basic Pitch model stays frozen; inference still consumes one audio clip.

The validation panel is frozen before training collection: ten authored validation patches across the five families, with reference and reserved performances. Both arms share source audio, original training retrieval, fixed-template and repeat-render baselines. Evaluation measures their ordinary top-1 outputs. Best-of-four with known notes is a training/evaluator diagnostic, not shipped audio-only ranking. No release model is promoted from this single-seed comparison.

## Run and recover

```sh
python research/training/run.py candidate-screen \
  --pool /path/to/alpha/pool.json \
  --checkpoint /path/to/qualified-supervised-source/checkpoint.pt \
  --output research/training/outputs/render-selected-screen
```

An optional `--plugin-path` selects the pinned native Vital binary. Output must be a new directory. For recovery, repeat the exact command with `--resume`. The full plan pins code, source checkpoint, source weights, train-only retrieval, target selection, validation panel and compute budgets before rendering.

Training collection allows at most 224 planned native attempts plus 14 recovery attempts. Deduplication, an incumbent winner or a failed source can use fewer calls. Validation allows 120 planned attempts plus 12 recovery attempts. Both ceilings are enforced, including uncommitted renders from an interrupted case. There is no automatic budget extension. Total ceiling: 370 attempts.

Collection commits after complete cases. An interrupted case regenerates transient audio; every original and repeated attempt stays in the append-only log. The retained dataset includes only accepted audio features and hashes, labels, proposed targets, scores and QA history. Raw audio is discarded. Feature storage, including the original bank, is bounded at 8 MiB by default.

Learning checkpoints contain model/optimizer state, minibatch RNG, draw counts, complete verified source lineage and selected training records. Resume rejects changed contracts or records. Exports use the existing audio-only inference format and embed their assets. Evaluation recognizes the new training-checkpoint format and preserves the original retrieval bank; it does not silently treat selected targets as ground truth.

This is a free local CPU workflow. No paid compute, audio archive, final-test use or cloud dependency is introduced. Linux/Colab native rendering and a deployed audio-only candidate ranker remain separate work.
