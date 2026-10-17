# Smaller checkpoints during initial collection

Initial collection repeatedly saved unchanged model weights, frozen frontend weights and the complete run contract. A local profile of the existing 64-example checkpoint measured about 15 MB of serialized run metadata and 23 MB of model weights, separately from optimizer and replay data. The new [storage path](../CHECKPOINT_STORAGE.md) writes that fixed initial state once and keeps a durable feature/cursor snapshot after every acknowledged render. Once initial collection is complete, full checkpoints and inference exports retain their previous self-contained formats.

## Paired storage benchmark

The [plan](data/checkpoint_io_plan.json) and [summary](data/checkpoint_io_summary.json) compare both methods on the same 65 growing replay prefixes, including the empty prefix. The benchmark uses fixed existing weights with an empty optimizer as an I/O fixture; it does not train or select a model. Execution order alternates by prefix. No other Python training or test job was running during this benchmark.

| Measurement | Full snapshot after each prefix | Initial state once plus collection snapshots |
|---|---:|---:|
| Write time | 99.19 seconds | 38.27 seconds |
| Primary serialized bytes | 2,793,863,750 | 322,601,952 |
| Time reduction | — | 61.42% |
| Serialized-byte reduction | — | 88.45% |
| Resolved final state | Identical | Identical |

All seven benchmark gates pass, including the declared minimum reductions of 20% in time and 50% in serialized bytes. The compact method writes one initial file and 65 collection snapshots. Peak process-tree memory is 664.53 MiB against an 8 GiB limit. These are local paired persistence measurements, not a general training-speed forecast or a disk-quota estimate. Backup-copy traffic is excluded from the serialized-byte count. Both arms use the same atomic writer, including the backup flush before promotion.

## Recovery checks

All 50 selected local checks pass, covering accepted and rejected renders, unacknowledged feature failures, an interruption partway through collection, exact resumed model/optimizer/RNG/replay/cursor state, paired feature recovery and existing training/evaluation behavior. Missing or corrupted initial files, altered references, attempted static-state overrides and a substituted trained base fail closed. A failed new-snapshot verification preserves both committed recovery files. An actual older self-contained checkpoint also loads in the benchmark.

The [native integration plan](data/checkpoint_native_plan.json) uses the same frozen combined pool and learning configuration as the earlier pipeline integration. It saves a two-file collection snapshot after 64 examples and before initial metrics or optimizer updates, then compares resumed training with the uninterrupted continuation. Rendering/source/feature implementation fingerprints still match the prior fixed-phase native source qualification; declared differences cover storage integration and the already qualified opt-in sampler. This integration retains the original `state_round_robin` policy.

The [native result](data/checkpoint_native_summary.json) passes all twelve gates. Its learning run ID matches the [earlier combined integration](data/combined_training_summary.json), confirming that the storage representation did not change the learning contract.

| Native integration measurement | Result |
|---|---|
| Acknowledged source attempts | 67: 64 accepted, three expected QA rejections, zero errors |
| Retained distinct training states | 64 |
| Updates | Twelve uninterrupted and twelve after collection-checkpoint recovery |
| Initial/collection/full writes in the uninterrupted run | 1 / 68 / 3; every acknowledged attempt persisted |
| Primary serialized bytes in that run | 567,869,837 |
| Exact recovery | Model, optimizer, Python/NumPy/Torch RNG, batch RNG, replay and cursor |
| New audio/features during recovery | None |
| Frozen frontend and final self-contained checkpoint | Both verified |
| Training segment | 272.15 seconds, including 79.46 seconds of checkpoint work, 63.42 seconds of feature extraction and 7.65 seconds of optimizer work |
| Recovery training segment | 21.23 seconds |
| Peak process-tree memory during the measured training segments | 1,298.22 MiB, below 8 GiB |

Those segment times exclude initial pool validation and resume preflight checks. They are not a matched end-to-end speed comparison with the earlier run: feature and source-processing time also varies, and this run includes a recovery-copy probe. The isolated paired benchmark above establishes the write-time reduction. The native integration establishes durable collection and exact recovery.

Native collection retains the source's requested phases. Exact recovery uses the retained features and makes no new native renders; it does not claim bit-identical regeneration of unacknowledged audio. The separate renderer correctness qualification fixes generator random phases and repeats requests in fresh processes. Model quality, broader source robustness, input layouts, graph/range coverage, data scaling and the final release gates remain open.

Only metadata is committed. Final-test audio remains reserved, Actions stays disabled, and the model is not promoted.
