# Paired synthesis-identity learning

The inverse model can now learn that two performances of the same training patch share a synthesis identity. This directly trains the 128-dimensional representation used by the patch decoder. Model inputs remain audio-derived features only; state and performance IDs select training pairs and never enter the network.

The [first fixed screen](reports/PAIRED_IDENTITY_RESULTS.md) found no held-out benefit: mean native reconstruction distance rose 8.25%, and reserved-performance retrieval fell from 25% to 20%. The paired interval includes zero. The loss remains optional with default weight zero; audio-based learning is next.

## Objective and sampling

`--sampling-policy paired_views` places two different reference performances of each state next to one another in the deterministic render schedule. Each epoch visits every training state once as a pair; two epochs rotate through all four reference performances. The ordinary `pairs` policy still means shuffled individual state/performance combinations.

Paired minibatches contain complete pairs from distinct states. The two normalized representations of a state are positives; other states in the batch are cross-view negatives. The loss averages cross-entropy in both directions over cosine similarities divided by temperature. It is added to the existing masked parameter loss with `--identity-weight`. No additional trainable projection or inference input is introduced.

Weight zero provides a supervised control with exactly the same paired data and minibatch policy. The existing defaults preserve the previous sampler, model weights, inference descriptors and training-contract fields. An identity-enabled run records its objective, weight and temperature in the recovery contract.

The replay stores the actual performance ID for paired views and validates it against each frozen recipe. Pending first views and pairs broken by QA rejection cannot become positive pairs. Recovery preserves the pending view and dedicated minibatch RNG. If a fixed accepted-example budget cannot supply enough complete distinct-state pairs, training stops explicitly.

State identity here is stored patch state, not a learned acoustic-equivalence label. Similar-sounding states can therefore act as negatives. The native reconstruction comparison and representation-spread diagnostics remain necessary when interpreting the objective.

## Bounded feasibility screen

Run from the repository root in the pinned training/native environment:

```sh
python research/training/reports/scripts/run_identity_screen.py \
  --pool research/training/outputs/authored-alpha-final-20261015/pool.json \
  --output research/training/outputs/paired-identity-screen
```

An explicit `--plugin-path` can select the pinned installed Vital binary. The runner writes its complete plan before rendering: source/pool hashes, all 128 training recipe IDs, arm settings, a validation panel and native-render budgets. It requires a new output directory.

The fixed design schedules 64 training states, two reference views each, and a bounded 128-example/8 MiB feature replay. It renders audio during ingestion and discards the waveforms after feature extraction. The audited cache records all scheduled outcomes and retains both views only when both pass QA. A broken pair is excluded without replacement. At least eight qualified states and all five families must remain; this is the minimum needed for the configured batch and coverage. Both comparison arms start fresh from the same qualified features and initialization/minibatch seed. Both use 1,000 updates, 16 views per batch, learning rate 0.001 and four CPU threads. The identity arm adds weight 1.0 at temperature 0.1. The runner checks feature hashes, final minibatch RNG and per-example use counts across arms.

The validation panel contains the first two authored validation states in each of the five alpha families, selected before outcomes. Both models share all sources, fixed-template/retrieval/repeat baselines, and reserved transfer performances. The budget is **128 training attempts + 120 comparison renders = 248**, with no automatic extension. Native rejection and paired-cache exclusions remain in the denominators. Training/evaluation checkpoints and logs remain available if interrupted; the component commands support the existing explicit recovery workflows.

The first feature-bank attempt exposed one clipped view and stopped at its fixed budget. Before either comparison arm was trained, the cache protocol was amended to retain only complete QA-qualified pairs. The already-rendered prefix can be recovered with `--recover-checkpoint FILE --recover-log FILE`; both file hashes enter the new plan. Recovery verifies every scheduled recipe, accepted audio hash, feature tensor and native provenance, then discards source model/optimizer state. It does not repeat the source renders or relax QA. Ordinary training-checkpoint feature seeding retains its previous strict accepted-prefix rule.

`--reuse-training DIRECTORY` can reuse completed arms for a new evaluation attempt. It requires unchanged runtime code, pool, arm settings and panel, records input artifact hashes, and validates the copied checkpoints against their shared feature source before rendering. The orchestration script may change; this allows a runner fix to preserve completed learning work. Existing completed-case evaluation recovery remains available through `compare --resume`.

The primary metric remains equally weighted reference/transfer MR-STFT distance per state. Additional diagnostics compare reference-audio and reserved-transfer-audio representations: bidirectional state retrieval, matched versus different-state cosine similarity, and normalized representation spread. Complete-case denominators are retained. A constant representation has chance retrieval and zero spread; high positive cosine alone is insufficient evidence of identity learning. These diagnostics do not select a release model.

This is one fixed single-seed feasibility screen. Detailed weight tuning, replication and larger learning curves are deferred while the next independent direction—audio-based learning—is implemented.

## Component commands

To train against an existing complete paired feature prefix:

```sh
python research/training/run.py train \
  --pool /path/to/alpha/pool.json --seed-replay /path/to/paired-features/features.pt \
  --output research/training/outputs/paired-identity --mode online --steps 1000 --fresh-examples 126 \
  --replay-capacity 128 --replay-mib 8 --batch-size 16 --render-budget 128 \
  --learning-rate 0.001 --sampling-policy paired_views \
  --identity-weight 1 --identity-temperature 0.1 --checkpoint-every 1000
```

This example uses a qualified 126-view cache; take the fresh-example target from the cache summary for another run. Use weight zero for the matched control. Feature seeding requires the same pool, paired sampler and frozen frontend, and resets learning state. A previous individual-view replay is not silently reinterpreted as paired data.

`compare --identity-diagnostics` records cross-performance representations using the source audio already rendered by the evaluator. `--validation-states FILE` optionally accepts an ordered JSON list of unique validation state IDs; training and test IDs are rejected before rendering. The panel and diagnostics protocol become part of the resumable evaluation plan.

The model and feature replay remain small enough for local CPU execution. Free Colab can be evaluated as an optimizer host using a compatible retained feature prefix. Native Linux rendering still needs separate qualification; this screen makes no Colab-native claim.
