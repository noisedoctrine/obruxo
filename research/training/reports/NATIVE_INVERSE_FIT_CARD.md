# Matched basic inverse training with native sound-loss gradients

Declared 2026-10-21 before training. The [native gradient method passed qualification](NATIVE_GRADIENT_RESULTS.md). This card tests whether its measured sound-loss gradients improve the existing small inverse model. It combines the new approximately 32-example fit check and a conditional development comparison in one bounded run. It does not expand the 14-output prediction head.

Frozen plan ID: `1530130a22f4e6b7143a3ba66ab2248a231e20d77e3805b83e960d23f5734365`. Six focused matched-training/inverse tests pass, including exact recovery of both real model/optimizer states, charged update logging, native audit recovery and rejection of incomplete quality evidence. All 64 planned source/performance pairs previously passed QA in the earlier training collection; the new run still renders and verifies them as needed.

## Shared data and models

Use the existing expanded basic pool, selected seed-zero inference bundle and matching 16,000-update training checkpoint. Freeze their byte hashes, model digest, codec, frontend and train-only retrieval memory in `plan.json`. Both arms start from the exact same 355,806-parameter model, with fresh matched Adam states. The incumbent files remain unchanged.

Select 32 category-balanced training states. Generate two distinct training performances per state using the existing deterministic MIDI generator, covering held notes, repeated notes, melodies and chords. Exclude every reserved evaluation MIDI recipe. The inverse input is always audio from the first performance; the training objective alternates between first and second performances across four epochs. Each state receives four updates. True MIDI and source parameters never enter the inverse network. Its frozen Basic Pitch frontend estimates pitch from audio.

Render each source/performance when that state is first used. Retain checksummed audio features and complete spectral targets, discarding training audio. Replay is bounded by both 512 records and 512 MiB. No pre-rendered WAV corpus is required.

## Training and recovery

Each arm takes **128 updates**, batch one, Adam at **0.0001**, float32 CPU, two Torch threads and gradient-norm cap five. The card pins every state selection and performance view. The parameter arm uses the existing masked MSE and categorical cross-entropy. The spectral arm uses only the unchanged A0 spectral/envelope/side objective, with **zero parameter anchor**. Its continuous and category signals come from the qualified finite-difference/conditional-loss estimator, with hard Gumbel-softmax temperature falling linearly from one to 0.5. Final patches use unperturbed argmax.

Checkpoint both models, both optimizers, RNG and trace after every completed matched update. Persist optimizer-attempt charges before execution, and persist each native loss with its exact candidate and request identity. Resume requires the same frozen plan, source, records and original wall-clock deadline. There are at most 132 charged matched optimizer attempts for 128 committed updates. Native request caps can stop recovery sooner; interrupted work is never free. Retain the final checkpoint only for comparison; no sweep, native checkpoint selection, best-of-search or helper refresh.

## Fit gate, then development comparison

After training, audit all 32 training states under both performances. Render the source again and require its audio hash to match initial collection; compare the incumbent, matched parameter arm and spectral arm. Separately for each performance group, the spectral arm must have mean primary loss at most **90% of the incumbent**, no worse than the matched parameter arm, and strictly lower mean legacy MRSTFT than the incumbent. All planned cases and exact source repeats are required.

Only if both fit groups pass, evaluate the first 12 category-balanced validation states under reference and transfer MIDI. These states and MIDI recipes are already development data. Infer each model's patch from reference audio once, then keep that exact patch for transfer. Compare the incumbent, both trained arms, fixed template and nearest-example retrieval from the unchanged train-only memory. Require exact repeats of all 24 source recordings.

Separately on reference and transfer, the spectral candidate must:

- Have mean native MRSTFT at most **90% of both the matched parameter arm and incumbent**.
- Beat both template and train-only retrieval in mean MRSTFT, with primary spectral loss no worse than the matched parameter arm.
- Keep the 90th percentile tail error within **110% of incumbent**. Tail error is the mean L/R difference in RMS over the final codec tail, divided per channel by reference RMS floored at 0.001.
- Complete every planned case with accepted native QA, exact repeats and all resource limits satisfied.

Missing or rejected cases fail. No test state is used. A fit failure stops before validation rather than releasing its data for further tuning. Either result is retained; no automatic extension follows. A pass still requires fresh validation, seed replication, listening and bundle qualification before basic reconstruction is complete.

## Explicit compute revision

The earlier 1,024-request A2 cap assumed cheap gradients from a qualified learned helper. The selected native alternative can need 22 renders for one training example. This card explicitly replaces that request allocation for this run, while retaining the existing four-hour, 8-GiB RAM and 2-GiB archive limits.

| Work | Maximum planned requests |
|---|---:|
| 32 states × two target performances | 64 |
| 128 gradients × 22 queries | 2,816 |
| Training fit audit: 64 source repeats + three models × 64 | 256 |
| Validation: 24 source/repeat pairs + five methods × 24 | 168 |
| Planned total | **3,304** |
| Charged cap including recovery | **3,328** |

Phase caps are 2,896 training requests and 432 audit requests. All failed/interrupted native attempts count. Use one fresh native worker per request, matching qualification. Prior timings imply roughly two hours; the **four-hour elapsed deadline** includes collection, training, audits, checkpointing and recovery. Stop at the first cap reached. This is local CPU work; no Colab or GitHub Actions changes.

Save full phase-controlled receipts, scalar probe measurements, target/feature hashes, checkpoints, per-case scores and diagnostic inference exports. Retain source audit audio for recovery, but not candidate audit audio. Later publication must identify that candidate scores come from the pinned evaluator rather than independent re-scoring of discarded audio. Verify the complete model update sequence offline from the logged native losses before reporting it as reproducible.

```powershell
python research/training/reports/scripts/run_native_inverse_fit.py --pool BASIC_POOL --initial INCUMBENT_EXPORT --retrieval MATCHING_TRAINING_CHECKPOINT --qualification NATIVE_QUALIFICATION_JSON --output NEW_DIRECTORY
```

The first invocation freezes the plan without rendering. Commit the implementation and plan card before adding `--execute`. Use `--execute --resume` only for the same attempted run and its original limits. Diagnostic exports do not promote or replace the incumbent.
