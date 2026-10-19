# Local basic spectral helper: frozen experiment card

Declared 2026-10-19, before native collection. This is A1 of the [staged plan](../../modelling/TRAINING_PLAN.md). It tests whether a small forward helper supplies useful sound-loss gradients. It does not train or promote the inverse predictor. All work runs locally; Colab is deferred.

## Fixed scope and data

Use the existing basic pool `outputs/authored-expanded-20261013/pool.json`, preserving its manifest and all test reservations. The machine-readable `plan.json` freezes every selected state, target and MIDI event before collection. Its canonical hash is `b47a1f146816a71b2b1d08a2fbf4af0a1835790c48cce03622aefa9fdea1675e`; source hashes, runtime, codec, phase policy and target/loss definitions are included. Resume requires this exact plan and checksummed committed features.

- Select 128 training states by round-robin over the six unison/filter combinations. Each supplies four generated performances and four paired alternatives: continuous +0.05, continuous -0.05, changed unison and changed filter enable. Clamp and canonicalize all variants; preserve source split ancestry. This plans 1,024 renders.
- Generate held notes, repeated notes, melodies and chords across MIDI 36–84, velocity 64/80/96/112, BPM 90/110/130/150, and varied note lengths/releases. Exact reference/transfer bank recipes are excluded from training.
- Select 12 validation states by the same category-balanced rule. Each uses one reserved reference and one reserved transfer performance, cycling across all four recipes in each bank. This is a development qualification panel, not fresh final confirmation.
- Fix oscillator 1–3 and sampler random phase in separate hashed render copies. Preserve the codec, original presets, other phase settings and seeds. Every native request gets a fresh worker; QA and renderer identity are checked.

Audio is rendered as needed by this pipeline and discarded after measurement during training collection. Retain 16 exact, uniformly sampled valid time frames for each of the three FFT resolutions and 16 RMS-envelope frames. Keep every frequency bin, both channels and each frame's absolute time. This approximates the training objective by sampling time; it does not average frequency/time bins. Qualification uses full recordings and full targets. Evaluation-only source audio is retained for resumable comparisons.

## Model and learning

The helper has **1,413,033 parameters**, below the two-million ceiling. A 7→32→32 note encoder combines pitch, velocity, duration, elapsed/released time and note state. Sum the valid note features; concatenate the 14 basic controls, performance context, query time and resolution indicator. Two 128-unit hidden layers feed full-frequency stereo magnitude heads (514, 2,050 and 8,194 values per queried frame) and a three-value L/R/side envelope head. These are helper measurements, not extra patch prediction heads. The inverse output remains 14 values.

Use float32 CPU, two learning threads, Adam at 0.001, batch eight, gradient-norm cap five, seed zero, 12,000 updates and checkpoints every 250 updates. The loss is the fixed A0 spectral/envelope/side objective. No learned normalization, helper refresh, parameter-loss term, hyperparameter sweep or checkpoint selection. Use the final helper. Freeze its weights before guidance while preserving input gradients.

A 30-update synthetic timing probe took 5.58 seconds with batch eight. One sampled target occupies 688,776 bytes; 512 such targets use about 336 MiB before small record metadata. This is a runtime/storage check, not learning or native-quality evidence. Seven focused tests pass, including exact optimizer/minibatch recovery, changed-feature rejection, gradient connectivity, paired recipe isolation and refusal to load an unqualified helper.

## Native qualification and gates

For each of 24 state/performance pairs, render ten patches: source, identical repeat, fixed imperfect start, continuous plus/minus changes, one categorical alternative, isolated pan change, isolated detune change, continuous-guided result and joint-guided result. This plans **240 qualification requests**, leaving 16 attempts for recovery inside the 256 cap. Balance categorical probes by alternating unison/filter changes. Detune comparisons explicitly exclude inactive single-voice cases and report their count; pan and other comparisons retain all planned cases. Failed or tied cases do not become successes through omission.

The imperfect start is a deterministic ±0.2 continuous perturbation of the source. Guidance receives only its start, target measurements and evaluator MIDI; source parameters do not enter the optimization. Apply 16 Adam updates at 0.03, bounded to ±0.15 from the start and legal continuous limits. Joint updates use straight-through hard Gumbel-softmax with temperature 1→0.5; final patches use raw-score argmax. This tests local direction usefulness, not a deployable inverse model.

All of the following must pass separately for reference and transfer:

- Mean helper forward loss is at most 90% of the best train-only mean/nearest-record/five-neighbor baseline on reference, and no worse than the best baseline on transfer. Baselines interpolate retained exact training frames onto evaluation times; all methods use the same full-target loss.
- Native ranking agreement is at least 70% separately for continuous changes, categorical changes, pan changes and active-detune changes. Report planned denominators and ties; tie tolerance is at least 1e-7 or measured repeat error, whichever is larger.
- Both continuous-only and joint guidance improve native MRSTFT in at least 60% of cases beyond repeat noise, with negative mean changes in both native MRSTFT and the new full spectral objective. Report paired family bootstrap intervals and hard-category changes.
- Every qualification render passes QA and every fixed-phase repeat is sample-exact. At least 90% of training requests are accepted. All resource limits pass.

An unsuccessful helper is exported as unqualified and ordinary inverse-training loading refuses it. Lower helper loss alone cannot pass. Preserve the failed evidence, diagnose it and declare one specific correction before a new bounded experiment; do not extend this run or broaden the patch vocabulary.

## Resource and recovery limits

One native worker, 8 GiB process-tree RAM, four hours elapsed including collection/training/qualification/recovery, at most 2,048 training attempts, 256 qualification attempts and 12,000 committed helper updates. Recovery reuses committed features and resumes exact model/optimizer/RNG state; interrupted native attempts still count. No automatic replacement of rejected training examples.

Replay is bounded by both 512 records and 512 MiB. Persisted features, evaluation tensors and checkpoint backups stay within 2 GiB. Checkpoint source and feature identities must match. Stop on the first reached budget; never silently increase it.

Run locally from the repository with its existing dependencies:

```powershell
python research/training/reports/scripts/run_basic_spectral_helper.py --pool research/training/outputs/authored-expanded-20261013/pool.json --output research/training/outputs/basic-spectral-helper-20261019
```

Use `--resume` only for that same output, frozen source and plan. The run writes its plan before any render. Its final summary records all attempts, resource measurements, qualification gates and failure status. No GitHub Actions or Colab setup is involved.
