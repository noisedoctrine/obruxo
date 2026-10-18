# Longer observations for slow attacks and releases

The original corpus reader found six eligible training states whose delay-plus-attack exceeds the current 1.5-second reference maximum. Their maximum span is 32.0003 seconds. Some also have long amplitude releases and effect tails. These are valid patches; their controls and frozen membership must remain intact.

`ObservationCodec` adds the explicit `bounded_long_observations_v1` performance profile over the unchanged native corpus renderer. Its bank retains all sixteen earlier entries exactly and adds held notes of 4, 12 and 40 seconds. Each duration has reference velocities 32/64 at pitch 60 and a transfer view at pitch 72/velocity 80. The resulting bank has fourteen reference and eleven transfer entries. The existing attack-compatible sampler selects adequate references from the contract's chosen bank; transfer views remain forbidden for training.

This profile records a 32-second tail. An initial native check showed that four seconds cut off five of these six sources; amplitude release spans reach about fourteen seconds, with additional effects. The longer recording budget preserves all source settings and the existing audio QA thresholds. Native state readback, clipping/silence checks and end-of-tail checks remain in force. The profile is optional; earlier codecs retain their original bank and two-/four-second budgets.

The maximum declared recording is 72.75 seconds: a 40-second held note, 0.75 seconds of performance padding and a 32-second tail. Float32 stereo at 44.1 kHz occupies about 24.48 MiB, below the existing 32 MiB worker/prefetch cap. Rendering stays offline and on demand. Longer recordings cost more CPU time and are not a reason to pre-render the whole dataset or assume paid compute.

## Feature and artifact boundaries

`ExtendedAudioFeatures` accepts the full recording up to an explicit eighty-second cap. It uses the same stereo spectrum, fixed output shapes, duration scale and frozen Basic Pitch model as the earlier frontend. Basic Pitch still processes one window at a time. No recording is silently cropped, and the model receives audio-derived features only. A late-signal check verifies that audio after sixteen seconds changes both feature branches. Short clips produce exactly the same tensors under the old and extended paths.

The extended frontend has a distinct identity and hashes its implementation dependencies. Training, checkpoint restoration, evaluation, paired feature collection and inference export select the frontend from the codec's observation profile. The earlier sixteen-second frontend identity and limits remain intact. Model exports contain the new contract and registered assets, without private source paths.

The new training contract includes a separately identified performance profile, bank digest and tail budget. Earlier serialized contracts omit the optional field and retain their exact identities. Render recipes verify the chosen bank, performance role and source contract. Existing source pool membership, original bytes, labels and the complete split ledger are preserved when extending observations. Use a new training run; do not resume an earlier codec's optimizer state under this profile.

## Reproduce

```text
python research/training/run.py extend-observations --pool ORIGINAL_PRIVATE_POOL.json --output EXTENDED_PRIVATE_POOL.json
python research/training/reports/scripts/qualify_long_observations.py --pool EXTENDED_PRIVATE_POOL.json --output NEW_QUALIFICATION_DIRECTORY
```

Use the pinned local Python/DawDreamer/Vital environment, with numerical-library thread limits set to two. The qualification verifies adequate references for every training entry, then diagnoses all six previously unobservable states with four random phases fixed and a fresh process for each repeat. It also trains a declared six-source panel under the unchanged full split ledger. Training keeps original requested phases; exact optimizer recovery uses retained features and requires no additional renders. This panel supplements the full 904-state pool and does not replace it.

The [result](reports/LONG_OBSERVATION_RESULTS.md) separates observation/pipeline checks from population-wide audio admission and useful model quality. Authored/corpus mixing, broader source robustness, decoder refitting, data scaling and release gates remain open. Final-test audio stays reserved. GitHub Actions stays disabled and compute remains free-first; Linux/Colab native execution is still unqualified.
