# Full scalar targets and training

`full-scalar-pool` connects the complete 775-field scalar inventory to the audio-only trainer, native renderer, split ledger, checkpoints and inference export. Its model has 424 bounded continuous outputs and 406 categorical heads, with 3,063 categorical logits. No head needs more than 97 choices.

The [completed integration](reports/FULL_SCALAR_RESULTS.md) records exact native codec checks, 400 CPU learning updates, successful export/reload and 50/50 accepted native validation renders. The strict tiny-overfit gate and broad-model quality gates remain open.

| Stored controls | Learned representation |
|---|---|
| Continuous values | Existing atlas mappings in stored units, with measured native endpoints and float32 control-grid serialization |
| Discrete modes and switches | Integral categorical outputs; both source metadata and measured native flags are checked |
| Nine-effect ordering | Eight mixed-radix digits, giving legal permutation ranks `0..362879` |
| Four pitch-quantization masks | Thirteen binary outputs each, giving integer masks `0..8191` |
| Oscillator/LFO cyclic phase | Wrapped phase error |

The measured native metadata describes effect order and pitch masks as continuous controls. The reviewed parameter inventory identifies their discrete meaning. Fractional values for either are rejected by this codec. Order decoding follows Vital's pinned [permutation implementation](https://github.com/mtytel/vital/blob/636ca0ef517a4db087a6a08a6a8a5e704e21f836/src/synthesis/framework/utils.cpp#L26).

Effect-order supervision keeps the relative order of enabled effects and appends disabled effect IDs in a fixed order. This removes contradictory labels for equivalent enabled chains. The codec still preserves the original stored rank exactly. A separate fixed-phase native comparison checks that this supervision transformation preserves sound for the authored effect fixture.

Losses are averaged by control family. Empty modulation slots and disabled modules are masked so their numerous fields cannot dominate learning. Inactive stored values survive at the declared float32 control precision; masking affects supervision. Model output layers start at the reviewed template values and modes, reducing the chance of unrelated effects or high gain appearing at initialization. Existing pilot and alpha model initialization, losses and saved identities retain their earlier behavior.

Native validation caught a near-zero gain rounding mismatch in the first prototype. The corrected decoder performs the affine control conversion in float32, matching the plugin's control grid. The strict native readback threshold remains unchanged. Projection is idempotent and is part of the codec identity.

This increment fixes all six nested structures to the reviewed template: wavetables, sample, LFOs, custom warps, random seeds and empty modulation connections. Encoding any changed nested state fails explicitly. The next coverage work is their learned representation and verified factory/curve vocabulary. All 775 scalar fields have outputs; the initial authored curriculum varies a declared subset across oscillator, filter, envelope, voice and effect families. That distinction is recorded in the integration report.

```powershell
python research/training/run.py full-scalar-pool --output research/training/outputs/full-scalars --count 256
python research/training/run.py codec-check --pool research/training/outputs/full-scalars/pool.json --output research/training/outputs/full-scalar-codec-check
python research/training/run.py train --pool research/training/outputs/full-scalars/pool.json --output research/training/outputs/full-scalar-training --steps 400 --replay-capacity 32 --render-budget 64 --worker-jobs 4
```

The modern pool carries the explicit full-preset renderer identity and a version-two training contract. Raw preset version, asset fingerprints, lineage and reserved-performance checks are enforced when creating recipes. The CLI and default stream factory select the pool's renderer. A checkpoint from another vocabulary or profile cannot be resumed under this contract.

Projection changes state hashes, so projected states inherit a frozen manifest of their original unprojected authored recipes. The complete variant manifest verifies every parent, preserves validation/test membership and rejects missing, duplicate or reparented states. The initial 256 corrected states preserve all original prototype split assignments; the test reserve is not rehashed into training.

Run the fixed integration once in a new output directory:

```powershell
python research/training/reports/scripts/run_full_scalar_screen.py --output research/training/outputs/full-scalar-integration
```

It freezes 256 authored states, five validation families, 400 CPU updates, bounded feature replay, source hashes and native attempt limits. It performs fifteen exact fixed-phase codec renders, two enabled-effect-order equivalence renders, at most 64 training renders and fifty native validation renders. It also reloads the audio-only inference bundle. These are integration measurements; the strict tiny-overfit gate is reported separately, and no model is promoted. No final-test audio or stored audio corpus is used. Broader training, native quality selection and release gates remain open.
