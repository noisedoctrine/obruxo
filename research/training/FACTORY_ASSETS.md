# Learned factory wavetable and sample choices

The factory path adds three numbered wavetable choices and one sampler choice to the [775-field scalar model](FULL_SCALAR_TRAINING.md). The installed factory bank supplies 19 wavetables and 10 samples, giving 20 wavetable and 11 sample choices when the init assets are included. Audio remains the only inference input. The frontend stays frozen, and the pipeline renders training audio on demand into bounded feature replay.

The [qualification and integration report](reports/FACTORY_ASSET_RESULTS.md) distinguishes source-import fidelity, exact codec correctness, learning checks and held-out sound measurements. This is a capability increment; it does not promote a finished model.

## Import the installed references

Run from the repository root in the pinned Windows native-rendering environment:

```powershell
python research/training/run.py factory-bank --factory-root 'C:\Users\angert\Documents\Vital\Factory' --output research/training/outputs/factory-bank
python research/training/run.py factory-pool --bank research/training/outputs/factory-bank/bank.json --output research/training/outputs/factory-pool --count 256
python research/training/run.py codec-check --pool research/training/outputs/factory-pool/pool.json --output research/training/outputs/factory-codec
python research/training/run.py train --pool research/training/outputs/factory-pool/pool.json --output research/training/outputs/factory-training --steps 1000 --replay-capacity 64 --render-budget 128 --sampling-policy state_round_robin --cpu-threads 4 --worker-jobs 4
```

Outputs must be new directories. Replace the factory root with the installed location. The default [source manifest](configs/factory-assets-v1.json) pins all 44 relative paths and file hashes. `--manifest` allows an explicitly reviewed replacement manifest; it does not bypass native qualification. A missing or changed reference fails before import. A failed native/audio check leaves an unqualified bank that the codec refuses. Import uses one isolated process per reference, a finite deadline, a fixed native-attempt budget and an 8 GiB measured memory ceiling.

The source manifest includes 15 LFO shapes for the upcoming curve representation. This increment qualifies their native loading and repeated audio, but does not predict their shapes. LFOs, custom warps, seeds and modulation connections remain fixed to the init template. Changed unsupported nested content is rejected explicitly.

## Preserve the source sound

Factory wavetables include older formats. The importer loads each original with the pinned Vital binary, captures native re-exported structure, explicitly declares the reviewed 1.5.5 header, and checks that re-export stabilizes. It then compares original and converted audio at frames 0, 128 and 256 with all oscillator and sampler random phases fixed. The allowed relative RMSE is 0.001 and absolute peak difference is 0.0001. Silence at every probe, clipping, invalid state or an ambiguous default replacement fails qualification. This import comparison allows small numeric differences; the subsequent codec check requires exact fixed-phase audio.

Mono/stereo PCM16 sample imports retain channel samples, frame count and sample rate without resampling or normalization. Native probes use three pitches and an exact repeated middle-pitch render. The full renderer still applies its documented four-sample prefix and at-most-one-PCM16-step readback policy; native sample-state readback is not byte-exact. LFO native probes require exact repeated audio. These checks cover the declared performances, not every possible setting or note.

Only reviewed names/authors are canonicalized when registering equivalent content; DSP-bearing data remains fingerprinted. Unknown assets fail encoding. Disabled oscillator/sample choices remain stored while their supervision is masked. The codec masks wave-frame supervision only for a confidently static single-keyframe Wave Source.

## Saved identities and split safety

The bank records the import policy, source and canonical asset hashes, native renderer, complete qualification result and source-code hashes. The codec embeds the qualified payload once in each private bundle, so inference needs no path into the local factory installation. Execution timing is excluded from codec identity; content, policy and qualification-plan identity remain pinned. Existing pilot, alpha and full-scalar artifact identities stay unchanged.

Asset mutations inherit the original full-scalar authored roots. All 256 roots keep their original train/validation/test membership, and pool reload reconstructs this mapping. Bank qualification uses the manually declared factory references, not held-out presets. Model training uses train roots only; the small integration panel uses validation roots. Final-test audio remains reserved.

## Repeat the bounded integration

```powershell
python research/training/reports/scripts/run_factory_asset_screen.py --bank research/training/outputs/factory-bank/bank.json --output research/training/outputs/factory-integration
```

The runner freezes 256 states, six validation families, source hashes, 1,000 CPU updates, 64 retained feature examples and fixed native-attempt limits. It performs 18 exact codec renders, at most 128 training renders and 60 native validation renders, then checks bundle restoration, frozen frontend and memory. It reports the strict tiny-overfit gate separately. It does not select or promote a release model.

Generated banks, pools, checkpoints and asset payloads stay in ignored local outputs. Committed evidence contains hashes and measurements. Qualification does not establish redistribution permission; asset notices and the distributable release vocabulary remain release work. Linux/Colab rendering remains unqualified, so this run uses free local compute. GitHub Actions stays disabled.
