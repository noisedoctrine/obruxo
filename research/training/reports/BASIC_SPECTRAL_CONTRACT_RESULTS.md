# Basic spectral contract: local correctness and resource result

Completed 2026-10-19. The new sound objective and basic control adapter pass local correctness checks. **This verifies the gradient path and native export, not a trained helper or improved reconstruction model.** The historical parameter-supervised trainer is unchanged.

[Machine-readable native evidence](data/basic_spectral_contract_summary.json) · [Active protocol](../STAGED_RECONSTRUCTION.md)

## Implemented contract

The stage still predicts nine normalized continuous values and categorical widths three/two. Training uses straight-through Gumbel-softmax, with a declared linear temperature schedule from 1.0 to 0.5. Category-dependent inactive defaults use tensor operations, retaining category gradients. Export uses deterministic raw-score argmax and the existing `PilotCodec`. No seed, curve, route or additional oscillator head was added.

Spectral targets preserve all frequency bins and time frames at FFT sizes 512/2048/8192, quarter-window hops and 44.1 kHz stereo. Hann windows, constant padding, magnitude scaling, valid-time masks, silence floors and loss weights are versioned. The objective combines spectral convergence/log-magnitude distance, left/right RMS envelopes and side RMS. It preserves absolute gain. These are declared engineering choices, not a claim of calibrated human perception.

The focused tests establish sound-only gradients into active continuous controls and both categorical heads while a fixture helper remains frozen; the fixture is not a trained or qualified sound predictor. All six hard category combinations agree with codec defaults. Tests cover identity/near-silence, gain/stereo/timing sensitivity, padding masks, bad inputs, and exact Gumbel/optimizer recovery with contract-change rejection.

## Native check

All **12/12 requests passed**, covering the three unison counts and two filter states with one repeat each. All **six phase-fixed repeat pairs were exactly equal**. Oscillator 1–3 and sampler random phase were fixed; other controls/seeds were preserved. Each request used a fresh worker. No source pool, split or historical checkpoint was modified; these are explicit synthetic contract fixtures, not a new training population.

Identity loss was zero, and gain, pan and timing perturbations remained visible in all six source cases. No new training WAV archive was written. The run made zero optimizer updates and used no final-test states.

## Local resource measurements

| Observation | Target bytes per example | Maximum examples in 512 MiB target replay |
|---|---:|---:|
| Four seconds | 8,526,525 (8.13 MiB) | 62 |
| Sixteen seconds | 33,993,334 (32.42 MiB) | 15 |

Four-second batches of 1/2/4/8 and a sixteen-second single example completed measurement plus loss backpropagation with finite gradients. The eight-example four-second batch took 0.80 seconds in this single probe. This is not a steady-state training throughput measurement.

The full native/profile run took **26.80 seconds**, with **1,050,157,056 bytes (about 1 GiB)** peak sampled process-tree RSS and no sampling errors. It stayed below the 30-minute/8-GiB limits. Peak memory includes the objective and target/prediction tensors; it excludes a future trained helper and inverse optimizer. Profile those before setting their batch size. The earlier 512-record limit alone would be unsafe for full spectra; the byte cap must govern replay.

Verification: **33 tests passed** across new spectral checks and existing basic-codec/scope tests. Ruff and whitespace checks passed. The initial sandboxed test invocation could not load an installed PyTorch DLL; tests succeeded with local DLL access. An initial test incorrectly applied a flat tensor hash to nested optimizer state; it was corrected to compare optimizer tensors and parameter groups directly, and the rerun passed. Neither issue required changing the training implementation or installing packages.

## Next local work

Freeze the A1 helper architecture/data/update card using these measured feature sizes, then train and qualify it against real Vital. Full spectra can be measured online and sampled at declared absolute frame times for bounded training replay; this sampling must be explicit and qualification must still score full recordings. Useful native gradient directions, unseen-performance behavior and inverse-model quality remain unproven. Colab work is deferred.
