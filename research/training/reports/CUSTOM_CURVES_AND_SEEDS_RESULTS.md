# Custom oscillator curves and seed-state integration

Completed locally on 2026-10-17. The model now has three custom oscillator curves and three full 32-bit seed outputs, extending the ordered connections and LFOs. Native state, audible-effect and slot-isolation checks pass. Training, recovery and export work. **The complete-audio gate fails on two clipped source renders, and strict tiny-overfit remains false. No model is promoted.**

## Native behavior and the seed-bit profile

The installed Vital 1.6.4 binary uses the custom curve in spectral modes 13, 14 and 15, and the oscillator seed in mode 6. Fixed-phase comparisons exercise each mode in each of the three oscillator slots. Changing another slot's curve or seed leaves the isolated oscillator's audio exactly unchanged. Inactive-mode changes also leave audio unchanged. These are installed-binary measurements; older source snapshots do not certify the newer fields.

Vital accepts an unsigned seed such as `4294967295` and saves it as signed `-1`. The new, separately identified `full-1.5.5-state-verified-seed-bits-v2` profile compares their exact 32-bit patterns. The old full profile retains its strict JSON comparison. No other scalar/nested checks or audio thresholds are relaxed. The public source format remains unsigned; signed inputs appear only in the direct native equivalence diagnostic.

| Native qualification | Result |
|---|---|
| Custom modes 13–15, all three oscillator slots | Nine fixtures with original/repeated/projected curves and cross-slot controls |
| Seed state boundaries, all three slots | Eight unsigned values per slot, each repeated; includes `16777217`, both sides of the signed boundary and `4294967295` |
| Seed audible effects and cross-slot isolation | Three fixtures, repeated seeds `4` and `123456789`; exact isolation |
| Inactive curves/seeds | Three fixtures; exact unchanged audio |
| Known silent seeds in the fixed isolated fixture | 18 exact-silence audio rejections; state loaded successfully |
| Total | 150 attempts: 132 accepted, 18 expected silence rejections; all eight native gates pass |
| Resources | 155.43 seconds; 414.73 MiB peak process memory, below 8 GiB; no sampling errors |

Separate [signed/unsigned confirmation](data/seed_bit_alias_confirmation.json) renders each encoding twice for four boundary/interior bit patterns: sixteen native attempts. Readback and audio match exactly within every group, including non-silent cases. Its portable script pins the plugin fingerprint, DawDreamer version, source, waveform, phase policy and note timing.

The original/custom-curve projections use 97 linear float32 points. In the eighteen native curve variants, the largest relative waveform RMS error is `0.012366`. This measures the declared approximation in those fixtures, not fidelity for every possible shape. Seeds preserve every bit without float regression; numeric proximity between seeds is not a sound-similarity measure.

The initial mode probe exposed native float32 serialization: decimal `0.8` returns as `0.800000011920929`. Its 68 attempts produced 51 accepted renders and seventeen custom-curve readback errors. Encoding that coordinate on the float32 grid gave 68/68 accepted renders with all seventeen baseline repeats exact. The first full nested check then accepted 75/114 attempts; eighteen silent renders were rejected and 21 signed-seed readbacks failed the old strict profile. Those failed runs remain retained. The final profile explicitly handles the verified seed-bit alias and separately accounts for silent states.

Silence is specific to the measured Basic Shapes waveform at frame 128, mode 6 and amount 0.6. Seeds `0`, `1` and `16777217` produce exact silence there. These rejected clips are not accepted as training audio, and the observation is not generalized to every waveform or morph amount.

## Bounded original-audio training

The 256 authored root assignments remain unchanged: 205 training, 27 validation and 24 reserved test. Thirteen source families cover all three custom modes in each oscillator, the three seed slots and a combined case. All three oscillators use the registered Basic Shapes asset at level `0.06`; master volume is stored as `3844`. This stage does not vary the sampler or the rest of the factory asset vocabulary. Combined training remains necessary.

Original custom curves have three to six points, powers and optional smoothing. Original LFOs and connection maps are also rendered directly. Separate original-curve tensors supervise reconstruction; only audio-derived timbre and frozen Basic Pitch features enter the model. Seeds are encoded as 96 binary labels, preserving the full unsigned range. There is no waveform archive.

| Measurement | Result |
|---|---|
| Model | 5,730,849 trainable parameters; 7,187 continuous outputs and 898 categorical heads |
| Training | 400 CPU updates, batch 16, four threads; 6,400 optimizer draws over 64 distinct source states/audio buffers |
| Training collection | 65 attempts, 64 accepted, one clipping rejection at peak `1.303725`; zero errors |
| Feature tensors | 2,474,240 bytes |
| Original LFO/map/custom-curve labels | 198,656 + 1,589,248 + 74,496 bytes |
| Combined replay | 4,336,640 bytes, within 8 MiB |
| Training/collection time | 404.67 seconds, including 10.00 seconds feature extraction and 234.87 seconds optimizer work |
| Peak process memory | 1,131.56 MiB, below 8 GiB; no sampling errors |
| Saved-target/native correctness | 39/39 renders accepted; all thirteen families; exact repeated/restored audio with generator random phases fixed |
| Export/reload | Model state and codec match; frozen frontend unchanged |

| Complete retained-set metric | Initial | Final |
|---|---:|---:|
| Combined loss | 1.720956 | 0.359504 |
| Continuous MAE | 0.079629 | 0.044439 |
| Weighted categorical accuracy | 77.8354% | 89.8742% |
| Original LFO reconstruction MSE | 0.111734 | 0.045032 |
| Original connection-map reconstruction MSE | 0.008802 | 0.001983 |
| Original custom-curve reconstruction MSE | 0.125142 | 0.036743 |

All thirteen families occur in retained training data, with two to seven examples each. Custom curves were active in 23/15/17 examples by oscillator; seeds in 6/8/7. All 64 original label sets (192 tensors) were independently verified against their source recipes. Predicted active custom curves have mean training RMSE `0.184758`. Seed recovery is **497/672 bits correct (73.96%), but only 1/21 whole seeds exact**. Strict tiny-overfit remains false. Lower curve error and overall categorical accuracy do not establish useful routing or seed recovery. The [training summary](data/nested_training_summary.json) preserves these separate results. The 400-update run is an integration check, not a data-scaling or seed-tuning study.

## Native sound panel and open gates

The frozen panel plans thirteen validation states under reference and reserved-transfer performances: 156 native renders. It attempted 140 and accepted 138. One seed-2 reference source clipped at peak `1.596333`, making that whole case unavailable for inference/evaluation; its transfer performance was not attempted. A seed-3 transfer source clipped at peak `1.039505`. Sixteen dependent renders were skipped. All 23 scorable model predictions rendered successfully; there were no state-readback errors. Literal source repeats were exact on all 23 scorable performances.

| Mean distance from original sound; lower is better | Reference, scored/planned | Transfer, scored/planned |
|---|---:|---:|
| Label reconstruction: representation error | 0.002041 (12/13) | 0.002468 (11/13) |
| Audio-only model | 1.353393 (12/13) | 1.312343 (11/13) |
| Training-only retrieval | 1.067708 (12/13) | 1.123221 (11/13) |
| Fixed template | 1.425369 (12/13) | 1.558317 (11/13) |
| Literal source repeat | 0 (12/13) | 0 (11/13) |

Prediction error remains much larger than representation error, and retrieval is better on both performances. The authored sources differ from prior stages, so these values do not establish an improvement over an earlier model. Headroom, strict overfit, combined coverage and useful model quality remain open. The failed source performances remain charged and visible; the validation panel is unchanged and final-test audio is untouched. The fundamental custom-curve and seed capabilities remain in the model.

The [native plan](data/nested_native_plan.json), [native summary](data/nested_native_summary.json), [mode probe](data/nested_mode_probe_summary.json), [training plan](data/nested_training_plan.json), [training summary](data/nested_training_summary.json) and [saved-target checks](data/nested_training_codec_summary.json) preserve runtime hashes, normalized source hashes, source identities, failures, coverage and artifact hashes. The final qualification and training sources were unchanged during their runs. Metadata is committed; third-party assets, model weights and waveforms remain private local outputs.

All 208 affected local regression checks passed in 581.68 seconds. They cover legacy profiles, the new seed profile, unsigned bit boundaries, wrong-bit rejection, original-label integrity, audio-only inputs, gradients, exact optimizer/RNG recovery and evaluation. Ruff and whitespace checks pass. Local checks overlapped training/native work, so timings are not isolated throughput benchmarks.

Free local CPU remains the qualified path and GitHub Actions stays disabled. The next work is an explicit combined capability matrix and broader control/performance curriculum, followed by measured scaling and the remaining quality/release gates. One-point curve records and other eligible raw-layout gaps remain explicit. Reproduction is in [CUSTOM_CURVES_AND_SEEDS.md](../CUSTOM_CURVES_AND_SEEDS.md); the full goal remains open in the [completion ledger](../IMPLEMENTATION_STATUS.md).
