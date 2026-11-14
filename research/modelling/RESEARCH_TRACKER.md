# Audio-to-Vital Modelling Research Tracker

Living record of modelling decisions, hypotheses, experiments, and unresolved questions. This is not a product roadmap. Restrictions on the Vital feature space are experimental controls, and should not be mistaken for statements about the intended final system.

Use the following labels where useful:

- **Settled:** a governing principle or deliberate implementation constraint;
- **Working approach:** the current preferred approach, subject to evidence;
- **Research question:** a consequential choice that requires experiments or literature review.

## 1. Problem Definition

Infer a valid Vital patch from exactly one reference audio clip.

### Settled principles

- **Audio is the only inference-time input.** The user never supplies MIDI or any other conditioning input.
- Pretrained models may derive latent pitch, note, performance, or other conditioning features internally from the audio. These imported models remain frozen; at most, train a small adapter on top. Training a dedicated audio-to-MIDI system is outside the core research problem.
- The primary success criterion is **perceptual similarity** between the reference and audio rendered from the predicted patch. Exact recovery of the hidden source parameters is secondary because multiple Vital patches can be perceptually equivalent.
- Source-parameter supervision may support a brief, declared warm start and diagnostics, but it must not remain an inverse-model training target after warm start. The model is selected by rendered-audio quality because different settings can produce equivalent sounds.
- A predicted patch should capture synthesis identity rather than memorize the performance in the reference clip. On paired synthetic data, test this by rendering the source and predicted patches under performances not heard by the inference model.
- Returning several ranked candidate patches is desirable if it can be implemented without disproportionate complexity. A simpler system may generate or optimize several candidates internally and expose the best-ranked results; this does not by itself require a generative decoder.
- **Every experiment report makes its control scope easy to scan.** Give each compared version its own compact table, not a version/arm column. Group rows by control family/capability; include disabled families instead of hiding them in “everything else.” Put ✅/❌ immediately after the family name and enabled/total instances next (for example, `1 / 3`). Keep control counts separate from dense output counts: one unison choice is one control even when it has three scores. Link the table to a maintained machine-readable inventory with one record per exact Vital field/instance, including fixed and disabled controls, update method, output representation, supervision/audio-loss path, range/choices, fixed value/source, masks, and notes. Distinguish shared-model training from direct patch optimization, warm-start labels from audio-loss training, fixed template values from omitted fields, and frozen from trainable heads.

The central modelling problem is therefore not plain parameter regression. It is a structured, multimodal inverse problem in which musical content and synthesis identity are entangled, the forward synthesizer is non-differentiable, and the target program contains continuous controls, categorical choices, routing, permutations, and variable structured data.

### Current Stage A control scope

This is a family-level scan of the current basic-codec inventory. ✅ means at least one instance in the family has an output head; ❌ means the controls are present but fixed for this experiment. Counts are controls/instances, not dense logits. Fixed controls are explicitly listed by meaningful family below; the linked [control inventory dataset](control_scope/stage_a_controls.jsonl) records the exact fields, fixed defaults, masks, and serialized paths. It contains 979 records: 772 scalar settings and 207 structured fields/assets/curves. Eleven scalar settings are adjustable, with 14 dense outputs (9 continuous values and 5 category scores). Source-setting and proxy-audio losses have both been tested on the 14-output model. The [four-schedule proxy screen](../training/reports/PRESET_LOSS_SCHEDULE_RESULTS.md) used real Vital for final scores, but covered only one validation pitch and did not establish a schedule winner after gain matching. No shared audio-trained model has passed the full reconstruction gate.

| Control family / capability | On? | Enabled / total | Head outputs | Current treatment |
|---|:---:|---:|---:|---|
| Oscillator level | ✅ | 1 / 3 | 1 continuous | Oscillator 1 only; 0.15–0.35 stored units |
| Oscillator pan | ✅ | 1 / 3 | 1 continuous | Oscillator 1 only; −0.7–0.7 |
| Unison detune | ✅ | 1 / 3 | 1 continuous | Oscillator 1 only; 0.5–2.5, masked at one voice |
| Unison voice count | ✅ | 1 / 3 | 3 scores | Oscillator 1 chooses 1, 2, or 4 voices |
| Oscillator on/off | ❌ | 0 / 3 | None | Fixed off/on values from template |
| Oscillator pitch: transpose, tune, and quantize | ❌ | 0 / 9 | None | Fixed by template |
| Oscillator phase and random phase | ❌ | 0 / 6 | None | Fixed phases for repeatable renders |
| Oscillator distortion amount, type, phase, and spread | ❌ | 0 / 12 | None | Fixed by template |
| Wavetable frame selection | ❌ | 0 / 3 | None | Fixed frame values |
| Wavetable assets | ❌ | 0 / 3 | None | Template assets retained |
| Wavetable frame spread | ❌ | 0 / 3 | None | Fixed by template |
| Unison blend | ❌ | 0 / 3 | None | Fixed by template |
| Oscillator detune range and power | ❌ | 0 / 6 | None | Fixed by template |
| Spectral morph type, amount, and spread | ❌ | 0 / 9 | None | Fixed by template |
| Spectral unison and stack style | ❌ | 0 / 6 | None | Fixed by template |
| Oscillator stereo spread | ❌ | 0 / 3 | None | Fixed by template |
| Oscillator destination and MIDI tracking | ❌ | 0 / 6 | None | Fixed by template |
| Oscillator interpolation and 2D view | ❌ | 0 / 6 | None | Fixed by template |
| Custom oscillator curves | ❌ | 0 / 3 | None | No curve heads; template curves retained |
| Filter cutoff | ✅ | 1 / 2 | 1 continuous | Filter 1 only; 50–120 stored units, masked while off |
| Filter resonance | ✅ | 1 / 2 | 1 continuous | Filter 1 only; 0.1–0.6, masked while off |
| Filter on/off | ✅ | 1 / 2 | 2 scores | Filter 1 chooses off or on |
| Filter model and style | ❌ | 0 / 4 | None | Fixed by template |
| Filter drive, mix, blend, and blend transpose | ❌ | 0 / 8 | None | Fixed by template |
| Filter formant controls | ❌ | 0 / 10 | None | Position, spread, transpose, and resonance fixed |
| Filter input and key tracking | ❌ | 0 / 4 | None | Fixed by template |
| Envelope attack/decay/sustain/release | ✅ | 4 / 24 | 4 continuous | Envelope 1 only; other envelope instances fixed |
| Envelope delay and hold | ❌ | 0 / 12 | None | Fixed by template |
| Envelope curve powers | ❌ | 0 / 18 | None | Fixed by template |
| LFO rate and timing controls | ❌ | 0 / 48 | None | Eight LFOs remain at template settings |
| LFO phase, smoothing, stereo, and key tracking | ❌ | 0 / 48 | None | Eight LFOs remain at template settings |
| LFO curves | ❌ | 0 / 8 | None | Curves remain at template settings |
| Modulation route sources | ❌ | 0 / 64 | None | All source fields fixed |
| Modulation route destinations | ❌ | 0 / 64 | None | All destination fields fixed |
| Per-route amount, power, polarity, stereo, and bypass | ❌ | 0 / 320 | None | All five controls for each of 64 routes fixed |
| Per-route input mapping curves | ❌ | 0 / 64 | None | Linear default; no mapping heads |
| Sampler controls | ❌ | 0 / 11 | None | Fixed by template |
| Sample asset | ❌ | 0 / 1 | None | Fixed by template |
| Chorus controls | ❌ | 0 / 12 | None | Fixed by template |
| Compressor controls | ❌ | 0 / 20 | None | Fixed by template |
| Delay controls | ❌ | 0 / 12 | None | Fixed by template |
| Distortion controls | ❌ | 0 / 8 | None | Fixed by template |
| EQ controls | ❌ | 0 / 13 | None | Fixed by template |
| Flanger controls | ❌ | 0 / 9 | None | Fixed by template |
| Phaser controls | ❌ | 0 / 10 | None | Fixed by template |
| Reverb controls | ❌ | 0 / 13 | None | Fixed by template |
| Filter-FX controls | ❌ | 0 / 15 | None | Fixed by template |
| Randomizer controls | ❌ | 0 / 32 | None | Fixed by template |
| Macro controls | ❌ | 0 / 4 | None | Fixed by template |
| Voice controls | ❌ | 0 / 5 | None | Fixed by template |
| Portamento controls | ❌ | 0 / 4 | None | Fixed by template |
| Pitch controls | ❌ | 0 / 2 | None | Fixed by template |
| Stereo controls | ❌ | 0 / 2 | None | Fixed by template |
| Polyphony | ❌ | 0 / 1 | None | Fixed by template |
| Master volume | ❌ | 0 / 1 | None | Fixed by template |
| Tempo, bypass, effect order, legato, mod wheel, MPE, oversampling, velocity, and view settings | ❌ | 0 / 9 | None | Fixed by template |
| **Total scalar settings** | ✅ | **11 / 772** | **14 dense outputs** | **11 selected values enter the codec** |
| **Structured fields/assets/curves** | ❌ | **0 / 207 records** | **None** | **Remain at the reviewed template state** |

## 2. Dataset and Curriculum Design

Training may combine controlled synthetic renders, externally sourced preset corpora, and future community-contributed recordings. Available labels and metadata may be used as noisy supervision for sampling, stratified evaluation, or a future sound classifier, but they are not required model inputs. Source-specific collection and analysis remain within their dataset workstreams.

The [balanced label-only diagnostic](../training/reports/BALANCED_SUPERVISED_CONTROL_RESULTS.md) used the same 14-output scope for 48 updates. Source-setting error fell, but average real-Vital scores worsened for all three sound groups. A separate [four-schedule screen](../training/reports/PRESET_LOSS_SCHEDULE_RESULTS.md) already compared 90-to-0 fading, fixed 90/10, a paper-style fixed weight and a 100-step label warm start followed by audio-only training. The fading version led on raw MR-STFT; the paper-style version led after gain matching. Their direct difference was inconclusive, so no schedule is selected yet.

### Performance variation curriculum

Different recordings of the same preset may range from slight variations of one performance to substantially different musical uses. Where multiple performances are available, estimate their similarity from available audio and performance features so training can progress from closely related variations toward broader performance invariance. This is a curriculum and sampling hypothesis, not a mandatory clustering pipeline.

### Asset taxonomy

Distinguish between:

- untouched canonical factory wavetables;
- factory wavetables modified in Vital's wavetable editor;
- imported or otherwise custom wavetables;
- canonical factory sample-oscillator assets;
- imported sample-oscillator assets.

Displayed names are insufficient because edited content can retain a factory name. Identify canonical content using the manually collected factory list plus normalized content fingerprints. An `Audio File Source` is not automatically imported: some factory wavetables use that representation.

### Dataset partitions

- Train and matched test: untouched canonical factory assets.
- Separate out-of-domain test: editor-modified and imported assets.

This partitioning is a curriculum and evaluation decision; it does not determine the model's parameterization.

### Stretch goals

- Drum and percussion classification for corpus analysis or stratified evaluation.

## 3. Model Architecture

### Encoder

**Working approach:** use separate representations for timbre and performance-related information derived from the same audio input.

- A trainable audio trunk learns features useful for synthesis reconstruction and may use an architecture different from the imported model.
- A frozen pretrained model such as Basic Pitch supplies performance-related latent features. A small trainable projector or MLP adapter is allowed; fine-tuning the imported model is not.
- Some form of self-supervised representation learning will be investigated for the trainable representation. Contrastive learning and JEPA-style prediction are leading families, but the exact method remains a research question.

The self-supervised objective is a major research axis. Avoid an unbounded combination search: select a small number of credible approaches based on the literature and test them against clear baselines.

### Conditioning and disentanglement

**Working approach:** condition the trainable audio representation on frozen performance-related features so the model can separate synthesis identity from what was played.

Cross-attention is the current preferred fusion mechanism because it is straightforward and supported by existing architectures. The exact fusion operator is not presently a major research question. Compare against at most one simple baseline, such as concatenation, if needed to verify that the additional mechanism earns its complexity.

A lightweight global-feature stream is inexpensive and unlikely to be harmful. Treat it as an implementation default rather than a major research axis unless evidence shows otherwise.

### Decoder

The decoder maps a learned audio representation to a structured Vital patch. A latent representation followed by a decoding mechanism is assumed; whether that latent is implemented as one pooled vector or several subsystem/query tokens is a secondary architectural detail, not a distinct training objective.

**Research question:** determine a decoder that can model Vital's heterogeneous and multimodal output space. Candidate families include:

- deterministic grouped prediction heads as a baseline;
- diffusion or flow-matching decoders for multiple valid solutions;
- graph- or set-structured decoding for relational outputs such as modulation routing.

Keep two design axes separate:

1. the **output representation** (grouped fields, sets, graphs, sequences); and
2. the **generative mechanism** (deterministic prediction, diffusion, flow matching, or another conditional generator).

For example, a GNN-based output representation and diffusion are compatible choices, but they answer different questions and should not be evaluated as one indivisible idea.

### Predicted synth parameters

**Working experimental constraint:** predict parameters over canonical factory wavetables; do not generate waveform or harmonic data.

For each oscillator, predict:

- factory wavetable category;
- wavetable frame position;
- wave-morph type and amount;
- spectral-morph type and amount.

This adds three continuous values per oscillator plus categorical choices. Standard oscillator morphing, filters, distortion, mixing, and modulation provide timbral flexibility without introducing a high-dimensional wavetable representation for each oscillator.

## 4. Training Objectives

### Objective hierarchy

1. **Primary:** perceptual or spectral similarity between the reference and a Vital render of the predicted patch.
2. **Warm start only:** supervised losses on known source parameters for a brief, declared initialization period; then train and select from audio feedback.
3. **Representation learning:** self-supervised objectives that encourage synthesis identity to remain stable across different performances of the same patch.

### Self-supervised representation learning

This is a research question. Likely starting points include conventional contrastive learning using same-patch/different-performance positives and JEPA-style latent prediction. Other methods should be added only when they offer a clear, literature-supported hypothesis rather than another arbitrary combination.

### Non-differentiable perceptual training

Vital does not provide gradients, so the primary rendered-audio objective cannot be backpropagated through it directly. This is a key research question, but it is only one subproblem within audio-to-parameter inference.

- A differentiable neural proxy for `Vital(patch, performance) -> audio features` is the leading candidate.
- Reinforcement learning is possible but currently considered less likely.
- Other black-box, approximate-gradient, search, reranking, or hybrid methods remain options for later inference-time refinement, after a base model passes its reconstruction-quality gate.

Status, 2026-11-14: a 21-family inverse-method screen found that direct Vital candidate edits and local score fitting improved the measured score when correct note events were supplied; proxy-gradient patch updates and per-clip TTT were inconclusive. Audio-estimated timing confounded the follow-up, and the families were not an independent holdout. No inference-time method is selected. Keep the four refinements deferred until the base model passes its quality gate; the proxy's role in shared audio-loss training is a separate decision. See the [screen results](../training/reports/INVERSE_METHOD_SCREEN_RESULTS_20261114.md), including the missing run details needed for a reproducible follow-up.

Do not describe the overall project as a neural-surrogate synthesizer. A surrogate is one possible credit-assignment mechanism for training the inverse model.

## 5. Evaluation

Primary evaluation is performed on rendered audio.

- Evaluate from exactly one reference audio clip, with no supplied MIDI or other inference-time conditioning.
- On synthetic paired data, compare source and predicted patches under both the reference performance and held-out performances. The latter tests whether the model recovered synthesis identity rather than overfitting to the observed notes.
- Report source-parameter metrics only as secondary diagnostics, grouped by parameter family and identifiability where possible.
- If multiple candidates are produced, evaluate both top-ranked quality and best-of-candidate quality so ranking and candidate diversity are not conflated.

- Matched evaluation measures performance within the canonical factory-asset vocabulary.
- Out-of-domain evaluation measures how well that constrained vocabulary approximates edited or imported assets.

These results must remain separate: out-of-domain error includes the representational limit of the permitted synth vocabulary.

## 6. Experiments and Ablations

Prioritize experiments that answer the major research questions:

- self-supervised representation learning: a simple supervised baseline versus a small number of contrastive or JEPA-style approaches;
- structured decoding: deterministic grouped heads versus the most credible multimodal decoder;
- perceptual credit assignment: parameter-only training versus a neural proxy or another viable rendered-audio optimization method;
- performance invariance: reference-performance reconstruction versus held-out-performance reconstruction.

Lower-priority implementation choices such as global metadata features, minor fusion variants, and pooled-vector versus query-token latents should not generate large ablation matrices unless they become demonstrated bottlenecks.

## 7. Deferred Output Capabilities

- Harmonic-bin prediction
- Direct waveform generation
- Wavetable-editor component graphs
- Additional wavetable source groups
- Imported audio sources
- Imported sample-oscillator content

## 8. Open Questions and Decision Log

### Settled

- Inference receives exactly one audio clip and no MIDI.
- Internally derived conditioning features are allowed; imported pretrained models remain frozen, with only lightweight adapters trained.
- Perceptual audio similarity is primary; source-parameter recovery is auxiliary.
- The model should generalize synthesis identity across performances.
- Multiple ranked candidate patches are desirable when implementation cost is reasonable.

### Primary research questions

1. Which self-supervised objective best produces a performance-invariant synthesis representation?
2. Which structured decoder best handles Vital's heterogeneous, relational, and multimodal parameter space?
3. How should perceptual reconstruction gradients or equivalent credit reach the inverse model through non-differentiable Vital?

### Working defaults, not primary research questions

- Use frozen Basic Pitch or a similar pretrained model for implicit performance features.
- Use cross-attention for feature fusion, with no more than one simple baseline unless fusion becomes a bottleneck.
- Retain a lightweight global-feature stream.
- Allow the decoder to use either pooled or query-based latent representations as implementation needs dictate.
