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
- Source-parameter supervision remains useful as an auxiliary signal, especially for identifiable controls, initialization, diagnostics, and controlled experiments. It must not become the governing definition of success.
- A predicted patch should capture synthesis identity rather than memorize the performance in the reference clip. On paired synthetic data, test this by rendering the source and predicted patches under performances not heard by the inference model.
- Returning several ranked candidate patches is desirable if it can be implemented without disproportionate complexity. A simpler system may generate or optimize several candidates internally and expose the best-ranked results; this does not by itself require a generative decoder.

The central modelling problem is therefore not plain parameter regression. It is a structured, multimodal inverse problem in which musical content and synthesis identity are entangled, the forward synthesizer is non-differentiable, and the target program contains continuous controls, categorical choices, routing, permutations, and variable structured data.

## 2. Dataset and Curriculum Design

Training may combine controlled synthetic renders, externally sourced preset corpora, and future community-contributed recordings. Available labels and metadata may be used as noisy supervision for sampling, stratified evaluation, or a future sound classifier, but they are not required model inputs. Source-specific collection and analysis remain within their dataset workstreams.

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
2. **Auxiliary:** supervised losses on known source parameters where useful and identifiable.
3. **Representation learning:** self-supervised objectives that encourage synthesis identity to remain stable across different performances of the same patch.

### Self-supervised representation learning

This is a research question. Likely starting points include conventional contrastive learning using same-patch/different-performance positives and JEPA-style latent prediction. Other methods should be added only when they offer a clear, literature-supported hypothesis rather than another arbitrary combination.

### Non-differentiable perceptual training

Vital does not provide gradients, so the primary rendered-audio objective cannot be backpropagated through it directly. This is a key research question, but it is only one subproblem within audio-to-parameter inference.

- A differentiable neural proxy for `Vital(patch, performance) -> audio features` is the leading candidate.
- Reinforcement learning is possible but currently considered less likely.
- Other black-box, approximate-gradient, search, reranking, or hybrid methods remain open where supported by evidence.

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
