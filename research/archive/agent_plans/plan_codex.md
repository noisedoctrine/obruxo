# NERETRAK Literature Review and Framework Plan

Date: 2026-06-14

## Executive Summary

NERETRAK should be framed as an offline inverse-synthesis system for Vital: generate paired `(preset, MIDI, audio)` examples locally, train an encoder that separates musical content from synthesis identity, and export predictions as valid `.vital` presets. The strongest near-term design is not a pure parameter-regression model. It should combine:

1. A structured `.vital` parameter head for valid preset export.
2. A timbre embedding trained with "same preset, different notes" positives.
3. A supervised parameter loss for recoverable controls.
4. An audio/spectral reconstruction or perceptual matching loss through a local Vital renderer.
5. Optional audio-to-MIDI features as auxiliary conditioning, not as a required source of truth.

The core research bet is disentanglement: identify preset identity across different note patterns, chords, velocities, and durations. Contrastive or JEPA-style objectives are useful because the synthetic dataset can define exact positives and negatives, but they should support the inverse preset model rather than replace direct preset prediction.

## 1. `.vital` File Structure and Model Representation

### Findings

Vital is open-source under GPLv3, and its public repository describes it as a spectral-warping wavetable synthesizer. The repository notes that built-in free presets are separately licensed and should not be redistributed, which matters for dataset policy. Source: https://github.com/mtytel/vital

The user-supplied lead, DBraun/Vita, is especially relevant because it exposes a headless Vital-style synth API. Its bindings expose:

- `load_json(json)` and `to_json()`
- `load_preset(filepath)`
- `render(midi_note, midi_velocity, note_dur, render_dur)`
- `render_file(output_path, midi_note, midi_velocity, note_dur, render_dur)`
- `get_controls()`
- `get_control_details(name)`
- modulation source/destination listing and connection methods

Source: https://github.com/DBraun/Vita/blob/main/src/headless/bindings.cpp

This suggests the right first move is empirical schema extraction:

1. Load Vital's init preset through Vita.
2. Export `to_json()`.
3. Enumerate all controls and control metadata with `get_controls()` and `get_control_details()`.
4. Mutate each control over its legal range, render probe notes, and record the JSON deltas.
5. Build a typed schema from observed keys, value ranges, enums, default values, and modulation routing structures.

### Representation Recommendation

Do not train a model to emit raw `.vital` text. Represent presets as a typed object:

- Continuous normalized controls: floats in `[0, 1]`, with inverse transforms back to Vital-native ranges.
- Discrete/enumerated controls: categorical logits, e.g. filter model, oscillator mode, sync option, distortion type.
- Binary gates: Bernoulli outputs for enable flags and routing toggles.
- Wavetable references: initially constrain to a small built-in/generated wavetable vocabulary; later use a learned wavetable latent or generated wavetable table.
- Modulation matrix: sparse edge prediction over `(source, destination, amount, bipolar/unipolar, curve)`.
- Metadata: preset name, author, comments can be templated and excluded from training.

Export path:

1. Model predicts typed preset object.
2. Validate against schema.
3. Fill missing/unsupported fields from init preset defaults.
4. Convert to Vital JSON.
5. Save as `.vital`.
6. Smoke-test by reloading in Vita and rendering a probe note.

### Important Caveat

Vital has a large parameter surface. Some parameters are perceptually redundant or weakly identifiable from short audio. The first milestone should target a constrained Vital subset:

- 1-2 oscillators
- basic waveforms or a small wavetable bank
- one filter
- amp envelope
- one modulation envelope/LFO
- a small effect set

Then expand once the evaluation harness is stable.

## 2. Best FOSS Audio-to-MIDI Option

### Recommendation: Basic Pitch as the First Baseline

Spotify's Basic Pitch is the best practical first choice. It is open-source, Apache-2.0 licensed, pip-installable, supports Windows/macOS/Linux, outputs MIDI plus note events, includes pitch bends, is instrument-agnostic, and supports polyphonic input, though it works best on one instrument at a time. Source: https://github.com/spotify/basic-pitch

Why it fits NERETRAK:

- Fully local inference.
- Lightweight enough for pipeline experiments.
- Handles synth-like monophonic and polyphonic clips.
- Produces note-event side channels that can help separate content from timbre.
- Compatible with the project's FOSS/local constraint.

### Alternatives

MT3 is a stronger research reference for multi-instrument transcription, but the public repo says training is not easily supported and points users toward Colab/pretrained checkpoints. That makes it less convenient as a local baseline. Source: https://github.com/magenta/mt3

Omnizart is broader, covering pitched instruments, vocals, chords, drums, and beats, with a Python CLI and MIT license. It is worth testing as a second baseline, especially for chord/beat features, but it is a larger toolbox rather than the simplest note-event baseline. Source: https://github.com/Music-and-Culture-Technology-Lab/omnizart

### Design Position

Audio-to-MIDI should be optional auxiliary structure, not a dependency required for correctness. During synthetic training, the true MIDI is already known. For real audio, Basic Pitch can provide an estimated note map, but the model should still accept raw audio/spectrogram input because transcription errors are inevitable.

Use MIDI in three ways:

1. Ground-truth MIDI during synthetic training.
2. Estimated MIDI as a test-time auxiliary feature.
3. Contrastive grouping: same preset rendered under different MIDI clips should map to the same timbre identity embedding.

## 3. Audio to Spectrogram: Prebuilt Layers

### Recommendation: Use Torchaudio in Training

Torchaudio's `MelSpectrogram` is the best default for a PyTorch training stack. It creates mel spectrograms from raw waveform tensors, supports CPU and CUDA, supports autograd and TorchScript, and is implemented as a composition of `Spectrogram` and `MelScale`. Source: https://docs.pytorch.org/audio/stable/generated/torchaudio.transforms.MelSpectrogram.html

Use:

- waveform input at a fixed sample rate, e.g. 44.1 kHz for renderer fidelity or 22.05/32 kHz for speed
- multi-resolution STFT/mel features
- log compression
- channel folding to mono for baseline, with stereo retained later for effects

### Good Secondary Tool: Librosa

Librosa's `feature.melspectrogram` is excellent for offline analysis, debugging, plotting, and feature prototyping. It supports time-series or precomputed spectrogram input and exposes common FFT/hop/mel parameters. Source: https://librosa.org/doc/latest/generated/librosa.feature.melspectrogram.html

### Feature Stack

Recommended baseline features:

- log-mel spectrogram, 128 or 256 mel bins
- linear STFT magnitude for high-frequency/detail-sensitive losses
- chroma or CQT features only if musical content conditioning needs them
- loudness/envelope curves for ADSR estimation
- optional phase-insensitive waveform losses during renderer-based validation

Multi-resolution spectral loss is likely important because Vital parameters affect sound at different scales: oscillator waveform, filter movement, envelope transient, LFO motion, and effects tails.

## 4. Contrastive, Triplet, and JEPA-Style Objectives

### Should We Use Negative Sampling?

Yes, but as an auxiliary objective. The dataset naturally gives high-quality positive and negative pairs:

- Positive pair: same preset rendered with different MIDI.
- Hard positive: same preset with different chord density, velocity, octave, and note duration.
- Easy negative: totally different random preset.
- Hard negative: nearby preset with only one or two changed parameters.
- Very hard negative: same oscillator/filter but different modulation or envelope.

Triplet or supervised contrastive learning can encourage the timbre encoder to ignore note content while preserving synthesis identity.

### Why Not Pure Contrastive?

A pure contrastive model learns an embedding, not a valid preset. NERETRAK ultimately needs a `.vital` file, so contrastive training should shape the latent space while supervised heads predict parameters and routing.

### JEPA-Style Use

JEPA-style learning is attractive because it predicts latent representations rather than reconstructing every spectrogram bin. I-JEPA showed that prediction in embedding space can learn semantic representations without hand-crafted augmentations, but its recipe is image-oriented. Source: https://arxiv.org/abs/2301.08243

For NERETRAK, a music-specific JEPA variant could be:

- Context: one render of a preset under MIDI pattern A.
- Target: another render of the same preset under MIDI pattern B.
- Predictor input: context embedding plus optional MIDI/content embedding.
- Target output: timbre embedding of target audio.
- Stop-gradient target encoder or variance/covariance regularization to avoid collapse.

This is very close to the project's core problem: predict the invariant synth identity despite changing content.

### Practical Loss Mix

Start with:

- Parameter loss: MSE for continuous controls, cross-entropy for categorical controls, BCE for binary controls.
- Modulation loss: sparse edge classification plus amount regression.
- Supervised contrastive or triplet loss over preset IDs.
- Timbre reconstruction/perceptual loss: render predicted preset against the known MIDI and compare spectral features to target audio.
- Optional MIDI consistency loss: predicted content embedding should match known/estimated note events.

## 5. Related Literature

### InverSynth

InverSynth directly addresses synthesizer parameter estimation from spectrogram or raw audio using convolutional networks. It is an important baseline conceptually, though its synth target is simpler than Vital. Source: https://arxiv.org/abs/1812.06349

### Sound2Synth

Sound2Synth estimates FM synthesizer parameters for Dexed and reports real-world-applicable results. It is relevant because it handles a large, real software-synth parameter space, not only toy synthesis. Source: https://arxiv.org/abs/2205.03043

### DDSP

DDSP shows the value of differentiable signal-processing inductive biases, especially separating pitch/loudness from timbre. NERETRAK cannot assume Vital is differentiable end-to-end, but DDSP motivates structured losses and disentangled control. Source: https://arxiv.org/abs/2001.04643

### SynthCloner and Neural Synth Proxies

Recent work is moving toward factorized timbre/content/envelope representations and neural proxy models for black-box synths. These are highly aligned with NERETRAK's direction, especially if direct renderer-in-the-loop training is too slow. Sources:

- SynthCloner: https://arxiv.org/abs/2509.24286
- Neural Proxies for Sound Synthesizers: https://arxiv.org/abs/2509.07635

## 6. Proposed NERETRAK Framework

### Data Generation

1. Define a constrained parameter schema from Vita/Vital.
2. Randomly sample valid presets with musically sane priors.
3. For each preset, render multiple MIDI clips:
   - single notes across octaves
   - repeated notes with different velocities
   - intervals/chords
   - short melodic phrases
   - held notes for envelope/effects tails
4. Store:
   - preset JSON
   - typed parameter vector
   - rendered audio
   - exact MIDI
   - render settings
   - preset family / mutation lineage

### Model

Inputs:

- raw audio and log-mel/STFT features
- optional true or estimated MIDI representation

Encoders:

- audio encoder for timbre and dynamics
- optional content/MIDI encoder
- disentanglement block separating `z_timbre` from `z_content`

Heads:

- parameter prediction head
- modulation graph head
- optional wavetable/head selection
- validity/repair head for impossible combinations

Training:

- supervised parameter prediction
- contrastive same-preset loss
- hard-negative nearby-preset loss
- render-and-compare validation loop
- optional JEPA-style target prediction between different renders of the same preset

Inference:

1. Accept audio clip.
2. Compute spectrogram.
3. Optionally run Basic Pitch for note events.
4. Predict typed preset.
5. Validate/repair.
6. Export `.vital`.
7. Render a probe comparison if MIDI is known or estimated.

## 7. Evaluation

Use multiple metrics because parameter identity and perceptual similarity diverge.

Parameter metrics:

- continuous parameter MAE/RMSE
- categorical accuracy
- modulation edge F1
- valid-export rate

Audio metrics:

- multi-resolution STFT distance
- log-mel distance
- envelope/loudness curve error
- pitch-conditioned similarity using the same MIDI
- ABX listening tests for matched input vs predicted preset

Generalization tests:

- same preset, unseen MIDI
- unseen parameter combinations
- polyphonic vs monophonic clips
- long notes vs short notes
- estimated MIDI vs ground-truth MIDI
- real user audio, no ground-truth preset

## 8. Risks and Open Questions

- Identifiability: many different Vital presets can sound similar. The system should optimize perceptual equivalence, not exact JSON recovery alone.
- Wavetables: arbitrary wavetable recovery is a major subproblem. Start with a closed wavetable vocabulary.
- Effects: reverb/delay/chorus can obscure oscillator/filter identity. Consider predicting dry-ish core first, then effects.
- Licensing: do not train on or redistribute Vital factory presets unless their license allows it. Synthetic random presets generated locally are safest.
- Renderer speed: if Vita rendering is slow, train a neural proxy or cache large render batches.
- Parameter priors: uniform random controls may generate many unusable sounds. Use musically biased priors and mutation families.

## 9. Recommended First Milestone

Build the written/specification version around a narrow proof of concept:

1. Use Vita as schema extractor and local renderer.
2. Limit Vital to a small synth subset.
3. Generate 10k-100k synthetic examples from random presets.
4. Use torchaudio log-mel inputs.
5. Train a baseline CNN/Transformer encoder with parameter heads.
6. Add supervised contrastive loss across same-preset/different-MIDI examples.
7. Compare with and without Basic Pitch auxiliary features.
8. Export `.vital`, reload, render, and measure spectral similarity.

This is ambitious but coherent: it gives NERETRAK a falsifiable research path without requiring cloud services, proprietary datasets, or hand-labeled presets.

