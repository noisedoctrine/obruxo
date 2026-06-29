# NERETRAK: Claude Literature Review & Framework Plan

> Independent research and framework design for a fully local, open-source pipeline that infers Vital synthesizer presets from audio clips.

---

## 0. Framing: The Core Tension

NERETRAK is not a standard audio classification or regression problem. It sits at the intersection of three hard subproblems:

1. **Timbre disentanglement** — separating what a synthesizer *is* (its configuration) from what it is *playing* (the note pattern).
2. **Structured output synthesis** — producing a semantically valid, interoperable `.vital` JSON file, not a free-form latent vector.
3. **Non-differentiable grounding** — Vital is a black-box C++ engine. You cannot backprop through it at training time.

The design space is therefore not "pick the best spectrogram encoder" but rather "find the smallest set of inductive biases that collapse all three problems to tractable subproblems at once."

---

## 1. Topic 1: `.vital` File Structure and ML Representation

### 1.1 What a `.vital` file actually is

`.vital` files are plain-text JSON. The format is serialized by Vital's `LoadSave` class. A file has the following top-level shape:

```json
{
  "synth_version": "1.5.5",
  "preset_name": "...",
  "author": "...",
  "comments": "...",
  "macro1": "...", "macro2": "...", "macro3": "...", "macro4": "...",
  "settings": { ... }
}
```

The `settings` object contains:
- **~775 scalar key-value pairs**: continuous floats in *native engine space* (e.g., filter cutoff in semitones, not 0–1). These are defined in `synth_parameters.cpp` via a `ValueDetails` struct holding `min`, `max`, `default_value`, and a `scale_type` (kLinear, kExponential, kQuadratic, kIndexed, etc.).
- **`modulations`**: array of `{source, destination, amount, bipolar, line_mapping}` objects. Crucially, **modulation routing cannot be set via VST `setParameter`**—only the Vita JSON API or raw JSON manipulation exposes this.
- **`wavetables`**: array of per-oscillator objects containing base64-encoded float32 keyframe arrays. Each keyframe is ~2048 samples. This dominates file size and is essentially a high-dimensional generative subproblem on its own.
- **`lfos`**: array of LineGenerator point arrays (not just the `lfo_N_*` scalar params).
- **`sample`**: optional embedded PCM in base64.

### 1.2 Critical insight: two distinct parameter surfaces

There are actually **two ways** to interact with Vital parameters:

| Surface | Access | Covers |
|---------|--------|--------|
| VST `getParameter`/`setParameter` (DawDreamer) | ≈771 scalar floats, VST-normalized [0,1] | Scalar controls only |
| Vita JSON API (`load_json`, `to_json`, `get_controls`) | Full preset including modulation routing, wavetables, LFO curves | Everything |

This means **DawDreamer is insufficient** for full preset recovery—modulation routing (arguably the most timbrally expressive part of Vital) is only accessible through the JSON path via Vita.

### 1.3 Recommended ML representation

Do **not** model the raw JSON blob. Instead, decompose into typed heads:

| Component | Representation | Notes |
|-----------|----------------|-------|
| Continuous scalars (~600) | Fixed-length vector, normalized to [0,1] per `ValueDetails` scale | Inverse-transform on export |
| Categorical/indexed (~100) | Integer class labels → softmax logits | e.g., filter model, osc waveform, distortion type |
| Boolean gates (~70) | Binary Bernoulli outputs | Enable flags, routing toggles |
| Modulation routing (up to 64 slots) | Sparse bipartite graph: per-slot (source_id, dest_id, amount, bipolar) | Source/dest from finite vocabulary; treat unfilled slots as null class |
| LFO curves | 32-point resampled float vector per LFO | Captures shape without full resolution |
| Wavetables | Phase A: index into factory table vocabulary (~80 tables) | Phase B: 16-dim latent from a separate wavetable autoencoder |
| Sample | Exclude from Phase A (`sample_on = 0` in training set) | Separate modality; out of scope initially |

**Export pipeline:**
```
model output → typed preset object → denormalize scalars via ValueDetails
→ build settings dict → vita.Synth().load_json(dict) → to_json() → write .vital
→ smoke-test: reload, render probe note, measure mel distance
```

Round-tripping through Vita applies version migration and validates control names for free.

### 1.4 Key normalization pitfalls

- Envelope times use **quartic** scale in Vital; ADSR parameters must be denormalized carefully.
- `effect_chain_order` is a float that encodes a permutation—treat it as a categorical, not a continuous value.
- VST-normalized values ≠ native engine values. Vita exposes both `set()` (native) and `set_normalized()` (VST). Pick one convention and be consistent throughout the entire pipeline.
- Boolean params must be **rounded** before export (predict as sigmoid, threshold at 0.5).

### 1.5 The identifiability problem

Many distinct Vital configurations are **perceptually equivalent**:
- An oscillator with level 0 makes all of its other parameters irrelevant.
- Phase offsets on some wavetable shapes produce identical audio.
- Modulation slot ordering is a permutation symmetry (slot 1 routing A→B is identical to slot 2 routing A→B with same amount).

This is the strongest argument for using a **generative/distributional decoder** rather than pure point-regression in later phases (see Section 5).

---

## 2. Topic 2: Best FOSS Audio-to-MIDI Option

### 2.1 Role clarification

First, be precise about *when* MIDI is needed:

| Phase | MIDI source | Use |
|-------|-------------|-----|
| Dataset generation | **Ground truth** — we control what MIDI we render | No transcription needed |
| Training (contrastive) | Ground truth MIDI used as conditioning signal | Contrastive grouping: same preset ↔ different MIDI renders |
| Inference (real audio) | **Estimated** via transcription model | Note-content conditioning to help disentangle |

The transcription model is only critical at inference time. During training, we always have the true MIDI.

### 2.2 Candidate analysis

| Tool | License | Polyphony | Pitch bends | Notes |
|------|---------|-----------|-------------|-------|
| **Basic Pitch** (Spotify) | Apache 2.0 | Yes | Yes | Lightweight, pip-installable, instrument-agnostic, best default |
| Onsets and Frames (Magenta) | Apache 2.0 | Yes | No | Piano-only, not suitable for synth timbres |
| piano_transcription_inference (ByteDance) | MIT | Yes | No | Piano-only, very high quality but limited domain |
| MT3 (Google) | Apache 2.0 | Yes | No | Multi-instrument but heavy, hard to run locally |
| Omnizart | GPLv3 | Yes | No | Broad but slow and complex to deploy |

### 2.3 Recommendation: Basic Pitch

**Basic Pitch** is the unambiguous first choice: Apache 2.0, pip-installable, instrument-agnostic, polyphonic, and includes pitch bend detection—relevant because synthesizers frequently have portamento, vibrato, and pitch modulation.

```python
from basic_pitch.inference import run_inference
from basic_pitch import ICASSP_2022_MODEL_PATH

model_output, midi_data, note_events = run_inference(
    audio_path,
    model_path=ICASSP_2022_MODEL_PATH
)
# note_events → piano-roll matrix (pitch × time) for model conditioning
```

### 2.4 Integration strategy: treat MIDI as optional conditioning

The model should be designed to **degrade gracefully** when MIDI is unavailable or noisy:
- During training: always provide ground-truth MIDI.
- During inference: provide Basic Pitch output, but with **classifier-free guidance**-style dropout during training (with some probability, mask out the MIDI conditioning) so the model can still predict from audio alone.
- A good ablation: compare with-MIDI vs. without-MIDI at inference time on a held-out set of real audio.

---

## 3. Topic 3: Audio → Spectrogram: Prebuilt Layers

### 3.1 Why on-the-fly GPU layers matter for this project

Vital renders at 44.1 kHz. Processing offline with librosa and storing pre-computed spectrograms for a 100k-clip dataset is wasteful and inflexible. GPU-native spectrogram layers allow:
- Data augmentation in the feature domain (SpecAugment, pitch shift).
- Ablating spectrogram parameters without re-running preprocessing.
- Future trainable-basis experiments.

### 3.2 Recommended stack

**Primary: `nnAudio`** — implements STFT/mel as 1D convolution layers in PyTorch, runs on GPU, and is 50–100× faster than CPU-based librosa for batch processing.

```python
from nnAudio.features.mel import MelSpectrogram

spec_layer = MelSpectrogram(
    sr=44100,          # Match Vita's render sample rate
    n_fft=2048,
    n_mels=128,
    hop_length=512,
    trainable_mel=False,   # Start fixed; ablate True later
    trainable_STFT=False,
)
```

**Feature stack (baseline):**

1. **Log-mel spectrogram (128 bins)** — captures coarse spectral envelope and timbral character.
2. **Linear STFT magnitude (up to Nyquist)** — essential for high-frequency detail: noise oscillators, filter resonance peaks, bitcrushing artifacts. Mel-scale compresses and obscures these.
3. **Envelope/loudness curve** — extracted as per-frame RMS. Directly useful for regressing ADSR parameters.

**Multi-resolution spectral loss:** Vital parameters affect the signal at wildly different time-scales—oscillator waveform shape (instantaneous), filter cutoff (can be static or moving), envelope transient (tens of ms), LFO motion (hundreds of ms to seconds). A single STFT window cannot capture all of these. Train with **multi-resolution STFT loss** across at least 3 window sizes (e.g., 256, 1024, 4096 samples).

### 3.3 Encoder backbone options

| Backbone | Pre-trained | Notes |
|----------|-------------|-------|
| **EfficientAT** | Yes (AudioSet) | Best performance in Neural Proxies paper; efficient |
| **AST** (Audio Spectrogram Transformer) | Yes (AudioSet) | DAFx24 synth-matching baseline; good spectrogram-to-label performance |
| **PaSST** | Yes (AudioSet) | Patch-level, efficient attention |
| **CLAP** | Yes (AudioSet+music) | Text-audio alignment; may over-specialize on semantic content |
| CNN from scratch | No | Fastest to iterate; weakest generalization |

**Recommended default**: Start with a **frozen EfficientAT** backbone as a feature extractor (per the Neural Proxies paper's findings), fine-tune after the preset regression heads are stable.

---

## 4. Topic 4: Contrastive, Triplet, and JEPA-Style Objectives

### 4.1 The disentanglement formulation

Define two latent spaces:

- **z_timbre** — the synthesis identity embedding. Should be invariant to: what notes are played, how many notes, their velocity, duration, octave.
- **z_content** — the musical content embedding. Should capture note pattern, polyphony, timing.

The objective is: the model should recover `z_timbre` → preset parameters, while `z_content` is used only for conditioning/decoupling.

We have **complete control over the data generating process** — this is the key advantage. We can construct oracle-quality positive and negative pairs:

```
Positive pair:  render(preset_P, midi_A) ↔ render(preset_P, midi_B)   # same synth identity
Negative pair:  render(preset_P, midi_A) ↔ render(preset_Q, midi_A)   # different synth identity
Hard negative:  render(preset_P, midi_A) ↔ render(preset_P_jitter, midi_A)  # nearby preset
```

### 4.2 Contrastive / triplet loss

**InfoNCE** (batch-level contrastive, also called NT-Xent/SimCLR) scales better than triplet loss and should be preferred:

$$\mathcal{L}_{\text{InfoNCE}} = -\log \frac{\exp(\text{sim}(z_A, z_P)/\tau)}{\exp(\text{sim}(z_A, z_P)/\tau) + \sum_{i \neq P} \exp(\text{sim}(z_A, z_i)/\tau)}$$

Use **preset ID** as the grouping label—all renders of the same preset are positives for each other. This is equivalent to supervised contrastive learning (Khosla et al., 2020) with batch construction by preset ID.

**Hard negative mining strategy:**
1. *Easy negative*: random different preset.
2. *Hard negative*: preset with parameter jitter (from the "Synthetic Doppelgängers" paper, arXiv:2406.05923). A preset `P_jitter` is created by perturbing 1–3 scalar parameters of `P` by a small delta. The model must learn to distinguish these.
3. *Category-level negative*: preset with the same oscillator waveform but different filter/envelope — tests whether the model overfits to coarse timbral category rather than fine preset identity.

### 4.3 JEPA-style objectives

JEPA (Joint-Embedding Predictive Architecture) predicts latent representations rather than reconstructing raw signals, which avoids the "collapse to mean" problem of pixel-level reconstruction.

For NERETRAK, a music-domain JEPA variant is naturally motivated:

```
Context:   z_timbre(render(preset_P, midi_A))
Predictor: takes z_timbre(A) + encode(midi_B)
Target:    z_timbre(render(preset_P, midi_B))  [stop-gradient / EMA target encoder]
```

The predictor must learn to *transport* the timbre representation across different musical contexts—exactly the invariance we want. This is closely related to Stem-JEPA (arXiv:2408.02514), which learns compatible stem representations from a mix.

**JEPA vs. InfoNCE for NERETRAK:**

| | InfoNCE | JEPA |
|--|---------|------|
| **Supervision signal** | Contrastive batch | Predictive in latent space |
| **Collapse avoidance** | Negative pairs | Stop-gradient + EMA target |
| **Best for** | Disentanglement with clean positive/negative labels | Representation pretraining without labels |
| **In NERETRAK** | Core disentanglement objective (we have labels) | Optional pretraining stage on larger unlabeled audio |

**Recommendation**: Use **supervised InfoNCE** as the primary disentanglement objective (since we have exact preset-ID labels), and consider JEPA pretraining on unlabeled audio (e.g., AudioSet) only if the synthetic-data-only encoder underfits.

### 4.4 Generative decoder for ill-posed inversion (Phase B+)

Point regression fails under permutation symmetries inherent in Vital (effect chain order, modulation slot assignment, parallel routing). The equivariant flow matching paper (ISMIR 2025, arXiv:2506.07199) showed that a **conditional normalizing flow** in the parameter space outperforms regression baselines precisely because it can represent multimodal output distributions.

**Recommended approach for Phase B:**
- Replace the MLP regression heads with a **flow matching decoder** conditioned on `z_timbre`.
- The decoder learns a probability density over the normalized parameter vector, allowing sampling of multiple valid presets from a single audio input.
- This also enables interactive usage: the user can re-sample the decoder to get alternative-but-perceptually-similar preset options.

---

## 5. Key Distinguishing Insights (What Other Plans May Miss)

### 5.1 The "many-to-one" problem is worse for Vital than for simple synths

Vital has ~775 parameters vs. ~16 in Massive (as studied in DAFx24). The combinatorial redundancy is vastly higher. This means:
- Parameter-level MSE loss will be uninformative on redundant parameters.
- The perceptual loss (rendering and comparing audio) is the only honest evaluation signal.
- You should define a **perceptually effective parameter subset** for Phase A: which parameters are actually audible and identifiable from short audio clips?

### 5.2 Disentanglement via the "preset family" concept

When generating the synthetic dataset, consider **preset mutation families**: start from a random base preset, then generate N variations by perturbing subsets of parameters. This creates a natural hierarchy:
- Same-family renders: share the broad timbral class.
- Same-preset renders: identical except for MIDI.

Training with family-level labels in addition to preset-level labels lets the model learn at multiple granularities of identity—coarse timbral category and fine preset identity.

### 5.3 Renderer speed is a practical bottleneck; plan for it

If Vita renders at ~10–100× real-time, a 100k-clip dataset of 3-second clips could take hours. Options:
1. **Pre-render the dataset** and store audio + preset pairs. Simple but large storage requirement.
2. **Train a neural proxy** of Vital (a neural network that maps parameter vector → spectrogram features) and run gradients through the proxy. The Neural Proxies paper (arXiv:2509.07635) showed EfficientAT-based proxies are effective for this.
3. **Hybrid**: pre-render, then fine-tune with RL (SynthRL approach) on hard cases.

The neural proxy is the most elegant because it makes the training loop entirely differentiable, but it introduces a second model to train and maintain.

### 5.4 Modulation graph prediction is a sequence problem

The modulation routing head is not a simple regression. It is a **sparse graph prediction problem** over a fixed vocabulary of sources (~20: env_1–4, lfo_1–8, macro_1–4, midi_cc, etc.) and destinations (~775 parameters). Concretely:

- Treat each of Vital's 64 modulation slots as a sequence position.
- Per slot: predict (source_id, dest_id, amount, bipolar) via a 4-headed output.
- Sort predicted slots canonically (alphabetically by source+destination) before loss computation to eliminate slot-ordering permutation.
- Use a **set transformer** or **pointer network** for the routing head rather than a simple MLP—these architectures are designed for unordered set outputs.

### 5.5 Dataset quality priors matter

Uniform random sampling of Vital parameters will produce many **uninteresting or unusable sounds**: completely silent patches (all levels at zero), extremely distorted/clipping sounds, inaudible high-frequency sounds. The training set should use **musically biased priors**:
- Oscillator levels: sample from Beta(2, 1) to bias toward non-zero.
- Filter cutoff: sample uniformly in MIDI note space (60–100), not raw Hz.
- Envelope times: sample from log-uniform distributions (short attacks are more common in music).
- Keep a **diversity metric** on the dataset (e.g., feature-space spread in a pre-trained audio embedding) to ensure the dataset isn't degenerate.

---

## 6. Proposed Architecture

```
Input audio (44.1kHz waveform)
    │
    ▼
[nnAudio mel layer] → [log-mel 128×T] ─────────┐
[nnAudio STFT layer] → [linear STFT ×T]         │
[RMS envelope] → [loudness curve]                │
    │                                            │
    ▼                                            ▼
[Frozen EfficientAT backbone]        [Optional: Basic Pitch → piano-roll]
    │                                            │
    └─────────────────────┬──────────────────────┘
                          │
                          ▼
               [Disentanglement block]
                /                   \
         z_timbre                 z_content
              │                        │ (auxiliary only)
              │                        │
              ▼                        ▼
    [Hierarchical preset decoder]  [MIDI consistency head]
    ├── Continuous scalar head (MSE)
    ├── Categorical head (cross-entropy)
    ├── Binary gate head (BCE)
    ├── Modulation graph head (set transformer)
    └── LFO curve head (MSE on 32-point resampled curve)
              │
              ▼
    [Vita export + smoke test]
              │
              ▼
         .vital file
```

**Training objectives:**
```
L_total = λ_param · L_param 
        + λ_con  · L_InfoNCE(z_timbre, preset_id)
        + λ_prox · L_proxy  (optional: neural proxy perceptual loss)
        + λ_midi · L_content (optional: MIDI consistency loss on z_content)
```

---

## 7. Phased Rollout Plan

| Phase | Scope | Key deliverable |
|-------|-------|-----------------|
| **0: Schema extraction** | Run Vita on init preset; enumerate all controls, ranges, scale types | Typed parameter index map |
| **1: Minimal PoC** | ~2 oscillators, 1 filter, amp envelope, no effects, no modulation routing | End-to-end pipeline works; can export valid `.vital` |
| **2: Contrastive disentanglement** | Add InfoNCE loss; train on same-preset/different-MIDI pairs | `z_timbre` clusters by preset ID; improved polyphony robustness |
| **3: Full scalar parameter set** | All ~775 scalar controls; add effects, LFO scalars | Full preset recovery for scalars |
| **4: Modulation routing** | Set transformer modulation graph head | Can recover basic modulation assignments |
| **5: Generative decoder** | Flow matching decoder replaces MLP heads | Handles permutation symmetries; multi-modal output |
| **6: Wavetable extension** | Factory wavetable index prediction; Phase B: wavetable latent | Expands to wavetable-heavy presets |

---

## 8. Evaluation Protocol

### Parameter-level (synthetic, ground-truth available)
- Continuous MAE/RMSE per parameter group (oscillator, filter, envelope, effects)
- Categorical accuracy per discrete parameter
- Modulation routing F1 (edge recall and precision on bipartite graph)
- Valid export rate (`.vital` successfully loads in Vita)

### Audio-level (render predicted preset and compare)
- Multi-resolution STFT distance (MSS loss)
- Log-mel spectrogram L2 distance
- CLAP cosine similarity (semantic audio similarity)
- Envelope error (loudness curve DTW distance)

### Generalization tests
- Same preset, unseen MIDI
- Held-out preset families not seen during training
- Real user audio (no ground-truth preset)
- Polyphonic vs. monophonic input clips

### Listening tests
- MUSHRA or ABX test: original audio vs. predicted preset rendered with same MIDI

---

## 9. Open Questions and Risks

| Issue | Details | Mitigation |
|-------|---------|------------|
| **Identifiability** | Many Vital configs are perceptually equivalent; parameter-level loss is misleading | Use perceptual audio loss as primary signal; design phased evaluation |
| **Wavetable recovery** | Arbitrary wavetables are a high-dimensional generative subproblem | Restrict Phase A to factory wavetable index prediction; defer custom wavetable to Phase 6 |
| **Effects obscure core identity** | Reverb/delay/chorus can make oscillator parameters impossible to identify from dry+wet signal | Train on dry renders first (`reverb_dry_wet = 1.0`); add effects in later phases |
| **Dataset distribution gap** | Random presets may not cover real-world sound design patterns | Use preset mutation families; consider adding a small curated preset bank (verify licensing) |
| **Renderer speed** | Vita rendering at scale could be a bottleneck | Pre-render to disk; or use a neural proxy model |
| **Vita version drift** | Vital preset format is not officially versioned; Vita may lag behind Vital | Pin Vital version in dataset metadata; version-lock Vita dependency |
| **Modulation slot permutation** | Multiple valid slot assignments for the same routing | Canonical sorting of predicted routing before loss; or use set-theoretic loss |
| **Parameter priors** | Uniform sampling produces many silent/unusable patches | Musically biased sampling distributions; keep/reject based on loudness floor |

---

## 10. Key References

| Paper / Tool | Relevance |
|---|---|
| [Vital source code](https://github.com/mtytel/vital) | Preset format, `synth_parameters.cpp`, parameter metadata |
| [Vita Python bindings](https://github.com/DBraun/Vita) | Local rendering, preset I/O, schema extraction |
| [InverSynth (arXiv:1812.06349)](https://arxiv.org/abs/1812.06349) | Foundational CNN-based inverse synthesis baseline |
| [Sound2Synth (arXiv:2205.03043)](https://arxiv.org/abs/2205.03043) | FM synth (Dexed) parameter estimation at real scale |
| [DDSP (arXiv:2001.04643)](https://arxiv.org/abs/2001.04643) | Differentiable DSP; motivates structured losses and disentangled control |
| [Neural Proxies for Synths (arXiv:2509.07635)](https://arxiv.org/abs/2509.07635) | Best approach for black-box gradient flow; EfficientAT backbone |
| [SynthRL (IJCAI 2025)](https://www.ijcai.org/proceedings/2025/1129.pdf) | RL-based cross-domain sound matching; inference-time fine-tuning |
| [Equivariant Flow Matching (arXiv:2506.07199)](https://arxiv.org/abs/2506.07199) | Generative preset decoder; handles permutation symmetry |
| [Synthetic Doppelgängers (arXiv:2406.05923)](https://arxiv.org/abs/2406.05923) | Hard negative mining via parameter jitter |
| [Audio-JEPA (arXiv:2507.02915)](https://arxiv.org/abs/2507.02915) | Self-supervised audio encoder pretraining |
| [Stem-JEPA (arXiv:2408.02514)](https://arxiv.org/abs/2408.02514) | Timbre/harmony/rhythm representation; directly analogous to our JEPA target |
| [DAFx24 AST for Massive](https://www.dafx.de/paper-archive/2024/papers/DAFx24_paper_95.pdf) | Spectrogram Transformer for synth parameter estimation |
| [Basic Pitch](https://github.com/spotify/basic-pitch) | FOSS audio-to-MIDI; auxiliary conditioning at inference |
| [nnAudio](https://github.com/KinWaiCheuk/nnAudio) | GPU-native spectrogram layer; 50–100× faster than librosa |
| [Supervised Contrastive Learning (NeurIPS 2020)](https://arxiv.org/abs/2004.11362) | InfoNCE with class labels; maps directly to preset-ID contrastive training |
