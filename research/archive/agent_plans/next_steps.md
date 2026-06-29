# NERETRAK: Next Steps

> Synthesis of the best elements across all four plans into a concrete forward path.

---

## Executive Judgment

After comparing all four plans, the **Cursor plan is the strongest single document** to use as the primary framework reference—it has the best technical depth on `.vital` format specifics, the most comprehensive literature citations, and the clearest phased rollout (Phase A/B/C). However, it should be **augmented** with specific contributions from the other plans:

| Augmentation | Source |
|---|---|
| Classifier-free guidance MIDI dropout | Claude |
| Musically biased sampling priors | Claude |
| Multi-resolution STFT loss rationale | Claude |
| Set transformer for modulation graph head | Claude |
| Train on dry renders before adding effects | Claude |
| GPL licensing / dataset policy analysis | Codex |
| MIDI-conditioned JEPA diagram and formulation | Gemini |
| Canonical slot sorting for modulation loss | Gemini |

The **Gemini plan** is the most immediately readable for a new collaborator onboarding to the project. Consider using it as a "companion summary" document.

---

## The Critical Path (What Must Be Done First)

Before any ML work begins, three **blocking infrastructure tasks** must be completed. Everything else depends on them.

### Blocker 0: Vita environment setup and validation

- Install `vita` (`pip install vita`) and verify it works on the target OS/hardware.
- Run the init preset through `synth.render(60, 0.7, 1.0, 3.0)` and confirm audio output.
- Check Vita GitHub issues for Windows-specific build problems.
- **Pin the Vita version** in a requirements file. Vital's preset format has changed between minor versions; version drift will corrupt your dataset.

### Blocker 1: Schema extraction and parameter index map

This is the single most valuable early artifact. Without it, nothing else can be designed concretely.

```python
import vita
synth = vita.Synth()

# 1. Export init preset JSON
init_json = synth.to_json()

# 2. Enumerate all controls with metadata
controls = synth.get_controls()
for name, ctrl in controls.items():
    details = synth.get_control_details(name)
    # Record: name, min, max, default, scale_type

# 3. For each control, set to extremes, render probe note, observe change
# 4. Build: parameter_index.json with all scalar params
# 5. Enumerate modulation sources and destinations
```

**Output:** `docs/vital-schema.md` — a typed index of all ~775 parameters with:
- Index number (for fixed-length vector)
- Native range (min, max)
- Scale type (kLinear, kExponential, kQuadratic, kIndexed)
- Default value
- Parameter group (oscillator / filter / envelope / LFO / effect)
- Perceptual identifiability rating (high / medium / low) — initially estimated, refined empirically

### Blocker 2: Minimal render pipeline

A working pipeline that:
1. Samples a random preset (even just the init preset with random scalar perturbations).
2. Renders a 3-second audio clip with a given MIDI note via Vita.
3. Computes a log-mel spectrogram via nnAudio.
4. Stores `{preset_json, param_vector, audio_path, midi_note, midi_velocity}` as a dataset record.

Even 100 clips is enough to validate the pipeline end-to-end before scaling to 100k.

---

## Phase 0: Proof of Concept (Target: ~2–4 weeks of focused work)

**Scope:** Minimal Vital subset only.
- 1 oscillator (osc_1 only, basic waveform, no wavetables)
- 1 filter (filter_1, cutoff + resonance only)
- Amplitude envelope (env_1: attack, decay, sustain, release)
- No modulation routing
- No effects

**Dataset:** 10k examples, each preset rendered under 3 MIDI clips (monophonic: low note, mid note, high note).

**Model:** CNN or small Transformer encoder with nnAudio mel front-end → MLP parameter heads.

**Training:**
1. Supervised parameter regression only (MSE on continuous, cross-entropy on discrete).
2. No contrastive loss yet.

**Evaluation:**
- Parameter MAE per group
- Render predicted preset → compute mel distance to input
- Valid export rate

**Success criterion:** Predicted preset renders audio that is recognizably similar in timbre to the input. Mel distance should be lower than a random-preset baseline.

---

## Phase 1: Disentanglement (Target: after Phase 0 is stable)

**Add:** InfoNCE contrastive loss with preset-ID labels.

**Dataset expansion:**
- Each preset rendered under 5+ MIDI clips (monophonic, chords, different velocities, different octaves).
- Add **hard negatives**: synthetic doppelgänger presets (1–3 parameter jitter).

**Architecture change:**
- Add a disentanglement block splitting `z_timbre` and `z_content`.
- Add MIDI encoder (piano-roll input) for content conditioning.

**Key ablation:** Compare mel distance and parameter MAE with and without contrastive loss on a held-out test set of unseen MIDI patterns.

---

## Phase 2: Full Scalar Parameter Set

**Add:** All ~775 scalar controls (oscillator waveforms, filter models, all envelopes, LFOs, effect scalars).

**Design decisions needed:**
- Which parameters to include/exclude (e.g., skip parameters that are perceptually irrelevant for short clips).
- How to handle categorical parameters (filter model, oscillator waveform): ordinal vs. one-hot vs. embedding.

**Dataset scale:** 100k+ examples.

---

## Phase 3: Modulation Routing

**Add:** Modulation graph head.

**Architecture:** Set transformer over modulation slots (up to 64), each slot predicting `(source_id, dest_id, amount, bipolar)`.

**Key challenge:** Modulation routing is not identifiable from a single short clip unless the modulation is audible. Consider:
- Only training modulation head on clips where modulation is audible (LFO rate audible in the clip duration, envelope modulation clearly triggered).
- Loss: F1 on predicted routing edges (threshold amount predictions).

---

## Phase 4: Generative Decoder (Phase B in Cursor plan)

**Replace MLP heads with a flow matching decoder** conditioned on `z_timbre`.

This addresses:
- Permutation symmetries (effect chain order, modulation slot assignment)
- Multiple valid solutions (the output distribution is multimodal)
- Interactive usage (re-sample the decoder for alternative preset suggestions)

**Reference:** Equivariant flow matching paper (ISMIR 2025, arXiv:2506.07199). The relaxed equivariance strategy adapts to discovered symmetries from data, which is appropriate given Vital's complex symmetry structure.

---

## Phase 5: Wavetable Extension

**Add:** Factory wavetable index prediction (classification over ~80 built-in tables per oscillator).

**Later:** Custom wavetable latent (train a separate wavetable autoencoder, then predict latent).

**Note:** Verify factory wavetable licensing before including them in training data.

---

## Key Design Decisions to Resolve Before Implementation

These are open questions where the four plans diverged. They should be resolved explicitly before implementation begins.

### Decision 1: Encoder backbone
**Options:** EfficientAT (pre-trained, efficient), AST (pre-trained, transformer), CNN from scratch (fast iteration).
**Recommendation:** Start with **CNN from scratch** for Phase 0 (fastest iteration, no fine-tuning overhead). Switch to **frozen EfficientAT** in Phase 1 (proven best in Neural Proxies paper).

### Decision 2: Sample rate for rendering
**Options:** 44.1 kHz (Vital native, maximum fidelity), 22.05 kHz (half; faster rendering and smaller feature maps).
**Recommendation:** Render at **44.1 kHz**, downsample to **22.05 kHz** for the spectrogram layer. Preserves high-frequency rendering fidelity while reducing compute.

### Decision 3: Clip duration for training
**Options:** 1 second, 3 seconds, variable.
**Recommendation:** **3 seconds** — long enough to observe LFO motion and envelope release; short enough to keep compute tractable. Render at note duration 2.0s + 1.0s release tail.

### Decision 4: MIDI conditioning strategy
**Options:** Always required, optional auxiliary, CFG dropout.
**Recommendation:** **CFG dropout** — during training, randomly mask out the MIDI conditioning with probability 0.2. At inference, always provide Basic Pitch output but the model can handle its absence.

### Decision 5: Modulation routing representation
**Options:** Sparse edge classification (MLP), set transformer, transformer sequence.
**Recommendation:** **Set transformer** (Phase 3+). For Phase 0–2, skip modulation routing entirely.

### Decision 6: Loss weighting
Starting point (tune via ablation):
```
L_total = 1.0 · L_param + 0.1 · L_InfoNCE + 0.01 · L_proxy (if proxy model is added)
```

---

## Immediate Action Items (This Week)

- [ ] **Install and validate Vita** on the target development machine.
- [ ] **Run schema extraction script** — enumerate all controls and export `parameter_index.json`.
- [ ] **Render 10 test clips** — confirm audio output quality and file size.
- [ ] **Set up nnAudio** — confirm GPU-native mel spectrogram works on target hardware.
- [ ] **Install Basic Pitch** — confirm inference runs on a test audio clip.
- [ ] **Decide on dataset storage format** — HDF5 for bulk audio + JSON sidecar for metadata.
- [ ] **Pin all dependency versions** — Vita, torchaudio, nnAudio, Basic Pitch.

---

## Document Map

| Document | Owner | Status |
|---|---|---|
| `original_task.md` | Human | ✅ Done |
| `plan_codex.md` | Codex | ✅ Done |
| `plan_cursor.md` | Cursor | ✅ Done |
| `plan_gemini.md` | Gemini | ✅ Done |
| `plan_claude.md` | Claude | ✅ Done |
| `compare_plans.md` | Claude | ✅ Done |
| `next_steps.md` | Claude | ✅ Done |
| `docs/vital-schema.md` | TBD | ❌ Not started |
| `docs/dataset-spec.md` | TBD | ❌ Not started |
| `docs/model-architecture.md` | TBD | ❌ Not started |
| `docs/evaluation-protocol.md` | TBD | ❌ Not started |
