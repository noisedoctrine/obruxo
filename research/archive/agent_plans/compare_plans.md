# NERETRAK Plan Comparison: Codex vs. Cursor vs. Gemini vs. Claude

---

## Overview

Four AI systems independently produced framework plans for NERETRAK. All four share the same core thesis (Vita-rendered synthetic dataset + disentanglement encoder + structured preset decoder), but diverge significantly in **depth, specificity, and research grounding**. This document systematically compares them.

---

## 1. Structural Completeness

| Dimension | Codex | Cursor | Gemini | Claude |
|-----------|-------|--------|--------|--------|
| `.vital` format analysis | ✅ Good | ✅ Excellent | ✅ Good | ✅ Good |
| ML representation design | ✅ Good | ✅ Excellent | ✅ Good | ✅ Good |
| Audio-to-MIDI recommendation | ✅ Basic Pitch | ✅ Basic Pitch | ✅ Basic Pitch | ✅ Basic Pitch |
| Spectrogram front-end | torchaudio | nnAudio | nnAudio | nnAudio |
| Encoder backbone | CNN/Transformer (generic) | EfficientAT/AST (specific) | AST (specific) | EfficientAT (specific) |
| Contrastive/JEPA treatment | ✅ Discussed | ✅ Excellent | ✅ Excellent | ✅ Excellent |
| Generative decoder (flow matching) | ❌ Not mentioned | ✅ Yes (Phase B) | ❌ Not mentioned | ✅ Yes (Phase B) |
| Modulation routing architecture | ✅ Sparse graph | ✅ Sparse graph | ✅ Adjacency/sequence | ✅ Set transformer |
| Phased rollout plan | ✅ Yes | ✅ Yes (A/B/C) | ✅ Yes (1–4) | ✅ Yes (0–6) |
| Evaluation protocol | ✅ Comprehensive | ✅ Comprehensive | ⚠️ Brief | ✅ Comprehensive |
| Dataset quality / prior design | ⚠️ Mentions it | ⚠️ Mentions it | ❌ Not addressed | ✅ Addressed explicitly |
| Renderer speed / proxy strategy | ✅ Mentioned | ✅ Mentioned | ✅ Mentioned | ✅ Detailed |
| Permutation symmetry problem | ❌ Not addressed | ✅ Yes (key insight) | ⚠️ Modulation slots only | ✅ Yes (all forms) |

---

## 2. Topic-by-Topic Analysis

### 2.1 `.vital` File Structure

**Cursor** is the most technically precise. It correctly identifies:
- The two-surface problem (VST `setParameter` vs. Vita JSON API)
- That modulation routing is **inaccessible via VST**—only JSON
- Exact scale types from `synth_parameters.cpp` (kLinear, kExponential, kQuadratic, kIndexed)
- That `effect_chain_order` encodes a permutation as a float (categorical, not continuous)
- Wavetable keyframes are base64-encoded float32 (~2048 samples/frame × keyframes × 3 oscillators)

**Codex** is correct but less detailed; it provides the right intuition about typed representation but doesn't call out the VST surface limitation.

**Gemini** provides a clear, well-formatted table but doesn't fully explore the VST vs. JSON API distinction.

**Claude** explicitly calls out both surfaces and the "two-surface" problem, the quartic envelope normalization issue, and the identifiability/permutation problem at the representation level.

**Winner**: Cursor (most technically rigorous); Claude close second.

---

### 2.2 Audio-to-MIDI

All four plans converge on **Basic Pitch** as the recommendation. The differentiation is in framing:

- **Codex**: Correct analysis; treats MIDI as auxiliary and recommends three distinct use modes (ground truth, estimated, contrastive grouping).
- **Cursor**: Best treatment. Clearly separates training vs. inference role; recommends ablation; notes transcription errors on complex polyphonic synth sounds. Also the only one to mention a **TypeScript/browser variant** (vita-node / Basic Pitch TS).
- **Gemini**: Clear and correct; provides a code snippet; notes the MIDI conditioning is not used at training time.
- **Claude**: Adds **classifier-free guidance-style MIDI dropout** during training so the model can infer without MIDI at inference, which none of the others explicitly propose.

**Winner**: Cursor for completeness; Claude for the CFG dropout insight.

---

### 2.3 Spectrogram Front-End

All four recommend GPU-native spectrogram layers. Key differences:

- **Codex**: Recommends `torchaudio` (standard but not GPU-native for transforms). Does not mention nnAudio.
- **Cursor**: Recommends `nnAudio` with correct reasoning (1D conv layer, 50–100× faster than librosa), provides working code snippet, and mentions ablating trainable vs. fixed mel basis.
- **Gemini**: Also recommends nnAudio; provides code snippet that is nearly identical to Cursor's.
- **Claude**: Recommends nnAudio; adds emphasis on **multi-resolution STFT loss** (3 window sizes to capture different time-scales of Vital parameters), and explains *why* each feature type matters for Vital specifically.

**Winner**: Cursor and Claude are tied. Codex is weakest here (torchaudio instead of nnAudio).

---

### 2.4 Contrastive / JEPA Objectives

This is the area with the most differentiation:

- **Codex**: Covers triplet, InfoNCE, and JEPA conceptually. Describes a music-specific JEPA variant (context = render under MIDI A, target = render under MIDI B). Solid but somewhat generic.
- **Cursor**: Most comprehensive. References five distinct methods (triplet, InfoNCE, synthetic doppelgängers, MERIT, masked triplet), provides clear positive/negative pair formulation, explicitly distinguishes JEPA (pretraining) from InfoNCE (core objective), and calls out the **equivariant flow matching paper** as a generative alternative for Phase B+.
- **Gemini**: Provides concrete math (InfoNCE formula) and a detailed JEPA diagram. Introduces the MIDI-conditioned JEPA variant clearly. Doesn't mention the flow matching alternative.
- **Claude**: Introduces **preset mutation families** (hierarchical identity), **category-level negatives** (same oscillator/different envelope), **classifier-free guidance MIDI dropout**, and explicitly analyzes when InfoNCE vs. JEPA should be used for this specific problem.

**Winner**: Cursor (breadth of references); Claude (novel insights specific to NERETRAK).

---

### 2.5 Architecture

| Aspect | Codex | Cursor | Gemini | Claude |
|--------|-------|--------|--------|--------|
| Mermaid diagram | ❌ | ✅ (both!) | ✅ | ❌ |
| Encoder backbone | Generic | EfficientAT/AST | AST | EfficientAT |
| Modulation head | Sparse edge | Sparse graph | Adjacency/transformer | Set transformer |
| LFO curve head | Mentioned | 32-point resample | 32-point resample | 32-point resample |
| Generative decoder | ❌ | ✅ | ❌ | ✅ |
| MIDI conditioning strategy | Auxiliary | Auxiliary + ablation | Auxiliary | CFG dropout |

**Winner**: Cursor (most complete); Gemini and Claude close.

---

### 2.6 Dataset Generation

- **Codex**: Lists MIDI variety types (single notes, chords, phrases, held notes). Mentions preset family/mutation lineage in stored metadata. Good.
- **Cursor**: Cleanest phased description (Phase A/B/C). Notes the `sample_on=0` training constraint. Mentions Vita version pinning.
- **Gemini**: Brief on dataset design; doesn't address sampling priors.
- **Claude**: Explicitly addresses the **musically biased sampling prior problem** (uniform random parameters produce many silent/unusable patches), proposes specific distribution shapes (Beta for levels, log-uniform for times), and suggests a diversity metric to guard against degenerate datasets.

**Winner**: Claude uniquely addresses dataset quality; Codex covers variety well.

---

### 2.7 Risks and Open Questions

| Risk | Codex | Cursor | Gemini | Claude |
|------|-------|--------|--------|--------|
| Identifiability/perceptual equivalence | ✅ | ✅ | ✅ | ✅ |
| Wavetable recovery | ✅ | ✅ | ✅ | ✅ |
| Effects masking core identity | ✅ | ❌ | ❌ | ✅ (train on dry renders first) |
| Renderer speed | ✅ | ✅ | ✅ | ✅ |
| Vita version drift | ✅ | ✅ | ❌ | ✅ |
| Permutation symmetry | ❌ | ✅ | ⚠️ (slots only) | ✅ |
| Licensing | ✅ | ❌ | ❌ | ❌ |
| Dataset distribution gap | ⚠️ | ⚠️ | ❌ | ✅ |

**Winner**: Claude (most complete risk table); Codex second (mentions licensing).

---

## 3. Unique Contributions Per Plan

### Codex unique contributions
- Explicitly addresses **GPL licensing** of Vital factory presets (important for dataset policy)
- Clearly proposes the render-and-compare validation smoke test as part of the export pipeline
- Recommends constraining to a narrow Vital subset first, then expanding

### Cursor unique contributions
- **Best technical detail on `.vital` format** (scale types, effect chain order as permutation, DawDreamer's mod-matrix limitation)
- References the **equivariant flow matching** (ISMIR 2025) paper—a critical insight not in the original task prompt
- References **SynthRL** (IJCAI 2025) and **Neural Proxies** papers by name
- Provides **vita-node** as a Node.js alternative
- Phase A/B/C rollout is the cleanest structured plan
- Cites the most papers (10+ with correct arXiv IDs)

### Gemini unique contributions
- **Most readable** plan; best formatting with tables and mermaid diagrams
- Concrete `nnAudio` code snippet with correct parameter choices
- Best JEPA diagram (ASCII art showing the predictor architecture)
- Modulation slot permutation mitigation: canonical alphabetical sorting

### Claude unique contributions
- **Classifier-free guidance MIDI dropout**: train model to infer with or without MIDI conditioning so it degrades gracefully at inference
- **Preset mutation families** as a training hierarchy (family-level + preset-level identity)
- **Category-level hard negatives** (same oscillator/filter, different envelope)
- **Musically biased sampling priors** with specific distribution shapes
- **Multi-resolution STFT loss rationale** tied specifically to Vital's time-scale hierarchy
- **Set transformer** for modulation graph head (better architecture for unordered set prediction)
- **Effects masking**: recommends training on dry renders first

---

## 4. Consensus Points (All Four Plans Agree)

These are the least-controversial design decisions—safe to treat as settled:

1. **Vita is the right tool** for schema extraction, dataset generation, and `.vital` export/validation.
2. **Basic Pitch** is the right FOSS audio-to-MIDI tool for inference-time conditioning.
3. **nnAudio** (or equivalent GPU-native layer) is preferred over offline librosa preprocessing.
4. **MIDI should be auxiliary, not required**—the model should accept audio-only at inference.
5. **Do not regress raw JSON blobs**—use a typed, fixed-length vector representation with separate heads.
6. **Contrastive training** using same-preset/different-MIDI pairs is central to disentanglement.
7. **Phased rollout** starting with a restricted Vital subset (1–2 oscillators, 1 filter, amp envelope).
8. **Wavetables are deferred** to a later phase; Phase A uses factory wavetable index prediction.
9. **Evaluation requires both parameter-level and audio-level metrics**—they diverge.
10. **Renderer speed / Vita version drift** are practical bottlenecks that need explicit management.

---

## 5. Significant Points of Disagreement

| Question | Codex | Cursor | Gemini | Claude |
|----------|-------|--------|--------|--------|
| Spectrogram front-end | torchaudio | nnAudio | nnAudio | nnAudio |
| Primary encoder backbone | Generic CNN/Transformer | EfficientAT (from Neural Proxies paper) | AST | EfficientAT |
| Generative decoder | Not discussed | Flow matching (Phase B) | Not discussed | Flow matching (Phase B) |
| MIDI at inference | Auxiliary | Auxiliary + ablation | Required for JEPA | CFG dropout (optional) |
| Modulation head architecture | Sparse edge classification | Separate routing head | Transformer sequence | Set transformer |
| Training signal for non-differentiable synth | Perceptual loss via rendering | Neural proxy OR render loop | Neural proxy | Neural proxy + SynthRL |
