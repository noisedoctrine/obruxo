# OBRUXO: Parameter-Hungry Components

**Author:** antigravity and user  
**Timestamp:** 2026-06-21 18:30

This discussion note now lives with the LFO representation experiment. The wavetable section remains exploratory and is not yet part of the executable benchmark.


## Big Picture

Some Vital components are awkward neural-network outputs because their serialized state is large, sparse, variable-length, and often non-identifiable from audio. LFO geometry and wavetable-editor state are two useful examples.

The working idea is not to predict every raw field. Instead:

1. The model predicts a **simple, fixed-size dense representation** containing the most useful choices and continuous controls.
2. A deterministic component parser expands that representation into the **full, sparse Vital configuration**, handling defaults, inactive slots, ordering, interpolation, and serialization details.
3. We judge the representation by how much rendered-audio similarity it buys per predicted dimension. It does not need to reconstruct the sound by itself; it works alongside the oscillators, filters, modulation matrix, effects, and every other predicted subsystem.

This is an exploratory design space rather than a commitment to recover the exact historical patch. Several different configurations may be equally good answers if they produce the right perceptual result.


## Shared Modeling Assumptions

* **Compact prediction, complete expansion:** The neural decoder should have fixed tensor shapes. Variable-length component state is constructed after prediction using masks, codebooks, templates, and deterministic fitting.
* **Rendered behavior over JSON identity:** Exact source JSON is a poor primary target because equivalent patches can have very different parameterizations. Parsed JSON remains useful for learning the schema, estimating corpus priors, constructing templates, and checking validity.
* **Raw values at the boundary:** The final preset must contain the raw values Vital expects. The model itself can predict normalized, categorical, or perceptually shaped coordinates; the parser converts those into Vital's raw domains.
* **No direct waveform prediction:** The model does not emit raw wavetable frames. It may predict compact controls that the parser expands into editor operations or harmonic data.
* **Linear remap sparsity:** A modulation connection with a linear mapping should omit `line_mapping`; only non-linear remaps need a drawable curve object.


## Component 1: LFO Shapes (`settings.lfos`)

### What Is Actually Stored

Vital separates drawable LFO geometry from ordinary LFO controls. A serialized shape looks like:

```json
{
  "name": "Triangle",
  "num_points": 3,
  "points": [0.0, 1.0, 0.5, 0.0, 1.0, 1.0],
  "powers": [0.0, 0.0, 0.0],
  "smooth": false
}
```

`points` is an interleaved list of `(x, y)` pairs. Segment curvature values live in the separate `powers` array; the shape is not serialized as `(x, y, z)` triplets. With `N` points there are roughly `3N` stored continuous numbers, although endpoint constraints and apparently unused boundary values reduce the effective degrees of freedom. Eight 100-point LFOs still give a useful worst-case warning of roughly 2,400 stored numbers, but this is not representative of normal use.

Rate, sync mode, tempo division, phase, stereo offset, delay, fade, smoothing, and keytracking are ordinary scalar controls outside `settings.lfos`. They should be predicted by the normal grouped parameter heads rather than folded into the shape representation.

### What We Currently Mean by "Custom"

The existing corpus analyzer hashes each shape, excluding its display name, and compares it with the most common shape for that Vital version. This detects **non-default geometry**, not provenance. It cannot by itself tell whether a shape was selected from Vital's built-in LFO preset menu, selected from a template and then adjusted, or drawn manually from scratch.

The distinction matters because regular menu shapes should be cheap categorical predictions, not evidence that most presets need arbitrary curve generation.

A controlled preset, `STOCK_LFO_TWEAKED.vital`, helps separate the cases. LFO1 was changed using ordinary dropdowns and knobs and routed to oscillator level. Its shape object remained exactly equal to the untouched Triangle objects in LFO2–8. The changes appeared only in scalar fields such as `sync_type`, `sync`, `tempo`, `phase`, `delay_time`, `stereo`, and `smooth_time`. The current geometry hash therefore correctly does **not** mistake these edits for a custom drawable shape.

The remaining ambiguity is stock shape selection. Corpus shapes carry names such as `Sin`, `Saw Up`, `Saw Down`, `Square`, `Side Chain`, and `Trance Gate`, but the geometry is baked into the same point-and-power representation. Names are useful evidence, not ground truth: edited shapes often retain names such as `Triangle`.

Vital's 15 default LFO shape presets give us a natural seed codebook:

* `Saw Down`, `Saw Up`, `Sin`, `Square`, `Staircase Down`, and `Triangle`;
* `Growing Oscillations`, `Nervous Groove`, `Pulse Series`, and `Random Pulses`;
* `Shuffle Gate`, `Side Chain 1`, `Side Chain 2`, `Split Gate`, and `Trance Gate`.

The first codebook experiment should use these known stock entries rather than discovering all clusters from scratch. Each entry should be stored by canonical geometry, not name alone. The model can then predict one of the 15 stock shapes, plus `edited/residual` and `free-form/unknown` paths. Corpus-derived additions should earn inclusion by covering a meaningful share of shapes not already approximated by this vocabulary.

The corpus statistic previously described as "64.7% use a custom LFO shape" should therefore be read as "64.7% use at least one non-default serialized LFO shape." A deeper inspection found 10,203 three-point shapes among 16,534 routed LFO instances, reinforcing the idea that the common case is highly regular even though a long free-form tail exists.

A more useful descriptive taxonomy would be:

1. **Default shape:** exact match to Vital's default geometry.
2. **Canonical stock template:** exact geometry match to one of the 15 built-in LFO presets.
3. **Template-derived:** close to a canonical template but modified.
4. **Free-form or unknown:** no credible template match.

We can collect canonical stock templates by selecting each built-in LFO preset and saving its state. Historical provenance beyond these behavioral categories is probably not recoverable, nor is it required for perceptual reconstruction.

### Dense Prediction and Sparse Expansion

A practical dense output could contain shape-family or codebook logits, a `free_form` or residual-presence probability, a small fixed residual grid such as 32 or 64 sampled values, and the ordinary scalar LFO controls in their own grouped head.

The parser would retrieve the selected canonical shape, add any predicted residual in function space, fit the result to valid Vital points and powers, simplify it within an error tolerance, enforce endpoint/loop/point-count constraints, and omit unnecessary custom geometry when the canonical shape is sufficient.

This gives the common regular shapes a very cheap path while preserving an escape hatch for unusual curves.

### Candidate Modeling Paths

* **Codebook + residual:** Classify a canonical or corpus-derived shape, then predict a small sampled residual. This currently looks like the strongest default because the distribution is dominated by simple, repeated structures.
* **Grid regression + native-curve fitting:** Predict the whole LFO function on a fixed grid, then fit and simplify it into Vital's point/power format. Ramer–Douglas–Peucker may help choose nodes, but geometric simplification alone is not enough; the fitter must also recover curvature and preserve discontinuities and periodic boundaries.
* **Parametric segment prediction:** Predict a bounded number of segment widths, endpoint values, and curvature powers. This maps closely to Vital but needs masks or a stopping mechanism for variable segment count.

Shape importance depends on context. Errors that are harmless in a slow filter sweep may be obvious when the LFO controls amplitude, pitch, or rapid wavetable movement. Evaluation should therefore include the predicted rate, routing destination, and modulation amount rather than assigning one universal perceptual weight to every point.


## Component 2: Wavetable Editor State (`settings.wavetables`)

### The Actual Compression Problem

Frame playback interpolation is not the hard part. Once a wavetable exists, mapping position `0–1` to neighboring frames is a small deterministic operation. The difficult question is how to expose some of the editor's useful timbral range without asking the model to emit raw frames or a huge sparse component graph.

We do not need to infer the exact editor recipe that historically created the reference. We need a compact output that the parser can turn into a valid, controllable wavetable configuration that helps the complete synth reproduce the audio.

The editor state still has useful internal structure:

* source or stock wavetable identity;
* table-level settings and morph behavior;
* an ordered chain of modifiers;
* sparse keyframe positions and local edits;
* playback position and oscillator morph controls downstream.

Global operations should be predicted once. Local edits should consume capacity only in active keyframe slots. Modifier order must be retained where operations are non-commutative.

### Dense Prediction and Sparse Expansion

There are several reasonable levels of ambition.

#### Template-based output

The model predicts template logits, stock-source logits, an active-keyframe mask, and a small continuous parameter vector. The parser selects the recipe skeleton, retrieves the stock source, fills its active controls, and emits only the required components and keyframes.

Example template:

```text
stock source -> spectral morph -> one global modifier -> one edited keyframe
```

This is restrictive, but it offers a clean way to test whether a little editor flexibility produces enough perceptual gain to justify a more general representation.

#### Retrieval + compact residual

The model predicts a stock wavetable or recipe ID plus a small residual description:

* global modifier controls;
* a few active keyframe positions;
* individual low-harmonic deltas;
* a compact high-frequency spectral envelope.

The parser retrieves the base content, expands the residual across the full harmonic representation, applies it only at active keyframes, and builds the complete sparse configuration. This is attractive because the model spends dimensions on the difference from a useful source rather than rebuilding it.

#### Fixed hierarchical slots

The model predicts a bounded set of dense slots:

```text
M modifier slots:
  presence, type, scope, order, compact parameters

K keyframe slots:
  presence, position, edit type, compact edit payload
```

The parser masks inactive slots, sorts modifiers by order, sorts keyframes by position, expands each type-specific payload, and enforces valid editor structure. This is more expressive than templates but also introduces a harder mixed discrete/continuous search problem.

The parser—not the neural network—should own enum validity, slot omission, default insertion, chain ordering, keyframe ordering, parameter expansion, and final serialization.

### A Reasonable Harmonic Pareto

The low/high harmonic split is a compression heuristic, not a claim that hearing has a universal boundary at exactly harmonic 12. A useful starting representation might predict:

* individual magnitudes for the first 12 harmonics;
* optional compact phase controls for only the lowest few harmonics;
* 8–16 log-spaced values describing the remaining spectral envelope;
* normalization or total-energy controls.

The parser expands this into the full harmonic array using deterministic interpolation and tapering. Vital's filters, oscillator morphing, distortion, unison, effects, and other components can compensate for detail omitted by this basis. The representation only needs to add useful timbral degrees of freedom to the complete patch.

The meaningful experiment is a Pareto comparison: for the same output dimensionality, compare explicit low harmonics plus compressed upper bands against uniform coarse bins, PCA/codebook coefficients, and retrieval without residual editing. Small sweeps such as 8, 12, and 16 explicit low harmonics are enough to test whether the proposed split earns its dimensions.

### Where a Differentiable Renderer Fits

A differentiable editor or synth proxy is a possible credit-assignment mechanism, not the output representation. It receives the parser's expanded configuration and predicts rendered audio features so perceptual loss can train the compact decoder.

Approximating the complete editor pipeline would be expensive and is only justified if simpler patch-to-audio proxies, black-box search, or reranking cannot provide useful credit. Bilinear frame lookup by itself does not address editor-state inference.


## Training and Evaluation Direction

Direct supervision against arbitrary preset JSON is highly unpreferable as the governing training signal. It rewards recovery of one historical parameterization even when another compact configuration renders the right sound.

The preferred conceptual loop is:

```text
reference audio
  -> compact dense component predictions
  -> deterministic expansion into a valid Vital patch
  -> Vital render or differentiable proxy
  -> perceptual/spectral loss
```

For controlled synthetic data, we can sample the compact variables directly, expand them, and render the resulting patches. This supplies clean labels in the representation we actually chose without teaching the model to imitate arbitrary source JSON. Existing presets remain useful for discovering common templates, codebooks, parameter ranges, co-occurrences, and edge cases.

Evaluation should report both rendered quality and representation cost. The useful question for each extension is: how much held-out perceptual improvement does it add per predicted dimension and per unit of decoder/parser complexity?
