# Era 2 LFO Representation: Research Priors And Design Contract

Era 2 starts from a stricter question than Era 1:

> Can we build a compact LFO representation that a deployed audio model can actually predict?

Era 1 showed that residual codebooks can reconstruct dense LFO curves well under oracle encoding. It also exposed the main conceptual trap: topology-aware codebook construction was allowed to become topology-conditioned code selection. That made the representation harder to reason about from the point of view of an audio-only deployed model.

The Era 2 contract is simple:

> Topology may help build the codebook, but it must not select codes after the codebook exists.

That contract does not settle every design choice. It clears the ground so we can ask the next important question cleanly: how should the model address atoms at runtime, and how many predictions does each addressing scheme require?

## The Model-Facing Contract

The deployed model receives audio-derived features and emits reconstruction codes. It does not get hidden metadata.

The baseline model may predict:

```text
base code
base phase
residual-layer atom indices
residual-layer phases
```

Later experiments may add gain or offset, but they are not part of the first Era 2 baseline.

Era 2 uses a fixed uniform x lattice for LFO curve representation:

```text
control_point_count = 97
```

This is now a settled decoder-side choice. Experiment 10 showed that a
97-control-point uniform lattice is the best valid high-end uniform row under
Vital's 100-point limit. Its derived 96 subdivisions align with the corpus's
dominant dyadic x-position structure while also supporting factor-3
subdivisions. The deployed model should not predict x positions, grid spacing,
or per-LFO grid parameters.

The model must not:

- receive topology as input;
- receive x-grid locations as input;
- predict topology as an auxiliary target;
- predict control-point x positions or grid spacing;
- use topology-conditioned masks;
- use topology-conditioned losses;
- write topology into the model-facing artifact schema;
- use topology-conditioned decoder lookup;
- get model prediction head accounting that depends on topology.

Topology labels can still be useful. They may help choose, balance, or diversify atoms during offline codebook construction. But once the codebook exists, the target path must be expressible without topology.

## Capacity Means Model Prediction Head Budget

Era 2 should evaluate representation capacity by the **model prediction head budget**: the output nodes allocated at the end of the deployed model for reconstructing one LFO.

This is not the same as:

- codebook storage size;
- number of stored atom curves;
- oracle search cost;
- construction runtime;
- serialized path-field count.

Those costs still matter, but they are not the binding constraint for the audio model. The model-facing bottleneck is the prediction head:

```text
model prediction head budget = categorical logits + continuous scalar outputs
```

The concrete accounting field is still `head_outputs`. Older notes sometimes called this dense output burden. Era 2 should be more explicit: report the categorical-logit formula, the scalar formula, and the combined `head_outputs` count.

That is why two codebooks with the same number of stored atoms can have different model prediction head budgets, and two codebooks with different storage sizes can require the same number of prediction-head nodes.

The core accounting symbols:

- `B`: base dictionary size, currently `32`;
- `D`: actual number of residual layers;
- `W`: atom choices available to a residual layer in a flat dictionary;
- `S`: scalar outputs, `D + 1` for phase-only baseline;
- `n`: branching factor for a tree-addressed codebook;
- `L`: tree path length, usually `ceil(log_n(W))`.

For every Era 2 row, report the model prediction head budget formula and the actual `head_outputs`. Do not rely on a W/D label alone.

## Nomenclature

**Residual layer** is the model-facing unit: one atom choice plus the scalar values needed to apply that atom.

The codebase often calls this a `stage`. In Era 2 prose, use `stage` only when talking about literal implementation fields such as `PhaseChain.stages`, `stage_N_index`, or `stage_label`. Otherwise use residual layer.

**Residual-layer pair** refers to the historical 7A/8/9 implementation pattern: one shared residual layer plus one topology-conditioned residual layer. This is not the public depth.

**D** is the actual number of residual layers. It is never a residual-layer-pair count.

**W** is residual dictionary width per residual layer when the layer is addressed as a flat dictionary.

**Serialized fields** are compact stored path fields: selected indices, phase values, and any scalar values written into artifacts.

**Model prediction head budget** is the number of output nodes allocated to the LFO component at the end of the deployed model: categorical logits plus continuous scalar outputs. The concrete count is `head_outputs`.

**Codebook storage** is decoder-side table size. It matters for deployment, but it is not the model's output head.

**Dictionary scope** says where atoms live. A per-residual-layer dictionary gives each residual layer its own atoms. A global dictionary reuses one atom set across residual layers.

**Atom addressing** says how the model chooses an atom from the available set. Flat categorical and tree path addressing are different addressing schemes even if they expose the same number of atoms.

Dictionary scope and atom addressing are separate axes. Era 1 blurred these with topology. Era 2 should not.

## Atom Addressing Is A First-Class Design Axis

The model has to choose atoms somehow. That choice interface is the real dense-output driver. Era 2 should make it explicit before comparing quality.

### Option 1: Per-Residual-Layer Flat Categorical

Each residual layer has its own dictionary of `W` atoms. The model emits one `W`-way categorical head per residual layer.

Phase-only accounting:

```text
head_outputs = B + D * W + (D + 1)
```

This is the cleanest first baseline because it is easy to audit. Each residual layer has a local job: choose one of its own `W` atoms and a phase.

What it buys:

- simple target schema;
- simple loss schema;
- no topology;
- no sequential addressing assumptions;
- direct comparison to Era 1 budgets after correcting the interface.

What it costs:

- storage grows with `D * W`;
- the model pays `W` logits per residual layer;
- atom meanings are layer-local, so layer 7 atom 3 is not the same object as layer 12 atom 3.

This should be the first Era 2 baseline.

### Option 2: Global Flat Dictionary Reused At Every Residual Layer

All residual layers choose from one shared dictionary of `W` atoms. The model still emits one `W`-way categorical choice per residual layer.

Phase-only accounting:

```text
head_outputs = B + D * W + (D + 1)
```

The output burden is the same as Option 1 if every residual layer still needs its own atom choice. The difference is storage and construction objective, not the number of runtime logits.

What it buys:

- much smaller atom storage;
- atom identities are shared across residual layers;
- potentially better regularity for a learned model.

What it costs:

- the oracle must optimize one dictionary across residual distributions from all residual layers;
- a single dictionary may underfit layer-specific residual structure;
- quality comparisons against per-layer dictionaries must be made at equal model prediction head budget, not equal storage.

This is a real Era 2 design axis, but not the first baseline unless we also implement the corresponding joint construction objective.

### Option 3: Larger Shared Pool Available To Every Residual Layer

Instead of each residual layer having `W` local atoms, every residual layer can choose from a larger global pool of `C` atoms.

Phase-only accounting:

```text
head_outputs = B + D * C + (D + 1)
```

This is just flat categorical addressing over a larger available set. It may be useful if the same pool can serve many residual layers, but the model still pays for all available choices at every residual layer.

What it buys:

- maximum reuse;
- easy interpretation;
- no topology or sub-dictionary selection.

What it costs:

- output burden grows quickly if `C` is large;
- every residual layer faces the full choice set;
- oracle construction has to cover all residual-layer distributions with one pool.

This should be treated as a global-dictionary variant, not as a free way to increase codebook capacity.

### Option 4: Tree-Addressed Codebook

A residual layer exposes `W` leaf atoms, but the model does not emit one `W`-way categorical head. Instead, it emits a path through a tree. With branching factor `n`, the path length is:

```text
L = ceil(log_n(W))
```

If each tree decision is an `n`-way categorical head, phase-only accounting is:

```text
head_outputs = B + D * n * ceil(log_n(W)) + (D + 1)
```

For binary addressing:

```text
head_outputs = B + D * 2 * ceil(log_2(W)) + (D + 1)
```

This can be much cheaper than flat `D * W` logits, but it is not just an accounting trick. It changes the learning problem.

What it buys:

- lower model prediction head budget for large `W`;
- a way to expose larger codebooks under a fixed prediction-head budget;
- a natural path toward structured atom organization.

What it costs:

- path errors can send the decoder to the wrong subtree early;
- tree structure becomes part of the representation;
- losses may need to supervise multiple decisions per residual layer;
- the oracle must either build a meaningful tree or organize atoms after selection;
- comparisons against flat categorical rows need equal model prediction head budgets.

Tree addressing is promising, but it should be a later Era 2 experiment. It should not be bundled into the first topology cleanup baseline.

### Option 5: Topology-Conditioned Addressing

This is the old family: topology selects a dictionary row, or the report flattens topology-specific rows into one wider head.

Accounting if flattened:

```text
head_outputs = B + sum(layer_codebook_size) + S
```

where topology-conditioned residual layers charge `3W` instead of `W`.

This is valid as a scoped comparison family, but it is not the Era 2 deployed baseline. It violates the rule that topology must not participate in code selection.

## Equal-Budget Evaluation

Era 2 should compare representations at equal model prediction head budget.

That means equal or near-equal `head_outputs`, not equal:

- number of stored atoms;
- total codebook memory;
- `W * D` alone;
- oracle search complexity;
- wall-clock construction time;
- serialized field count.

The reason is practical. The deployed model has a finite prediction head for the LFO component. If one representation needs 305 output nodes and another needs 1089 output nodes, they are not equally costly just because their codebook storage is similar. If one representation stores more atoms but exposes them through a cheaper tree path, the storage increase is not the same kind of cost as a larger flat softmax.

A fair comparison table should include at least:

- addressing scheme;
- dictionary scope;
- `B`, `D`, `W`, and optional `n`;
- scalar families included;
- model prediction head budget / `head_outputs`;
- codebook storage;
- oracle construction time;
- median RMSE;
- P95 RMSE;
- strict perfect-LFO rate;
- any decoder policy used.

But the grouping axis should be model prediction head budget. Everything else explains why rows differ after the budget match.

## Priors Carried Into Era 2

Residual stacking is the main direction.

The best evidence from Era 1 points toward additive residual codebooks, not direct grids, as the compact reconstruction backbone. Direct grids remain useful as a fallback or editor-node warning, especially for custom-ish LFOs, but they are not the main compact representation.

Depth matters more than width right now.

The useful rows kept getting deeper. Wider dictionaries helped, but depth was the more reliable way to buy tail-error improvement per output. Era 2 should keep testing narrow/deep stacks instead of assuming a wide per-layer alphabet is the natural endpoint.

Model prediction head budget is the comparison axis.

W/D labels are not enough. Deeper chains add scalar phases. Wider dictionaries add categorical logits. Tree addressing changes the atom-choice formula entirely. Any fair comparison must use model prediction head budget.

Phase is baseline.

Phase alignment is how residual atoms become reusable corrections. A phase-free representation can exist as an ablation, but it is not the Era 2 baseline.

Gain and offset are not baseline.

Gain and offset have not earned default status. Gain may deserve a later targeted follow-up; offset is weaker so far. Both should wait until the topology-free representation contract is clean.

Decoder policies are allowed to be free, but not invisible.

Clipping and snap add no model outputs. That makes them decoder-policy choices, not representation-head choices. But they can change oracle target selection, so reports must not treat them as purely cosmetic post-processing.

Topology can shape atoms, not select codes.

Topology-balanced construction is still a plausible way to get a better dictionary. Topology-conditioned residual-layer lookup is not part of the deployable baseline.

## Settled Era 2 Decisions

The first Era 2 baseline is phase-only:

```text
base code + base phase + residual-layer atom index + residual-layer phase
```

The first baseline uses:

- fixed uniform `control_point_count=97`;
- constant `W`;
- actual `D` residual layers;
- per-residual-layer flat dictionaries;
- flat categorical addressing;
- no gain;
- no offset;
- no topology-conditioned residual-layer interface;
- no topology condition in encoding or path target generation;
- no topology condition in decoding;
- no topology-dependent masks or losses;
- no topology field in the model-facing artifact schema;
- no topology term in model prediction head accounting.

The fixed x lattice is decoder-owned. It does not add model prediction head
outputs. The derived subdivision count is 96, which matters for grid-alignment
discussion, but the vector-shape parameter is the 97 control-point count. The
model spends its LFO budget on base selection, residual-layer atom selection,
and phase scalars.

For phase-only accounting:

```text
head_outputs = 32 + D * W + (D + 1)
```

This is the baseline that Era 1 never cleanly supplied.

The following are explicitly out of the first baseline:

- topology sub-selection, whether flattened as `3W` or represented as bucket-plus-code;
- binary-tree indexing;
- variable per-layer `W`;
- gain/offset scalar families;
- global-shared dictionary claims without a construction objective that actually optimizes a global dictionary.

## Open Questions

Does topology-balanced construction still help when topology cannot participate in code selection?

This is the most important open question. Topology may still be useful as an offline sampling or balancing signal. It just has to disappear before model-facing targets exist.

Do per-layer dictionaries remain better than a true global dictionary at equal model prediction head budget?

Per-layer dictionaries are acceptable for the first baseline because they match the strongest Era 1 direction. A global dictionary is still interesting, but it needs its own construction objective. Its advantage is not fewer prediction-head nodes if each residual layer still emits a flat `W`-way choice; its advantage is reuse and storage.

How should tree-addressed codebooks be built?

A tree can reduce model prediction head budget, but only if atom organization is meaningful. The tree could be built by clustering atoms, by residual-error neighborhoods, by topology-blind shape similarity, or by learned confusion structure. This needs its own design pass.

How deep should narrow/deep stacks go under clean accounting?

Era 1 suggests depth is under-explored. Era 2 should budget-match narrow/deep rows directly under `32 + D * W + (D + 1)`, then later under tree-addressed formulas if tree addressing is tested.

Which clipping policy should become default?

Clipping is promising and cheap in model prediction head terms. The default should be chosen with median, P95, and strict perfect-LFO rate visible, because different clipping policies can trade common-case smoothness against tail repair.

When should gain return?

Gain should return only after the clean phase-only baseline is established. The first follow-up should be narrow and should avoid the known Experiment 9 residual-only accounting ambiguity.

Does the representation need a direct residual or fallback for custom/editor-faithful shapes?

Direct grids kept warning us about custom-ish shapes and node preservation. Era 2 can keep the core representation compact while still planning a fallback or refit path for shapes where sampled RMSE and editor-state faithfulness disagree.

## First Era 2 Experiment

The first run should answer one question:

> How much quality do we get from residual layering when topology is used only to build atoms, and the deployed model predicts only atom indices and phases from audio?

Use a clean phase-only, per-residual-layer flat-categorical baseline:

```text
head_outputs = 32 + D * W + (D + 1)
```

Model-facing targets:

```text
base_index
base_phase
residual_layer_1_index
residual_layer_1_phase
...
residual_layer_D_index
residual_layer_D_phase
```

No target, mask, loss, artifact column, decoder lookup, or model prediction head formula should depend on topology.

The first run should also hold the x lattice fixed:

```text
control_point_count = 97
```

No row should spend model prediction head budget on x-coordinate prediction,
grid selection, or variable grid spacing. Discussion may refer to the derived
96 subdivisions when explaining lattice alignment, but row configuration should
use the control-point count.

The first screen should compare by matched model prediction head budget. Existing topology-conditioned rows can provide budget anchors:

- around `305` outputs;
- around `577` outputs;
- around `1089` outputs.

Those are anchors, not baselines. The new rows must be topology-free in the model-facing interface.

Offline topology-aware construction can be tested as a controlled construction axis only if both variants produce the same topology-free target schema:

```text
construction A: topology-blind atom selection
construction B: topology-balanced atom selection
same deployed code interface
same model prediction head accounting
```

That isolates the actual question: does topology help choose better atoms when it is not allowed to select codes?

## Second-Wave Era 2 Comparisons

After the first clean baseline exists, compare atom-addressing schemes at equal model prediction head budget.

A useful second-wave grid would include:

- per-residual-layer flat dictionaries;
- global flat dictionary with joint construction;
- larger global pool if output budget allows;
- tree-addressed dictionaries with one fixed branching factor.

Each row should declare:

```text
dictionary_scope
addressing_scheme
D
W or C
n, if tree addressed
scalar_outputs
categorical_logits
head_outputs
```

Then compare quality at matched `head_outputs`. This is the point where "equal-capacity codebook" becomes meaningful: equal capacity means equal model prediction head budget, not equal memory size or equal oracle difficulty.

## How To Talk About Era 1 From Now On

The old rows are not invalid. They are scoped.

Use phrasing like:

> Inside the topology-conditioned family, W16D32 was the best-quality Experiment 8 row.

or:

> The W4/W6 narrow/deep rows are strong evidence that depth is worth testing under clean accounting.

Avoid phrasing like:

> W16D32 proves the deployable representation is output-efficient.

or:

> Topology is only an offline construction detail.

That second sentence is false for the current 7A/8/9 implementation. Topology is also used as side information for residual-layer lookup.

## Separate Work Items

The Experiment 9 affine accounting issue should be audited separately. The suspected `residuals_only` / `residual_only` mismatch makes residual-only gain/offset conclusions unsafe until verified.

Binary-tree indexing should be treated as atom-addressing research, not as part of the topology cleanup. It changes the learning problem and the cost formula.

Editor-node faithfulness should remain visible. Good sampled-curve RMSE is not the same thing as reconstructing the original editable Vital LFO node structure.

## The Era 2 Standard

An Era 2 baseline row is clean only if the deployed model can be described without topology:

```text
audio -> base code, base phase, residual atom indices, residual phases -> decoder -> LFO curve
```

If topology appears after codebook construction, the row is not a topology-free baseline. It may still be a useful experiment, but it answers a different question.

If two rows have different atom-addressing schemes, compare them at matched model prediction head budgets. The thing we are rationing is not primarily storage, code count, or oracle effort. The scarce resource is the number of LFO prediction-head nodes available for picking reconstruction codes.
