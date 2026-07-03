# Era 2 Runtime Atom-Selection Budgets

This note is a design research pass, not an experiment plan. It answers one
question we need to settle before planning the next LFO run:

> If topology is gone from the deployed runtime path, what are the clean ways
> an audio model can choose residual atoms, and what does each way cost in
> model prediction-head nodes?

The answer matters because "codebook capacity" is easy to talk about
incorrectly. For Era 2, the scarce resource is not primarily codebook memory,
oracle search time, or the number of atoms we can store. The scarce resource is
the **model prediction head budget**: the output nodes allocated at the end of
the deployed model for the LFO component.

The deployed model receives audio-derived features only. It must emit enough
information for the decoder to reconstruct the LFO curve. Topology may be used
offline to help build or balance a codebook, but topology must not appear in
runtime inputs, targets, losses, masks, decoder lookup, artifact schemas, or
head-output accounting.

## Core Accounting

Use these symbols unless an experiment row states otherwise:

- `B`: base dictionary size. The current Era 2 default is `32`.
- `D`: number of residual layers.
- `S`: reconstruction scalar outputs. For the phase-only baseline,
  `S = D + 1`: one base phase plus one phase per residual layer.
- `W`: flat atom choices available to one residual layer.
- `C`: global atom choices available to each residual layer.
- `P`: basis coefficients emitted for one residual layer.
- `n_l`: branching factor at tree/path level `l`.
- `L`: number of tree/path levels.
- `E`: dimensionality of a continuous address vector.

The general formula is:

```text
head_outputs = base_selection_outputs
             + residual_atom_selection_outputs
             + reconstruction_scalar_outputs
```

For the current base dictionary:

```text
base_selection_outputs = B = 32
```

For the current phase-only scalar contract:

```text
reconstruction_scalar_outputs = S = D + 1
```

So most of this note is about the middle term:

```text
residual_atom_selection_outputs
```

That term changes depending on how atoms are addressed.

## Acceptance Criteria

A runtime atom-selection method is worth carrying into Era 2 only if it passes
these checks:

1. It is compatible with an audio-only deployed model.
2. It has a clear model prediction-head budget formula.
3. It can be implemented and tested without redesigning the whole LFO system.
4. It is already a known pattern in classification, vector quantization,
   compression, retrieval, or codebook addressing.
5. It does not smuggle topology or oracle-only metadata into the runtime path.

This note intentionally separates **dictionary scope** from **atom addressing**.

Dictionary scope says where atoms live:

- per-residual-layer dictionaries;
- one global dictionary reused by all residual layers.

Atom addressing says what the model emits to select an atom:

- one flat categorical choice;
- a small coefficient vector that synthesizes the residual correction;
- a path through a tree-like address;
- a continuous address vector followed by nearest-neighbor lookup.

Those are independent axes. A global dictionary does not automatically reduce
the prediction-head budget. A tree address does not automatically make the
codebook good. A continuous address does not automatically make a large
codebook easy to learn.

## External Anchors

The accepted mechanisms below are not new inventions. They map to established
families:

- flat categorical prediction is the ordinary neural classifier / softmax head;
- hierarchical or path-style classification is related to hierarchical softmax,
  introduced to reduce large-vocabulary prediction cost;
- basis-coefficient reconstruction is related to sparse coding, dictionary
  learning, and non-negative matrix factorization;
- continuous codebook addressing is related to vector quantization and VQ-VAE,
  where a model uses discrete codebook entries as latent representations.

Useful references:

- [Softmax function and hierarchical softmax overview](https://en.wikipedia.org/wiki/Softmax_function)
- [Neural Discrete Representation Learning / VQ-VAE](https://arxiv.org/abs/1711.00937)
- [Learning the parts of objects by non-negative matrix factorization](https://doi.org/10.1038/44565)
- [K-SVD: An Algorithm for Designing Overcomplete Dictionaries for Sparse Representation](https://doi.org/10.1109/TSP.2006.881199)

The references justify the mechanism families. They do not settle which one is
best for LFO reconstruction.

## Approaches In Plain Terms

The methods below are different answers to the same runtime question:

> When a residual layer needs a correction atom, what exactly does the model
> predict?

The simplest answer is **flat categorical selection**. The residual layer gets
a menu of atoms, and the model chooses one item from that menu. If the menu has
`W` atoms, the model pays `W` logits for that residual layer. This is the
baseline because the accounting is blunt and auditable. Quick Big-O for the
residual atom-selection head: `O(DW)`.

The next answer is **basis-coefficient reconstruction**. The residual layer
does not choose one atom from a menu. Instead, it emits a small vector of
continuous coefficients, and the decoder combines a fixed set of basis curves
into the residual correction. The residual-layer head scales as `O(DP)`, where
`P` is the number of coefficients per residual layer.

The tree-shaped answer is **path addressing**. The model walks through a fixed
address path: first choose a branch, then a branch under that, and so on until
it reaches a leaf atom. This can turn a large flat choice into several small
branch choices. The important caveat is that Era 2 only gets the cheap formula
if the model really emits path-level decisions, not a separate classifier for
every internal tree node. With fixed small branching factor, the residual
atom-selection head can scale like `O(D log W)`.

The most different answer is **continuous addressing**. The model emits a small
continuous vector, and the decoder finds the nearest atom in an embedding space.
This can make the head budget independent of the number of stored atoms, but it
turns atom selection into a geometry problem rather than a direct classification
problem. If address dimension is fixed, the residual atom-selection head is
`O(D)` with respect to stored atom count.

All four accepted approaches can be topology-free. The question is not whether topology
is present; it is what the model must emit, how many head nodes that requires,
and how much reconstruction quality we can buy at equal model prediction head
budget.

## Accepted Method 1: Flat Categorical Per Residual Layer

**Claim:** This is the clean baseline. Each residual layer chooses one atom from
a flat set of available atoms.

**Concept:** Think of each residual layer as having a fixed menu of correction
shapes. The model looks at the audio-derived features and says, "for this
residual layer, use item 11 from the menu." It does that independently for each
residual layer, then emits the phase values needed to place those corrections.

This is not clever, and that is the point. If the menu has `W` items, the model
needs `W` logits to choose from it. There is no hidden routing, no topology
bucket, and no extra runtime fact deciding which menu entries are available.

Quick Big-O:

```text
residual atom-selection head: O(DW)
```

The model emits:

```text
base_index logits: B
residual_layer_1 atom logits: W_1
...
residual_layer_D atom logits: W_D
scalars: S
```

The decoder receives one atom index per residual layer plus the scalar values
needed to apply those atoms.

For variable width by residual layer:

```text
residual_atom_selection_outputs = sum_d W_d
head_outputs = B + sum_d W_d + S
```

For constant `W`:

```text
residual_atom_selection_outputs = D * W
head_outputs = B + D * W + S
```

For the phase-only baseline:

```text
head_outputs = 32 + D * W + (D + 1)
```

This method is clean under the no-runtime-topology contract if `W_d` is defined
by residual layer only. The decoder already knows which residual layer it is
applying. Residual-layer identity is not side information supplied by the
dataset; it is part of the fixed decode loop.

### Dictionary Scope Variants

Per-residual-layer dictionaries:

```text
residual layer d chooses from its own W_d atoms
```

This is easy to audit. Atom `3` in residual layer `7` is not the same object as
atom `3` in residual layer `8`.

Global dictionary reused at every residual layer:

```text
each residual layer chooses from the same C atoms
```

Accounting:

```text
residual_atom_selection_outputs = D * C
head_outputs = B + D * C + S
```

If `C = W`, the global dictionary and the per-residual-layer dictionary have the
same model prediction-head budget. The difference is storage, atom sharing, and
the oracle construction objective. Global storage reuse is not a free reduction
in output-head cost.

### What Future Rows Must Log

- `addressing_scheme = flat_categorical`
- `dictionary_scope = per_residual_layer` or `global`
- `D`
- `W_by_residual_layer` or `C`
- `categorical_outputs = sum_d W_d` or `D * C`
- `scalar_outputs = S`
- `head_outputs = B + categorical_outputs + S`
- `topology_used_at_runtime = false`

### Budget Envelope

For a fixed number of exposed choices per residual layer, flat categorical is
the high-clarity, high-cost reference point:

```text
best case:  D * W  residual atom-selection outputs when W is small
worst case: D * W  residual atom-selection outputs when W must grow large
```

There is no hidden compression in the address. If a residual layer can choose
from `W` atoms, the model emits `W` logits for that layer.

## Accepted Method 2: Basis-Coefficient Reconstruction

**Claim:** A residual layer can stop selecting a single atom and instead emit a
small set of coefficients for a fixed basis of residual curves.

**Concept:** This is the distinct alternative found in the literature pass. It
comes from the sparse coding / dictionary learning / NMF family: represent a
signal as a weighted combination of learned parts instead of assigning it to one
prototype. In LFO terms, one residual layer would store `P` basis curves. The
model emits `P` coefficients. The decoder reconstructs the layer's correction
by combining those curves.

This is not a tree. There is no path, no branch, and no hidden sequence of
decisions. It is also not nearest-neighbor lookup. The predicted coefficients
are the reconstruction code.

The tradeoff is conceptual: this method is no longer single-atom selection. It
is still a clean residual-layer runtime strategy, but the residual layer's unit
becomes:

```text
P coefficients -> weighted basis sum -> residual correction
```

instead of:

```text
one atom index -> one stored atom -> residual correction
```

Quick Big-O:

```text
residual basis-coefficient head: O(DP)
if P is fixed: O(D)
```

The model emits, for each residual layer:

```text
coefficient_1
coefficient_2
...
coefficient_P
```

The decoder receives:

```text
coefficients c_1..c_P
basis curves phi_1..phi_P
```

and reconstructs:

```text
residual_layer_curve = sum_p c_p * phi_p
```

The model prediction-head cost is:

```text
residual_reconstruction_outputs = D * P
head_outputs = B + D * P + S
```

For the phase-only baseline:

```text
head_outputs = 32 + D * P + (D + 1)
```

This method is clean under the no-runtime-topology contract if the basis curves
are fixed before deployment and coefficients are predicted from audio-derived
features only. Topology may help offline basis construction only if it
disappears before model-facing targets and decoder lookup.

The basis can be constrained in several simple ways:

- signed dense coefficients, like a small linear basis;
- non-negative coefficients, closer to NMF and additive parts;
- sparse coefficients, where training encourages most coefficients to be zero.

The head-budget formula is the same for all three if the model emits all `P`
coefficients. Sparsity changes the target distribution and decoder behavior,
not the number of output nodes, unless we introduce an explicit sparse-index
scheme later.

### What Future Rows Must Log

- `addressing_scheme = basis_coefficients`
- `dictionary_scope = per_residual_layer_basis` or `global_basis`
- `D`
- `basis_count = P`
- `coefficient_constraint = signed`, `nonnegative`, or `sparse_regularized`
- `basis_construction_policy`
- `coefficient_target_policy`
- `continuous_basis_outputs = D * P`
- `scalar_outputs = S`
- `head_outputs = B + continuous_basis_outputs + S`
- `topology_used_at_runtime = false`

### Budget Envelope

This method does not expose `W` discrete atom leaves, so it should not be
evaluated as if `P` and `W` were the same kind of capacity. The fair comparison
is matched model prediction head budget:

```text
flat categorical budget: D * W
basis coefficient budget: D * P
```

At equal budget:

```text
P ~= W
```

But the meaning is different. Flat categorical buys one choice among `W` stored
curves. Basis coefficients buy a `P`-dimensional continuous family of curves.

Best practical case:

```text
small P captures the major residual-shape variation smoothly
```

Worst practical case:

```text
the residual corrections are not well described by a small linear/additive
basis, so P has to grow until the method loses its budget advantage
```

## Accepted Method 3: Path-Addressed Codebook

**Claim:** The model can emit a sequence of local decisions that forms a path
to a leaf atom. This is the clean version of the "tree" idea for Era 2 budget
accounting.

**Concept:** This is like replacing one big atom menu with a small decision
route. The first prediction chooses a broad branch. The next prediction chooses
a branch inside that branch. After `L` decisions, the path lands on a leaf atom.

The appeal is that a binary path of length `8` can reach up to `256` leaves
while using `16` branch logits per residual layer rather than `256` flat logits.
The danger is also clear: an early wrong branch can send the decoder to the
wrong part of the tree. The tree structure is therefore part of the
representation, not just a cheaper way to write down the same flat dictionary.

Quick Big-O:

```text
general: O(D * sum_l n_l)
fixed branching factor: O(D log W)
```

For each residual layer, the model emits:

```text
level_1 branch logits: n_1
level_2 branch logits: n_2
...
level_L branch logits: n_L
```

The decoder receives a path:

```text
(branch_1, branch_2, ..., branch_L)
```

The number of reachable leaves is:

```text
leaf_capacity = product_l n_l
```

The model prediction-head cost is:

```text
residual_atom_selection_outputs = D * sum_l n_l
head_outputs = B + D * sum_l n_l + S
```

For fixed branching `n`:

```text
L = ceil(log_n(W))
residual_atom_selection_outputs = D * n * ceil(log_n(W))
head_outputs = B + D * n * ceil(log_n(W)) + S
```

For binary path addressing:

```text
residual_atom_selection_outputs = D * 2 * ceil(log_2(W))
head_outputs = B + D * 2 * ceil(log_2(W)) + S
```

This method is clean under the no-runtime-topology contract if the path is
predicted from audio-derived features and the tree is fixed before deployment.
The path must not be selected by topology, waveform class, or oracle metadata.

### Important Accounting Trap

Classical hierarchical softmax often reduces computation over a large class
set, but it does not automatically reduce our **model prediction head budget**.
If the deployed model has a separate output classifier for every internal node,
then the head may contain many node-local logits even though only one path is
used for a given sample.

For Era 2, the cheap version is **path-address prediction**:

```text
one branch-choice head per path level
```

not:

```text
one branch-choice head per internal tree node
```

If an implementation uses node-local classifiers, it must account for the nodes
it actually emits. Do not claim `log(W)` budget just because the codebook is
drawn as a tree.

### What Future Rows Must Log

- `addressing_scheme = path_address`
- `dictionary_scope = per_residual_layer_tree` or `global_tree`
- `D`
- `branch_factors = [n_1, ..., n_L]`
- `leaf_capacity = product_l n_l`
- `reachable_atom_count`
- `unused_leaf_count`, if `leaf_capacity > reachable_atom_count`
- `tree_build_policy`
- `path_loss_policy`
- `head_sharing_policy = per_level` or `per_internal_node`
- `categorical_outputs = D * sum_l n_l` for per-level heads
- `categorical_outputs = D * internal_node_output_count` for node-local heads
- `scalar_outputs = S`
- `head_outputs = B + categorical_outputs + S`
- `topology_used_at_runtime = false`

### Budget Envelope

For at least `W` leaves with fixed branching `n`:

```text
path cost = D * n * ceil(log_n(W))
```

Best plausible case under per-level heads:

```text
D * min_n(n * ceil(log_n(W)))
```

Worst case under honest accounting:

```text
node-local tree heads can approach the size of the tree, not the path length
```

So this method is promising, but only if the implementation really emits path
level decisions rather than a large set of internal-node classifiers.

## Accepted Method 4: Continuous Address Into A Codebook

**Claim:** The model can emit a continuous address vector, and the decoder can
select the nearest atom in an embedding space.

**Concept:** This option stops asking the model to emit a discrete label
directly. Instead, each atom has an embedding vector, and the model predicts a
point in that embedding space. The decoder picks the nearest atom to that point.

On paper, this is very cheap: an `E`-dimensional address costs only `E`
continuous outputs per residual layer, even if the codebook stores many atoms.
The tradeoff is that the whole method depends on the address space being well
behaved. Nearby points need to mean nearby or interchangeable residual
corrections. If the geometry is poor, nearest-neighbor lookup becomes an
unstable substitute for a clean classification target.

Quick Big-O:

```text
residual atom-selection head: O(DE)
if E is fixed: O(D) with respect to stored atom count
```

The model emits, for each residual layer:

```text
address_vector: E continuous outputs
```

The decoder receives the vector and performs:

```text
nearest_atom = argmin distance(address_vector, atom_embedding)
```

The model prediction-head cost is:

```text
residual_atom_selection_outputs = D * E
head_outputs = B + D * E + S
```

This is clean under the no-runtime-topology contract if the atom embeddings are
fixed before deployment and the address vector is predicted from audio-derived
features only.

This method is attractive because the prediction-head budget does not grow
directly with the number of stored atoms. A residual layer could search a large
codebook using a small vector.

But it changes the problem. The model is no longer directly predicting a
categorical reconstruction code. It is predicting a point in an address space,
and the decoder quantizes that point. The quality of the method depends on the
geometry of the embedding space, the distance metric, and how training targets
are generated.

This should be treated as an accepted research direction, but not the first
thing to test if the goal is a high-velocity baseline. It adds training and
diagnostic ambiguity compared with flat categorical or path-address heads.

### What Future Rows Must Log

- `addressing_scheme = continuous_address`
- `dictionary_scope = per_residual_layer_embedding` or `global_embedding`
- `D`
- `address_dim = E`
- `codebook_size`
- `embedding_training_policy`
- `distance_metric`
- `nearest_neighbor_policy`
- `categorical_outputs = 0` for residual atom selection
- `continuous_address_outputs = D * E`
- `scalar_outputs = S`
- `head_outputs = B + continuous_address_outputs + S`
- `topology_used_at_runtime = false`

### Budget Envelope

For residual atom selection:

```text
continuous address cost = D * E
```

Best plausible case:

```text
E << W
```

Worst practical case:

```text
E grows large enough, or nearest-neighbor quality is unstable enough,
that the method loses its budget advantage or becomes hard to train
```

The formula is clean. The empirical risk is higher than flat categorical.

## Rejected Or Deferred Methods

### Within-Layer Compositional Factor Codes

Deferred.

This is the idea that one residual layer's atom is named by several sub-code
choices, and the decoder composes those sub-codes into one correction atom. It
is related to product quantization and other compositional codebook ideas.

The reason it is deferred is conceptual, not because it is impossible. The LFO
representation is already factorized across residual layers: the full
reconstruction code is a tuple of one atom choice per residual layer. That
across-layer factorization is the basic residual stack, not a distinct
within-layer atom-selection strategy.

A within-layer compositional code would be a different design:

```text
one residual layer -> several sub-code choices -> one composed correction atom
```

That may become useful later, but it needs its own construction objective and
decoder semantics. It should not be mixed into this first pass as if it were the
same kind of method as flat, basis-coefficient, path, or continuous addressing.

### Topology-Conditioned Selection

Rejected for Era 2 baseline work.

Any method where topology chooses a sub-dictionary, mask, loss branch, target
schema, or decoder lookup is outside the deployed runtime contract.

It may remain useful as an Era 1 comparison family, but it cannot be used to
claim topology-free deployable output efficiency.

### Oracle Runtime Nearest-Residual Search

Rejected as a deployed model interface.

The oracle can inspect residual curves during offline path construction. The
deployed model cannot. If atom choice depends on the already-known target
residual curve at runtime, then the model has not solved atom selection from
audio.

### Sampled Softmax Or Training-Only Approximation

Rejected for model prediction-head accounting.

Sampled softmax and related approximations can reduce training cost for large
class sets, but the deployed class universe is still large unless the runtime
address itself changes. That is not the same as reducing the LFO prediction-head
nodes.

### Soft Mixture Over Atoms

Deferred.

A model could emit mixture weights over atoms instead of selecting one atom.
But if the mixture spans `W` atoms, the output cost is still usually `W` weights
per residual layer. It also changes the representation from discrete atom
selection into continuous blending. That may be useful later, but it is not the
cleanest next step.

### Autoregressive Residual-Layer Selection

Deferred.

One residual layer's selected atom could condition the next residual layer's
choice. That may improve modeling, but it introduces ordering, teacher-forcing,
and runtime sequencing questions. It does not by itself reduce head budget
unless paired with another addressing scheme.

### Variable Runtime Candidate Sets

Rejected unless the candidate set is determined only by fixed residual-layer
identity or previous model-emitted codes.

If the candidate set is chosen by topology, oracle metadata, waveform labels,
or target-side facts, it violates the Era 2 contract. If it is chosen by the
residual layer index, it is just a per-residual-layer dictionary. If it is
chosen by earlier predicted branch decisions, it is a path address and should
be accounted that way.

## Stress Tests

These are symbolic checks, not experiment rows.

### Fixed `D`, Varying `W`

Flat categorical:

```text
head_outputs = B + D * W + S
```

Path-address, fixed `n`:

```text
head_outputs = B + D * n * ceil(log_n(W)) + S
```

Basis coefficients:

```text
head_outputs = B + D * P + S
```

Continuous address:

```text
head_outputs = B + D * E + S
```

Interpretation:

Flat categorical is the clearest baseline but scales linearly with `W`.
Path-address methods can expose larger discrete address spaces with roughly
logarithmic output growth. Basis coefficients replace discrete atom selection
with a continuous basis family. Continuous address can decouple output count
from codebook size, but only if the learned address geometry works.

### Fixed Model Prediction Head Budget

Let:

```text
H = available head_outputs
H_atom_budget = H - B - S
```

`H_atom_budget` is the residual atom-selection budget.

Flat categorical with constant `W`:

```text
D * W <= H_atom_budget
W <= floor(H_atom_budget / D)
```

Path-address with fixed `n`:

```text
D * n * L <= H_atom_budget
L <= floor(H_atom_budget / (D * n))
leaf_capacity = n^L
```

Basis coefficients:

```text
D * P <= H_atom_budget
P <= floor(H_atom_budget / D)
continuous_basis_dimension = P
```

Continuous address:

```text
D * E <= H_atom_budget
E <= floor(H_atom_budget / D)
```

Interpretation:

At equal model prediction-head budget, flat categorical buys a small number of
direct choices. Path addressing can buy many more nominal discrete addresses.
Basis coefficients buy a continuous family of residual corrections. The open
question is whether that family captures the useful residual variation.

### Fixed Effective Address Count

Suppose each residual layer should expose at least `W_eff` possible residual
corrections.

Flat:

```text
cost = D * W_eff
```

Balanced path:

```text
cost = D * n * ceil(log_n(W_eff))
```

Basis coefficients:

```text
not a discrete W_eff-leaf method; compare by P at matched head budget
```

Continuous:

```text
cost = D * E
```

Interpretation:

Flat addressing is the honest upper-cost reference. Path addressing is the main
simple discrete way to get logarithmic output growth. Basis coefficients are the
main simple non-discrete alternative: they trade atom choice for continuous
reconstruction coefficients. Continuous addressing has the smallest formula
when `E` is small, but it is the least directly comparable because selection
quality depends on geometry rather than class probability.

## Required Tracking Fields For Future Era 2 Manifests

Every row should report these common fields:

```text
addressing_scheme
dictionary_scope
base_dictionary_size
D
scalar_families
scalar_outputs
residual_atom_selection_outputs
categorical_outputs
continuous_address_outputs
head_outputs_formula
head_outputs_actual
topology_used_in_construction
topology_used_at_runtime
topology_used_in_targets
topology_used_in_loss
topology_used_in_decoder_lookup
topology_used_in_head_accounting
```

For flat categorical rows:

```text
W_by_residual_layer
global_codebook_size
```

For basis-coefficient rows:

```text
basis_count
coefficient_constraint
basis_construction_policy
coefficient_target_policy
continuous_basis_outputs
```

For path-address rows:

```text
branch_factors
path_length
leaf_capacity
reachable_atom_count
unused_leaf_count
tree_build_policy
path_loss_policy
head_sharing_policy
```

For continuous-address rows:

```text
address_dim
codebook_size
embedding_training_policy
distance_metric
nearest_neighbor_policy
```

The topology fields should be boring for clean Era 2 runtime rows:

```text
topology_used_at_runtime = false
topology_used_in_targets = false
topology_used_in_loss = false
topology_used_in_decoder_lookup = false
topology_used_in_head_accounting = false
```

`topology_used_in_construction` may be true or false, as long as it disappears
before model-facing targets exist.

## What Should Be Tested First Later

This note does not choose Experiment 10 rows. It does narrow the path.

The first high-velocity Era 2 baseline should be:

```text
flat categorical atom addressing
per-residual-layer dictionaries
phase-only scalars
no runtime topology
```

Reason:

It is the easiest row to audit. It gives us the missing no-runtime-topology
baseline. It also creates the clean reference formula:

```text
head_outputs = 32 + D * W + (D + 1)
```

After that baseline exists, the next comparison should not be "more atoms" in
the abstract. It should be:

> At the same model prediction head budget, does a structured address expose
> more useful residual corrections than flat categorical selection?

The first structured follow-up should likely be basis-coefficient reconstruction
or path addressing, not continuous nearest-neighbor addressing. Basis
coefficients are the cleanest distinct non-tree alternative. Path addressing is
the cleanest structured discrete alternative. Continuous addressing is real, but
it adds embedding-geometry and nearest-neighbor training questions before we
have the clean baseline.

## Bottom Line

The clean Era 2 contract does not force one atom-selection method. It forces us
to account honestly for whatever method we choose.

Flat categorical selection is the baseline because it is direct:

```text
one residual layer, W choices, W logits
```

Global dictionaries change storage and atom reuse, not necessarily output-head
cost.

Basis coefficients change the game: they do not expose more discrete atom
leaves, they expose a continuous reconstruction family:

```text
D * P
```

Path-addressed codebooks are the main simple discrete way to make the model
prediction-head budget grow sublinearly with the number of reachable atoms:

```text
D * n * ceil(log_n(W))
```

Continuous address vectors can be even cheaper on paper:

```text
D * E
```

but they are a larger modeling shift.

For future Era 2 comparisons, the unit of fairness is:

```text
same or near-same model prediction head budget
```

not the same number of stored atoms, not the same oracle search effort, and not
the same historical W/D label.
