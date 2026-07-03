# LFO Experiments 1-9 Conceptual Audit Notes

## The Central Issue

The `3W` accounting in the Experiment 8/9 reports is arithmetically consistent. The problem is what it stands for.

The intended rule is simple:

> Topology may guide oracle codebook generation, but it should not be part of the deployed model interface.

That means topology can affect how atoms and dictionaries are discovered offline. It must not affect what the deployed model receives, predicts, masks, losses against, serializes, or decodes with.

The current 7A/8/9 setup mixes two ideas that should be kept separate:

1. **Offline topology-aware dictionary construction**: use known topology labels while building atom dictionaries.
2. **Topology-conditioned code selection**: use topology later to choose which atoms are available for a path.

The first is the intended use. The second changes the representation being evaluated. If the deployed model receives audio only, topology should not influence which reconstruction codes it is expected to emit.

The deployed model should not receive topology as input. It should not predict topology explicitly. It should not use topology-conditioned masks, losses, artifact schemas, or decoder lookups. Topology can be used upstream while discovering atoms, but the final residual-layer code interface should be topology-free.

Right now, topology is not just an offline construction detail. It enters the residual-layer lookup path.

That makes the headline conclusion:

> The project is missing a clean no-runtime-topology baseline. We need a baseline where topology is used, if at all, only during oracle atom/dictionary construction. The model-facing representation should require only atom indices and scalars, with no topology condition in encoding, decoding, target generation, loss schema, artifact schema, or output-head accounting.

Until that baseline exists, the W16D32, W8D32, and narrow/deep efficiency claims are valid only inside the current topology-conditioned experiment family. They do not yet prove broader output-head efficiency for an audio-only deployed model.

## Terminology I Use Below

**Residual layer** means one model-facing residual unit: one atom choice plus the scalar values needed to apply that atom. In the code, this is usually called a `stage`. I use `stage` only when referring to literal code fields such as `PhaseChain.stages`, `stage_N_index`, or `stage_label`.

**Residual-layer pair** means the internal 7A/8/9 training-loop unit. The loop variable `d` creates one shared residual layer and one topology-conditioned residual layer. Public/report `D` is therefore `2 * d`.

**Serialized fields** are compact fields written to path artifacts, such as one local `stage_N_index`, phase, and gain per residual layer. They are not the same thing as deployed output heads.

**Deployed head outputs** are the report-side estimate of what a downstream model would need to emit: categorical logits plus scalar outputs.

**Per-layer dictionaries** means each residual layer has its own atom dictionary. **Global dictionaries** would mean one dictionary shared across residual layers. Experiments 7A/8/9 use per-layer dictionaries.

## Two Contexts That Need To Stay Separate

There are two places where topology can appear, and they have very different meanings.

### 1. Offline oracle/codebook construction

This is where atoms and dictionaries are discovered. It may use metadata that would not be available to an audio-only deployed model, including topology labels. That is acceptable only if the metadata shapes the learned atoms/dictionaries and then disappears from the final code interface.

The current code goes further. When `chain.topology_conditioned` is true, the path construction/decoder code uses known topology metadata to choose the residual-layer dictionary row. That is no longer just topology-aware codebook generation; it is topology-conditioned code selection.

That can be useful for analysis, but it must be named clearly. It is not the same interface as a deployed predictor that sees audio and emits reconstruction codes.

### 2. Deployed model inference

This is the model-facing contract: given audio features, what must the model predict?

For a clean audio-only representation, the deployed model should predict:

```text
base code + residual-layer atom indices + phase/gain/offset scalars
```

It should not receive a topology condition. It should not predict a topology label. It should not rely on topology-conditioned masks, topology-conditioned loss schemas, topology-conditioned artifact schemas, or topology-conditioned decoding lookup.

This is where the current experiments are ambiguous. The reports charge topology-conditioned residual layers as flattened `3W` heads, but the actual oracle/decoder implementation uses:

```text
external topology condition + local W-way atom index
```

Those are related, but not equivalent. The first requires topology side information during code selection. The deployed-model rule above says that side information should not be present.

## What The Code Actually Does

In `PhaseChain`, each residual layer is stored as `[condition, code, phase]`.

When `chain.topology_conditioned` is true, `_conditions_for` returns `dataset.topology`. Encoding and decoding then use that condition to select the topology row before selecting the local atom index:

```text
stage[conditions, encoding.stage_indices[layer_index]]
```

So the path is not literally a flat `3W` atom selection. It is topology row selection plus a local `W` atom index.

The path artifacts reflect this. They store local residual-layer indices such as `stage_1_index`, `stage_2_index`, etc., along with phase/gain values. They do not store a separate topology prediction target.

Implication:

The current representation depends on topology during path construction and decoding. That violates the intended deployed-model rule: topology should help build the codebook, not select codes after the codebook exists.

## What The Reports Actually Count

The Experiment 8/9 report generators recompute deployed `head_outputs` by charging:

- `W` categorical logits for shared residual layers;
- `3W` categorical logits for topology-conditioned residual layers;
- scalar outputs for base/residual phases and optional gain/offset families.

That is internally consistent. It treats a topology-conditioned residual layer as if its three topology rows were flattened into one categorical head.

Example: Experiment 8 W8D16 phase-only final-only.

- There are 8 shared residual layers and 8 topology-conditioned residual layers.
- Report-side residual logits are `8 * (8 shared + 3 * 8 topology) = 256`.
- Add 32 base logits: `32 + 256 = 288`.
- Add scalar phases: `D + 1 = 17`.
- Report-side `head_outputs = 305`.

The same artifact also has different legacy/storage counts:

- runner `categorical_logits = 163`;
- runner `dense_outputs = 196`;
- serialized fields `predicted_outputs = 50`.

Those are not contradictions. They are different accounting layers. The contradiction appears only when prose treats the report-side `3W` deployment accounting as if it proves the actual oracle/decoder path is topology-free. It does not.

Implication:

The `3W` numbers can still be used for within-family comparisons, but the reports should say exactly what they assume: topology-conditioned residual layers are being charged as flattened categorical heads for deployment accounting, while the oracle path uses topology side information.

## What Is Offline Versus Online

Topology use during **offline construction** is not the issue by itself. This is the place where topology belongs.

For shared residual layers, the `topology_balanced_common_then_tail` policy uses topology labels to balance the candidate pool before fitting atoms. That is an offline construction choice.

For topology-conditioned residual layers, the code goes further. It loops over topology conditions, restricts members to each topology, and fits separate atom dictionaries for smooth, continuous, and discontinuous topology rows.

This means topology is doing two jobs:

- helping construct atoms offline;
- selecting which atom dictionary row is used during path construction/decoding.

The second job is the conceptual problem for audio-only deployment.

Implication:

In the clean version, topology stays in offline atom/dictionary construction only. The deployed path still chooses from a topology-free residual-layer dictionary using only predicted atom indices and scalars.

## Dictionary Scope

The 7A/8/9 dictionaries are per residual layer.

Shared residual layers repeat the same `[W]` atom dictionary across topology rows. Topology-conditioned residual layers store separate `[W]` dictionaries for each topology row. Saved manifests alternate labels such as:

```text
layer_1_shared
layer_1_topology
layer_2_shared
layer_2_topology
...
```

I found no evidence in 7A/8/9 that a single global dictionary is learned once and reused across residual layers.

Implication:

Do not describe 7A/8/9 as testing global-shared-across-layers dictionaries. If global sharing is a desired hypothesis, it needs a separate construction objective and run.

## Decoder-Free Policies

Clipping and snap policies add zero deployed output-head dimensions. Experiment 9 sections 9B and 9C keep `head_outputs=305`, `scalar_outputs=17`, and `predicted_outputs=34` across clipping/snap rows.

But zero output-head cost does not mean these policies are target-neutral.

Clipping participates in candidate scoring through `_apply_decoder_step_np`. Snap is applied during Experiment 9 candidate finalization through `_finalize_experiment9`. So these policies can change which atom indices, phases, or gains are selected, even though they do not add model outputs.

Implication:

Keep clipping/snap in the "decoder/free policy" bucket for output-head accounting. Also document that they may change oracle targets.

## Gain, Offset, And Phase Accounting

Phase accounting is separate from the topology issue.

For Experiment 8/9 phase-only rows, scalar outputs are `D + 1`: one base phase plus one phase per residual layer. Gain and offset families add more scalar outputs when enabled.

There is a separate Experiment 9 issue:

- schedules use `target_scope="residuals_only"`;
- `_affine_applies(policy, target)` checks for `f"{target}_only"`;
- residual calls pass `target="residual"`, so the matching string would be `residual_only`, not `residuals_only`.

Observed consequence:

- 9A residual-only rows have `scalar_outputs=17` and `head_outputs=305`;
- those counts match phase-only rows despite labels such as `residuals_only_phase_gain_offset_raw`.

Implication:

Do not rely on Experiment 9 residual-only affine conclusions until the `residuals_only` / `residual_only` path is audited and corrected. This is independent of the topology-conditioned selection issue.

## Which Existing Conclusions Still Hold

These conclusions still hold inside the tested topology-conditioned family:

- W16D32 is the best-quality row in Experiment 8's topology-conditioned, phase-only, final-only size screen.
- W8D32 is cheaper than W16D32 under the report-side `3W` accounting and remains a useful narrow/deep reference in that family.
- Experiment 9's 9D narrow/deep rows show strong quality at matched or near-matched corrected budgets, again inside the topology-conditioned family.
- Clipping/snap comparisons are valid as zero-output-cost decoder-policy comparisons, with the caveat that selected oracle targets may change.

These conclusions need qualification:

- topology is not only an offline construction detail in current 7A/8/9 code;
- current 7A/8/9 rows do not prove a topology-free deployed representation;
- current output-efficiency claims should not be generalized beyond the topology-conditioned experiment family;
- Experiment 9 residual-only affine conclusions are currently ambiguous.

## The Missing Baseline

The needed baseline is:

- phase-only;
- no gain/offset;
- constant `W`;
- `D` actual residual layers;
- per-residual-layer dictionaries are fine;
- no topology-conditioned residual layers in the model-facing code interface;
- no topology condition in encoding/path target generation;
- no topology condition in decoding/reconstruction lookup;
- no topology-dependent masks or losses;
- no topology field in the model-facing artifact schema;
- no topology adjustment in output-head accounting.

For that baseline, phase-only deployed head accounting should be:

```text
head_outputs = 32 + D * W + (D + 1)
```

This gives the project a clean answer to the key question:

> How much quality do we get from residual layering when topology is used only to build the codebook and the deployed model predicts only atom indices and scalars from audio?

Only after that exists should we compare topology-conditioned rows as an optional enhancement.

Useful matched-budget anchors from the current reports:

- Experiment 8 W8D16: `305` head outputs;
- Experiment 8 W8D32: `577` head outputs;
- Experiment 8 W16D32: `1089` head outputs;
- Experiment 9 9D W4/W6 budget rows.

Compare by matched `head_outputs`, not just by W/D label.

## Recommended Edits And Next Runs

1. Rewrite language that says or implies topology is only offline when the code uses it online.

   Suggested framing: "Topology labels may be used while building the codebook. In current topology-conditioned runs, they are also used as side information for residual-layer lookup, so those runs are not topology-free deployed-model baselines."

2. Qualify efficiency claims.

   W16D32, W8D32, and 9D narrow/deep results are valid within the topology-conditioned family. Broader output-efficiency claims need the no-runtime-topology baseline.

3. Run the matched-budget no-topology baseline before making architecture-level efficiency claims.

   The baseline should remove topology from encoding, decoding, target generation, losses, artifact schema, and head accounting. Offline topology-aware atom balancing can be tested separately, but only if the final code interface and code selection remain topology-free.

4. Separately audit Experiment 9 affine accounting.

   Specifically check the `residuals_only` / `residual_only` naming mismatch before interpreting residual-only gain/offset rows.

5. Treat binary-tree indexing as future work.

   I found no current 7A/8/9 binary-tree or structured-index implementation. It should not be described as part of the present fix.

## Short Version

The reports' `3W` numbers are not numerically wrong. The conceptual problem is that they sit on top of an implementation where topology is used online to choose residual-layer dictionary rows.

For an audio-only deployed model, topology should be used only to build the codebook. It should not be input, output, mask, loss condition, artifact-schema field, or decoder lookup condition. The project therefore needs a clean no-runtime-topology baseline before it can make broad claims about residual-layer output efficiency.
