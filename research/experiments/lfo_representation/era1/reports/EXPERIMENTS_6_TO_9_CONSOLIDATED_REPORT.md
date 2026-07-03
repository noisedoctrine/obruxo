# LFO Experiments 6-9 Consolidated Report

Experiments 6 through 9 are the first coherent arc of the LFO representation work. They start with representation-family selection, move into additive residual codebook construction, then stress-test width, depth, scalar modifiers, decoder policies, and budget-matched narrow/deep stacks.

The short version: residual codebooks are the right direction, depth is the most reliable lever so far, phase belongs in the baseline, gain/offset are not yet settled, clipping is a useful zero-output decoder policy, and topology-aware construction helped in Era 1. The important correction is that topology-aware construction was later mixed with topology-conditioned code selection. That means the strongest efficiency claims remain valid inside the topology-conditioned family, but they are not yet proof of a clean audio-only deployed representation.

## What These Experiments Were Really Testing

The useful lens is not "which row won?" It is which parts of the representation became credible enough to carry forward.

Experiment 6 compared broad representation families: direct sampled grids, shared residuals, topology residuals, partition/switch variants, and additive residual stacks. Experiment 7A fixed on the additive shared+topology family and asked how to build the atoms. Experiment 8 scaled residual width and depth under a cheap screen. Experiment 9 tested affine scalars, clipping/snap policies, and budget-matched narrow/deep stacks.

Across the arc, three costs must stay separate:

- **Serialized fields**: compact path fields written to artifacts, such as selected indices and phases.
- **Deployed head outputs**: categorical logits plus scalar outputs a downstream model would need to emit.
- **Codebook storage**: decoder-side tables of atoms, which matter for shipping but are not model output heads.

The later reports use deployed `head_outputs` as the main model-facing cost. That is the right comparison axis, but only if the interface assumptions are stated clearly.

## Representation Family: Residual Codebooks Beat Direct Grids On Tail RMSE

Experiment 6 established the first major prior: structured additive residual codebooks are the best sampled-curve reconstruction family in the useful output range.

The best additive rows at eval-1920 were:

| candidate | dense outputs | median RMSE | P95 RMSE |
| --- | ---: | ---: | ---: |
| additive K16, 8 actual residual layers | 180 | 0.0073663 | 0.0331399 |
| additive K12, 8 actual residual layers | 148 | 0.0076170 | 0.0339565 |
| additive K8, 8 actual residual layers | 116 | 0.0076528 | 0.0384051 |
| Grid192 | 192 | - | 0.0544862 |

That does not make direct grids irrelevant. Grid96/Grid192 remained stubbornly useful for custom-ish curves and editor-node preservation. The sampled curve can sound or look close while the original editor nodes are not preserved. That matters because the final Vital-facing representation may need either a direct residual, a refit pass, or a fallback path for highly custom shapes.

**Takeaway:** residual codebooks are the main compact reconstruction path. Direct grids are not the primary representation, but they remain a useful warning and possible fallback for custom/editor-faithful cases.

## Depth Emerged Early As The Better Lever

Experiment 6 already showed depth was not saturated. In the additive family, adding residual layers kept improving P95. Width helped too, but the marginal value of width flattened faster.

At 8 actual residual layers, the K12 -> K16 width step improved P95 by only about `0.0008166` for 32 extra outputs. Earlier depth moves bought much larger P95 reductions per added output.

Experiment 8 made this sharper. The best rows all used high `D`, and narrow/deep rows were often better than wider/shallow rows at similar or lower head-output budgets.

Top Experiment 8 phase-only size rows:

| W | D | head outputs | median RMSE | P95 RMSE |
| ---: | ---: | ---: | ---: | ---: |
| 16 | 32 | 1089 | 0.000650555 | 0.010222 |
| 8 | 32 | 577 | 0.000920192 | 0.0120029 |
| 16 | 28 | 957 | 0.000809140 | 0.0123547 |
| 8 | 28 | 509 | 0.001153030 | 0.014326 |
| 8 | 24 | 441 | 0.001469390 | 0.0172763 |

Experiment 9 then pushed the narrow/deep idea under equivalent budgets. W4/W6 rows at larger depths improved tail error against W8 anchors, especially at the W8D32-equivalent and larger budgets.

Representative 9D rows:

| row | head outputs | anchor | P95 RMSE | perfect LFO rate |
| --- | ---: | ---: | ---: | ---: |
| W4D58 | 555 | W8D32 / 577 | 0.010271 | 88.972% |
| W6D42 | 579 | W8D32 / 577 | 0.00930753 | 90.5919% |
| W4D86 | 807 | W8D48 / 849 | 0.0059836 | 95.2648% |
| W4D116 | 1077 | W8D64 / 1121 | 0.00343919 | 98.8785% |

**Takeaway:** Era 2 should treat residual depth as the main quality lever. Width remains useful, but not as the first thing to expand.

## Phase Became Baseline, Not An Optional Modifier

The residual codebook representation depends on circular alignment. By Experiment 8, phase was always enabled and counted as part of the baseline:

```text
base phase + one phase per residual layer = D + 1 scalar outputs
```

This was the right move. Treating phase as optional would confuse the representation with a weaker ablation. Phase is part of how the atoms are applied, not an embellishment.

For phase-only deployed accounting in the topology-conditioned family, the report-side formula was:

```text
head_outputs = 32 + sum(layer_codebook_size) + (D + 1)
```

For the clean no-runtime-topology baseline that Era 2 still needs, the corresponding formula becomes:

```text
head_outputs = 32 + D * W + (D + 1)
```

**Takeaway:** phase is baseline. Future reports should not describe phase as an optional modifier axis unless the experiment is explicitly testing a phase-free ablation.

## Gain And Offset Did Not Earn Baseline Status

Experiment 7A found that modifier policies mostly added outputs without changing the decision. For the winning construction policy, `none`, `base_gain`, and `global_offset` were effectively tied at P95, with `none` cleaner because it spent fewer outputs.

Experiment 8 repeated the pattern at W12D16. Offset degraded P95, gain was effectively tied with phase-only under final-only clipping, and gain+offset did not justify its cost.

Experiment 9 explored base/residual affine scopes and normalization. The best 9A affine row was base+residual gain/raw by P95, but the result is not enough to promote gain into the core baseline. There is also a separate accounting bug or ambiguity: scheduled `residuals_only` rows do not appear to activate residual affine counting because the implementation checks `residual_only`.

**Takeaway:** keep gain/offset out of the Era 2 baseline. Revisit gain after the topology-free baseline is clean, and audit the Experiment 9 `residuals_only` / `residual_only` mismatch before trusting residual-only affine conclusions.

## Decoder Policies Are Cheap But Not Harmless

Experiment 8 and 9 both support the category of zero-output decoder policies. Clipping and snap do not add model-facing outputs. They change deterministic reconstruction rules after or during atom composition.

Experiment 8 showed intermediate `[-1, 1]` clipping improved W12D16 phase+gain P95 from `0.0268194` to `0.0234246` at the same head-output count.

Experiment 9 showed clipping remained a useful axis, but not a single-metric slam dunk. Per-layer `[0, 1]` won 9B P95; bipolar clipping had the best median in that section.

Snap was more dangerous. Rails improved P95 relative to no snap, but badly damaged strict perfect-LFO rate. That makes snap look like an aggressive tail repair that hurts many already-close curves.

The audit adds one important nuance: zero output cost does not mean target-neutral. Clipping participates in candidate scoring, and snap is applied during candidate finalization in Experiment 9. These policies can change which oracle paths are selected.

**Takeaway:** clipping remains a serious decoder-policy candidate. Snap should not be a blanket default. Future reports should call these zero-output policies, while also noting that they can affect oracle targets.

## Topology Helped, But The Interface Became Ambiguous

Topology-aware construction was one of the strongest Era 1 signals. Experiment 7A's winner was `topology_balanced_common_then_tail / none`, with median `0.0026222`, P95 `0.034651`, and P99 `0.054708`. It improved every topology bucket in the 7A breakdown.

But the audit showed a boundary problem. Topology was not only used to build atoms. In 7A/8/9 topology-conditioned chains, topology also selects which dictionary row is active during path construction and decoding.

That creates two different concepts:

- **Allowed going forward:** topology helps build a better codebook offline.
- **Not allowed in the deployable baseline:** topology participates in the model-facing code-selection interface.

The `3W` deployed accounting in Experiments 8/9 is arithmetically consistent. It flattens three topology-specific dictionaries into one wide categorical head. The problem is that the implementation path uses:

```text
external topology condition + local W-way atom index
```

while the report-side deployment accounting charges it as:

```text
one flattened 3W-way residual-layer head
```

Those are related but not equivalent. Neither is the clean audio-only baseline.

**Takeaway:** topology-aware construction remains a plausible offline prior. Topology-conditioned code selection should be removed from the Era 2 baseline.

## What Remains Valid From Era 1

These findings survive the audit, with scope:

- Additive residual codebooks are the best compact reconstruction family tested so far.
- Depth is the strongest quality lever in the tested family.
- W16D32 is the best-quality Experiment 8 row inside the topology-conditioned family.
- W8D32 is a useful compact deep reference inside that same family.
- Experiment 9's narrow/deep W4/W6 rows are strong evidence that deeper stacks deserve priority.
- Phase belongs in the baseline.
- Gain/offset are not baseline features yet.
- Clipping is a real zero-output decoder-policy lever.
- Direct grids remain important as a custom/editor-node warning or fallback candidate.

These findings need qualification:

- Current output-efficiency claims are scoped to the topology-conditioned family.
- Current 7A/8/9 rows do not prove a topology-free deployed representation.
- The `3W` accounting is correct as a convention, but it is not evidence that topology is absent from the runtime code-selection path.
- Experiment 9 residual-only affine rows need a separate accounting audit.

## The Main Gap After Experiment 9

The missing comparison is not another topology-conditioned variant. It is a clean no-runtime-topology residual codebook baseline.

That baseline should be:

- phase-only;
- no gain/offset;
- constant `W`;
- actual `D` residual layers;
- no topology-conditioned residual-layer interface;
- no topology condition in encoding or target generation;
- no topology condition in decoding;
- no topology-dependent masks or losses;
- no topology field in the model-facing artifact schema;
- no topology adjustment in output-head accounting.

For phase-only, the deployed head-output formula should be:

```text
head_outputs = 32 + D * W + (D + 1)
```

Useful matched-budget anchors from Era 1:

- W8D16 topology-conditioned reference: `305` head outputs;
- W8D32 topology-conditioned reference: `577` head outputs;
- W16D32 topology-conditioned reference: `1089` head outputs;
- Experiment 9 W4/W6 narrow/deep budget rows.

These anchors should be used as budgets, not as proof that the old interface is acceptable.

## Consolidated Recommendation

Era 2 should start by rebuilding the residual codebook experiment around an audio-only deployed-model contract. Keep the pieces that worked: additive residual stacking, phase alignment, narrow/deep scaling, output-head accounting, and careful decoder-policy comparisons. Remove the topology-conditioned code-selection interface from the baseline. Let topology compete only as an offline construction strategy.

Once the clean baseline exists, the project can make fair claims about output efficiency. Until then, the best Era 1 rows are strong topology-conditioned references, not the final evidence for a deployable LFO code representation.
