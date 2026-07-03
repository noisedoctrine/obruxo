# Audit: Fix Topology-Conditioned Codebook Cost Accounting

## Context

Experiments 7A, 8, and 9 all report `head_outputs` using a formula that charges topology-conditioned residual layers `3W` categorical logits instead of `W`:

```text
head_outputs = 32 + sum(layer_codebook_size) + (D + 1) * (I_phase + I_gain + I_offset)
```

where `layer_codebook_size = W` for shared layers and `layer_codebook_size = 3W` for topology-conditioned layers (source: Experiment 8 and Experiment 9 "Output-Head Accounting" sections).

This `3W` comes from flattening three topology-specific dictionaries (smooth / continuous / discontinuous) into one wide softmax, because "there is no separate topology classifier in this accounting; the model emits one categorical code per residual layer" (Experiment 8/9). Every tested configuration in 7A/8/9 uses a fixed architecture of alternating shared and topology-conditioned layers, so this `3W` tax is baked into every reported number, including the Pareto frontiers and the "W16D32 is the quality leader" conclusion.

**Arithmetic check: this is not a bug.** I re-verified the `3W` formula against every row in Exp 8's size screen and Exp 9's 9D table - all reconcile exactly. The problem is conceptual: an implementation choice (flattening 3 topology tables into 1 wide softmax) is being reported as if it were an inherent cost of the representation, and it isn't the only way to build this. The sections below are about drawing the actual design boundaries correctly so the next redesign doesn't recreate the same conflation.

## The three things that must be kept strictly separate

The current docs blur three genuinely different concerns into one number and one architecture. Going forward, every design decision must be filed under exactly one of these, and any place where one influences another must be called out explicitly rather than left implicit.

### A. Oracle (offline): atom discovery and dictionary construction

This is the process that decides which observed training residuals become the frozen atoms in a codebook, before the model ever trains - eg. `greedy_global_improvement`, `topology_balanced_common_then_tail`. It has no runtime cost and does not appear in `head_outputs`. Topology labels are fully allowed here (eg. "balance the candidate pool across smooth/continuous/discontinuous before selection") because this is pure data curation, not a runtime decision.

### B. Model (online): what the model must predict to select and compose atoms

This is the only part that has a real "cost" in the `head_outputs` sense. It has been under-specified in the prior reports, and it has (at least) two separable sub-decisions of its own:

**B1. Dictionary scope** - does every layer draw from the same global dictionary, or does each layer have its own dictionary?

* Global-shared-across-layers: one `[W]` dictionary reused at every layer.
* Per-layer dictionaries: each of the `D` layers has its own `[W]` dictionary.
* Both are legitimate; this is not the thing being rejected.

**B2. Index representation** - *how* does the model's prediction encode "which of the W atoms did you pick"? This changes the parameter/output accounting independently of W and D, and was not previously treated as a design axis at all - it was silently assumed to be a flat categorical. At least two representations should be considered:

* **Flat categorical**: one `W`-way softmax per layer. Cost per layer = `W` logits. This is what 7A/8/9 all used.
* **Binary tree path**: the codebook is organized as a binary tree of depth `log2(W)`; the model predicts a sequence of binary decisions (the path from root to the selected atom). Cost per layer = `log2(W)` binary outputs instead of `W` - much cheaper on paper, but the decisions may be sequential/conditional on each other rather than a single independent prediction, which changes the learning problem, not just the output count. This needs its own accounting line, separate from the flat-categorical formula, if it's pursued.
* Other tree branching factors (not just binary) are possible too, and would sit between these two on both axes (cost, structure).

**What is explicitly rejected in B, regardless of representation:** a layer predicting a *sub-dictionary selection* (eg. "which of 3 topology-specific tables") as a hidden extra decision bolted onto the atom index. Whether that sub-selection is implemented as a flattened wide softmax (`3W`, current approach) or as a cheaper hierarchical bucket-then-atom head (`W + 3`), both were considered and both are out - the model should never have more than one dictionary to choose from at any layer, full stop. This is orthogonal to B1/B2: it doesn't matter whether atoms are chosen via flat categorical or binary tree, or whether the dictionary is global or per-layer - no layer gets a topology (or any other) sub-selection layered inside its choice.

### C. Decoder policy (free): operations with zero parameter cost

These are deterministic post-processing rules applied after the model's predicted indices/scalars are used to gather and compose atoms. They do not change what the model predicts and must never appear in `head_outputs`. Precedent already exists for this category - Experiment 9 explicitly frames clipping and snap variants as "zero-output decoder choices" ("9B and 9C policies are zero-output decoder choices" and clipping "has zero output-head cost, so its quality-per-output ratio is undefined rather than merely large" - Exp 8).

* **Intermediate (between-layer) clipping**: clip the running prefix after each layer.
* **Global (final-only) clipping**: clip once at the very end.
* **Both together**: intermediate clipping during composition plus a final clip.
* All variants of this are free - they affect reconstruction quality but must be verified to never add a single output to `head_outputs`, regardless of which combination is used.

## Where A and B are allowed to couple - and where they must not

This part needs the most care during the audit, because "the oracle and the model are separate" is true but not the whole story - the oracle's construction strategy should sometimes be *conditioned on* the model-side design choice in B1, even though the oracle itself never adds runtime cost.

* **If B1 = global-shared-across-layers**: the oracle must construct a *single* dictionary that works well when reused at every layer - i.e. it should optimize atom selection jointly across the residual distributions seen at all `D` layers, not just at one. This is a real coupling: the construction objective changes based on the model-side sharing decision, even though the atoms themselves are still 100% offline.
* **If B1 = per-layer dictionaries**: the oracle can fit each layer's dictionary independently against that layer's own residual distribution (closer to what 7A/8/9 already do).
* **Topology-aware construction (category A) is decoupled from topology-conditioned prediction (category B).** The oracle can use topology labels to balance or diversify a dictionary's contents without the model ever making a topology-related decision at runtime. This decoupling is already correctly modeled in the existing `topology_balanced_common_then_tail` construction policy - the mistake in the prior work was pairing that construction policy with topology-conditioned *layers* (a B-level mechanism) rather than testing it against a purely global- or per-layer-shared B1 design.
* **Index representation (B2) is a pure model/decoder-interface choice and should not require any special oracle awareness** beyond knowing how many atoms `W` it needs to produce - a binary-tree-organized codebook still just needs `W` well-chosen atoms from the oracle; only how the model addresses them changes.

## What to do next

This document is primarily diagnostic. Start by auditing the code, configs, generated artifacts, and written reports to determine what actually happens at each layer of the system. Treat the concerns above as hypotheses to verify, falsify, or refine - not as already-proven implementation facts.

The original remediation proposal is preserved below because it captures the suspected design direction if the concerns are confirmed. However, it should **not** be executed yet. Treat it as an additional object of investigation: evaluate whether the code/artifacts support its assumptions, whether its proposed baseline is actually the missing comparison, and whether any part of it rests on a misunderstanding of the current implementation.

### Diagnostic audit to perform first

1. **Verify the actual output-head accounting path.**
   Trace where `head_outputs` is computed in code and where the reported values in Experiments 7A, 8, and 9 come from. Confirm whether topology-conditioned residual layers are actually charged as `3W` outputs in the implementation/artifacts, or whether `3W` is only a reporting convention. Reconcile the code, configs, tables, and written reports.

2. **Verify what a topology-conditioned layer actually predicts.**
   Determine whether the model really emits one flat categorical over `3W` possible atoms, emits a topology/bucket decision plus an atom decision, uses a masked subset, or does something else. The key question is: at runtime, does the model choose from multiple topology-specific dictionaries inside a layer, or from exactly one dictionary?

3. **Verify dictionary scope in the implemented experiments.**
   For Experiments 7A, 8, and 9, identify whether dictionaries are global-shared-across-layers, per-layer, or some hybrid. Do not infer this only from prose; verify it from code/config/artifacts. If dictionaries are per-layer, confirm whether each layer has its own atom set. If global, confirm whether the same atom set is reused across layers.

4. **Verify how the oracle constructs atoms.**
   Inspect the offline atom discovery / codebook construction path. Determine which policies are actually used for the reported experiments, whether topology labels are used only during offline construction, and whether the construction objective is per-layer or joint across layers. Specifically check whether topology-aware construction is separable from topology-conditioned runtime prediction in the current implementation.

5. **Verify whether any topology decision leaks into the online model path.**
   Check model inputs, outputs, masking logic, target generation, loss functions, and decoding. The question is not only whether there is a named “topology classifier”; it is whether topology information affects the model’s runtime atom-selection interface in any form.

6. **Verify clipping and decoder-policy cost accounting.**
   Confirm from code and artifacts that intermediate clipping, final clipping, snap variants, or any other decoder policies add zero output-head dimensions. Note any case where a supposedly “free” decoder policy changes model outputs, loss targets, or accounting.

7. **Verify whether gain/offset/phase accounting is independent of the topology issue.**
   Do not redesign these, but confirm the current formula and implementation: which scalar heads are present, how many outputs they add, and whether they interact with dictionary selection or topology-conditioned layers.

8. **Verify whether existing conclusions depend on the suspected conflation.**
   Re-read the Pareto frontier claims, the “W16D32 quality leader” conclusion, and any “output-head efficient” language. Identify which claims are purely arithmetic under the existing formula, which claims assume `3W` is an inherent representational cost, and which claims would require a controlled no-sub-selection comparison before being trusted.

### Prior remediation hypothesis to evaluate, not execute yet

The earlier proposal included the following next steps if the audit confirms that topology-conditioned layers conflate offline topology-aware construction with online topology-conditioned prediction. Do **not** run these experiments or make these fixes during the diagnostic pass. Instead, evaluate whether each item follows from the evidence.

1. **Adopt the flat, no-sub-selection baseline first**, using flat categorical indexing (B2) and a single global-or-per-layer dictionary scope (B1 - pick one and state it explicitly):

   ```
   head_outputs = 32 + D × W + (D + 1)
   ```

   `W` constant across layers (no per-layer width variation), no gain/offset (out of scope, see below), no topology sub-selection of any kind.

   **Audit question:** Is this actually the missing baseline? Does the current implementation lack this comparison, or does some artifact already contain it under a different name?

2. **If dictionary scope is global-shared**, update the oracle construction process to optimize jointly across all `D` layers' residual distributions rather than per-layer, per the coupling note above. Document which construction policy was used and why, the same way 7A documents its five policies.

   **Audit question:** Does the current code support global-shared dictionaries, and if so, does the oracle already optimize jointly across layers or not?

3. **Treat binary-tree (or other structured) index representation as a separate follow-up experiment**, not bundled into the first baseline. It has a materially different cost formula (`D × log2(W)` instead of `D × W`) and a different learning problem (sequential path prediction vs. single softmax), so it deserves its own comparison rather than being folded into the topology fix.

   **Audit question:** Is binary-tree indexing present anywhere in code/configs/artifacts, or is it purely a proposed future direction?

4. **Verify clipping/decoder-policy variants (category C) add zero cost** under the new formula, the same way Exp 9's 9B/9C sections already treat them - this should be a non-issue but is worth an explicit checkbox given how much implicit cost was hiding in the topology handling.

   **Audit question:** Do clipping and snap policies truly remain decoder-only under the actual implementation?

5. **Run the missing controlled experiment**: all-shared (or all-per-layer, pick one) layers, no topology-conditioned layers, flat `D×W` cost, topology-aware-if-applicable oracle construction. Compare against the existing 7A/8/9 topology-conditioned results at matched `head_outputs` budget. This isolates whether topology-conditioned *layers* (B-level) ever earned their keep, independent of topology-aware *construction* (A-level), which no existing report tests separately.

   **Audit question:** Is this controlled experiment actually missing? If missing, what exact existing runs would it need to be compared against? Do not run it during this audit; just identify whether it is needed and what it would control for.

6. **Re-audit existing conclusions for this bias.** Any claim that topology-conditioned configurations are "output-head efficient" (eg. the Pareto frontiers, "W16D32 is the quality leader" at 1089 outputs) was measured under the `3W` accounting and should not be treated as settled until compared against a same-budget flat-categorical, no-sub-selection alternative.

   **Audit question:** Which existing conclusions are directly affected if the concern is confirmed, and which remain valid either way?

### Expected deliverable

Produce an audit summary, not a fix. The deliverable should state:

* what the code actually does;
* what the artifacts actually report;
* where code, artifacts, and prose agree;
* where they disagree or are ambiguous;
* which concerns in this document are confirmed;
* which concerns are falsified;
* whether the prior remediation hypothesis is supported, unsupported, or needs revision;
* what exact follow-up work would be justified after the audit.

## Explicitly not in scope

* Gain and offset scalars - leave as-is (still just `(D+1)` optional add-ons each), revisit separately.
* Per-layer variable `W` - rejected, do not introduce.
* Any topology sub-selection inside a layer's prediction, flattened or hierarchical - rejected in both forms; the model should never choose among multiple dictionaries within one layer.
