# Broader sound-training coverage does not resolve held-out reconstruction

Completed 2026-10-22 under the [frozen 64-state card](BROADER_NATIVE_INVERSE_CARD.md). Both models complete **128 updates**. Sound training passes the complete 64-state fit requirements and the separate retained-original-32 requirements. Held-out native spectral error improves **5.82% on reference MIDI**, but is **0.24% worse on transfer MIDI** than the starting model. The unchanged 10% target fails. **Stage A2 remains incomplete; no model is promoted.**

[Independent replay and native audit](data/broader_native_inverse_fit_summary.json) · Frozen source `3bca06f` · Plan `8f2a9c858d8c0d2ca7d69deebc1c2a5f2f5d45e4e040619fde7de1d1574f2f53`

## What was compared

Retain the original 32 patches and exact MIDI recipes, then add 32 from training assignments using the declared category-constrained coverage rule. Development/test targets and sound scores do not enter selection. Both arms start from identical historical initial weights with fresh Adam state. Two passes over 64 patches replace four over 32; the update count, optimizer, fixed category scales, frontend, 14 outputs, 355,806 parameters and native renderer stay unchanged. Every target renders fresh during training. The sound arm receives no parameter anchor; audit/export choices use raw-score argmax.

## Both training-fit requirements pass

Mean primary sound loss, lower is better:

| Population / performance | Starting model | Parameter arm | Sound arm | Sound gain vs starting model |
|---|---:|---:|---:|---:|
| All 64 / reference | 0.58436 | 0.54941 | **0.39638** | 32.17% |
| All 64 / additional MIDI | 0.60745 | 0.58532 | **0.41989** | 30.88% |
| Original 32 / reference | 0.56675 | 0.53998 | **0.39410** | 30.46% |
| Original 32 / additional MIDI | 0.62013 | 0.60344 | **0.43671** | 29.58% |

All four groups meet the original primary-loss, parameter-comparator and MRSTFT fit requirements. Added states do not hide an original-subset failure. This establishes fitted-population learning, not generalization.

## Held-out development result

All 12 declared states appear under both performances. Each method predicts once from reference audio and reuses the same patch for transfer. Lower native MRSTFT is better.

| Method | Reference | Transfer |
|---|---:|---:|
| Template | 2.69093 | 2.70709 |
| Training-only retrieval | 1.06871 | 1.17729 |
| Historical starting model | 0.98473 | **1.00708** |
| Matched parameter arm | 1.04057 | 1.06259 |
| Sound-trained arm | **0.92744** | 1.00952 |

Against the starting model, the sound arm reduces reference error **5.82%** and raises transfer error **0.24%**. Against the parameter arm, improvements are **10.87% reference / 4.99% transfer**. Both groups miss the requirement to beat both learned controls by 10%. Template/retrieval, primary-loss-versus-parameter and tail limits pass. Primary transfer loss is nevertheless **5.58% worse than the starting model**. No case, denominator or threshold changed.

Seven of 12 states improve against the starting model on reference; four of 12 on transfer. Whole-state paired 95% MRSTFT-delta intervals are **−0.19525 to +0.07256** and **−0.12005 to +0.11621**. Both include no benefit. These remain descriptive development results; final-test states are untouched. This trial does not establish that the training-only coverage rule improves generalization, so do not expand it again automatically.

Both category heads receive finite nonzero gradients at all 128 updates. Median raw-score gradient L1 is 7.04e-4 for unison and 1.09e-3 for filter. These are learning-path diagnostics, not sound-quality gates or evidence for adding more outputs.

## Cost and verification

- **3,332 native requests, all accepted:** 2,652 training and 680 audit requests. All **152 source-repeat checks** are sample-exact. No interruption or retry occurs.
- **128 charged matched optimizer attempts**, all committed for both arms.
- **6,249.32 seconds (104.16 minutes)** elapsed; **1,652,416,512 bytes (1.54 GiB)** peak process-tree RAM; **1,393,142,576 bytes (1.30 GiB)** retained evidence. All original time/RAM/archive caps and the revised attempt cap pass. Retained evidence is below the 1.55-GiB planning allowance.
- Independent replay reproduces every actual update, both final models, both Adam states and Gumbel RNG. The full and original-subset fit gates, training-only selection trace, native requests and hard audit patches reconstruct exactly. Retained source-repeat audio independently reconstructs the cached input features and spectral targets.

Candidate audit audio is discarded as declared; its scores are checked through evaluator receipts, not independently rescored from retained candidate waveforms. Twenty focused checks, Ruff, whitespace and final-outcome checks pass locally. The user authorized battery-powered execution with checkpoints. No power setting is changed. Diagnostic exports remain `qualified_for_release=false`; GitHub Actions stays disabled and Colab deferred.

Next, check the historical initial model and its input features against the corrected renderer before another fit. The stopped cold-renderer run and settled run retain **32 matching patch/MIDI records**, with identical frontend and initial weights. They allow a paired input/prediction comparison without new audio. This is a hypothesis check, not a claim that initialization is the cause; cold records remain diagnostic and cannot be imported as settled training targets. A3 confirmation, second seed, actual listening, bundle qualification and later bounded capabilities remain pending.
