# Basic sound-loss training learns, but misses the development target

Completed 2026-10-21 under the [frozen fresh-run card](SETTLED_NATIVE_INVERSE_CARD.md). Both matched models complete **128 updates**. The sound-trained model passes both training-performance fit gates, with roughly **33–34% lower primary sound loss** than the starting model. On held-out development patches, native spectral error improves **5.48% on reference** and **3.50% on transfer**, below the required 10%. The numerical development gate fails. The retained worker timeout also fails the original all-attempts-accepted condition. No model is promoted and Stage A2 remains incomplete.

[Full replay/native audit](data/settled_native_inverse_fit_summary.json) · [Preserved 106-update interruption](data/settled_native_inverse_timeout_prefix.json) · [Host sleep evidence](data/settled_native_inverse_host_sleep.json) · Frozen training source `063bf30` · Plan `7fc5c19575d3869d3b82d9790b2c89d50c0e92f69ce702971d1446a69c8e5801`

## What the small model learned

The head remains **14 values**, with 355,806 trainable parameters. Both arms start from the same historical model, see the same audio inputs and state/performance order, and use fresh matched Adam optimizers. Sound training has zero parameter anchor. Every source and candidate uses the qualified settled renderer and fixed random-phase policy. All measurements are generated on demand; no old cold targets or stopped-run prefix is imported.

| Training performance | Starting primary loss | Parameter arm | Sound arm | Fit gate |
|---|---:|---:|---:|---|
| Reference | 0.56675 | 0.54676 | **0.37732** | Pass |
| Additional training MIDI | 0.62013 | 0.64291 | **0.40721** | Pass |

Each group includes all 32 training states. The sound arm meets the required 10% primary-loss reduction, is no worse than the parameter arm, and lowers legacy MRSTFT. This establishes learning on the fit population; generalization is evaluated separately.

## Development comparison

All 12 predeclared development states are present under both performances. Each method infers one patch from reference audio and reuses it for transfer. Lower native MRSTFT is better.

| Method | Reference | Transfer |
|---|---:|---:|
| Fixed template | 2.69093 | 2.70709 |
| Train-only retrieval | 1.06871 | 1.17729 |
| Historical starting model | 0.98473 | 1.00708 |
| Matched parameter arm | 1.04032 | 1.04689 |
| Sound-trained arm | **0.93080** | **0.97187** |

The sound arm beats template/retrieval and has lower primary spectral loss than the parameter arm. Its MRSTFT gains over the parameter arm are **10.53% reference / 7.17% transfer**; gains over the starting model are **5.48% / 3.50%**. Both groups miss the required 10% improvement against both learned controls. Tail-error limits pass. No threshold, case membership or denominator changed.

Seven of 12 states improve against the starting model on each performance. Paired family-level 95% intervals include no benefit: MRSTFT change −0.19390 to +0.08534 on reference and −0.15337 to +0.08513 on transfer. These intervals are descriptive and are not fresh confirmation. This panel has now influenced development; final-test states remain untouched.

## Category learning needs attention

Native category alternatives change the sound objective throughout training, but little of that signal reaches the category scores. Median score-gradient L1 is **1.11e-16 for filter on/off** and **8.53e-7 for unison voices**, versus **1.55** for the continuous outputs. These are raw diagnostics, not calibrated comparisons between parameter families. The filter score receives exactly zero gradient on eight updates despite a measurable native response; unison does so on one update.

Across the 32 training states, the sound-trained model changes **zero filter choices** and **one voice-count choice** from the starting model; the parameter arm changes four and seven. Exact source-category recovery is not a quality gate. These measurements suggest checking whether overly decisive existing scores prevent useful category learning. Inspect actual score margins and a bounded correction to the Gumbel learning path before another full training run. Do not expand the vocabulary or simply extend this run.

## Recovery, cost and verification

- **3,017 native attempts:** 3,016 accepted, zero audio-QA rejections and one retained worker timeout. The identical timed-out request later succeeds. All **88 source-repeat checks** match exactly.
- **129 charged optimizer attempts** for 128 committed matched updates. The interrupted update is charged; its partial measurements are retained.
- **12,300.29 seconds (3 hours 25 minutes)**, including the verified roughly 76-minute battery-triggered host sleep. The original four-hour deadline is not reset.
- **1,616,433,152 bytes (1.51 GiB)** cumulative peak process-tree RAM; **762,659,933 bytes (0.71 GiB)** retained evidence, including the interruption snapshot. All declared resource caps pass.
- Both final model weights, both Adam states and Gumbel RNG replay exactly from logged native losses. The archived 106-update checkpoint independently replays within the same sequence. Saved inputs and spectral targets are independently reconstructed from exact source-repeat audio; hard audit patches, native receipts, profile metadata and quality gates are checked.

Candidate audit audio is discarded as declared. Its scores are verified against the pinned evaluator/receipts rather than independently rescored from retained waveforms. Five focused report/recovery/diagnostic tests, Ruff and whitespace checks pass locally. The model remains a diagnostic export with `qualified_for_release=false`. GitHub Actions remains disabled and Colab deferred.

The next work remains inside the basic stage: investigate the weak category-learning path, freeze its correction and budget before another matched run, then require the existing held-out improvement gates. Fresh confirmation, second seed, actual listening and bundle qualification follow only after a promising qualified comparison. Wider oscillators, effects, sampler and modulation are still later stages.
