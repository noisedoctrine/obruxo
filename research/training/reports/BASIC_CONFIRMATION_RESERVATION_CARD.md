# Reserve the basic confirmation panel before scoring a candidate

This step reserves data only. It generates no audio, extracts no features, trains nothing and scores no model. A2 and all existing reconstruction/preservation gates remain required before a separate confirmation execution plan can be frozen. Reserved cases remain excluded from training even if the current fit fails.

The existing 512-state population has 404 training, 44 used development and 64 reserved test states. Extend its deterministic authored prefix to 4,096 states while preserving every old target, lineage and split. Select **192 new validation roots**, 32 from each of the six unison/filter combinations. Rank within each combination by a fixed rule/state hash; take the first 32. These are independent authored parameter draws within the same one-oscillator basic vocabulary, not evidence about additional sound capabilities. Do not select cases using predictions, quality scores or source audio.

Reserve eight sets of three fresh performances: one reference, one changed-register held note and one changed-note/velocity pattern. Every category combination receives four families per set, preventing performance assignment from revealing a category. The 24 performances must differ from the original bank and the development fit's performances. In the later evaluation, infer each method's patch once from reference audio and reuse it across all three performances; known MIDI remains renderer/evaluator input only.

Size planning uses the independently audited, already-used direct-fit evidence. Its paired relative standard deviations are approximately 0.514 reference / 0.629 transfer. A 48-family panel would have approximate 95% interval half-widths of 14.54% / 17.80% of the control mean. A 10% half-width estimate requires 102 / 153 families; select 192 before fresh scoring. This normal approximation uses 32 reference and only 12 transfer families, so it does not guarantee fresh-panel precision or power. Actual family-level paired intervals still decide admission; no metric threshold is relaxed.

The proposed confirmation budget explicitly replaces the old 960-request allowance **before collecting any confirmation audio**:

| Per family | Requests |
|---|---:|
| Source under reference and two transfers | 3 |
| Candidate, matched parameter control, previous model, fixed template and training-only retrieval under all three | 15 |
| Exact fixed-phase reference-source repeat | 1 |
| Total for 192 families | 3,648 |
| Recovery reserve | 192 |
| Hard request cap | 3,840 |

Declare four hours, 8 GiB process-tree RAM and 8 GiB retained evidence, one native worker and two CPU learning threads. This is evaluation, with zero optimizer updates. Keep 44.1-kHz stereo float32, all oscillator/sampler random phases fixed, 32,768-frame preroll and a two-second tail. Each planned observation fits five seconds; raw audio is bounded by 6,435,072,000 bytes, plus a 512-MiB archive reserve, within 8 GiB. Failed/invalid outputs remain recorded and prevent clean admission. A fit failure never authorizes this evaluation or a second seed.

Before rendering, the separate execution plan must pin admitted candidate/control weights, previous-model and retrieval artifacts, full numerical source/runtime, scoring/interval/tail rules, failure handling and listening selection. Qualify the training-only retrieval path and no-render checkpoint/recovery behavior. Keep all old regression cases and final-test reservations. This reservation is not a completed evaluation or a model release.

Preparation has ten minutes / 4 GiB and zero native requests. Run from the repository root with the existing local artifacts:

```powershell
python -I research/training/reports/scripts/reserve_basic_confirmation.py --pool research/training/outputs/authored-expanded-20261013/pool.json --development-fit research/training/outputs/basic-direct-native-fit-20261024 --pending-fit research/training/outputs/trainable-basic-sound-fit-20261026 --features research/training/outputs/mixed-basic-features-20261022/features.pt --output research/training/outputs/basic-confirmation-reserved-20261026
```

The output must be new. It contains the expanded pool, immutable reservation and measured preparation summary. The original pool and all running-fit inputs remain unchanged. Five local checks cover independent roots, split preservation, balanced assignment, valid bounded performances and family-level precision planning with tamper rejection.
