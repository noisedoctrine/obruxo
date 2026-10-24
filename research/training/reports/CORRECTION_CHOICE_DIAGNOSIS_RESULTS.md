# The correction did not change its final category choices

Completed 2026-10-24 under the [saved-data card](CORRECTION_CHOICE_DIAGNOSIS_CARD.md), plan `42d39182defcf2ebcdd5d40121fa7defd40f299f19d8a0625aceebf96fe41f45`. The [bound report](data/correction_choice_diagnosis_summary.json) reconstructs 108 final model/input predictions and inspects 192 retained category-probe observations from the independently verified [failed fit](CONDITIONED_SHORT_FIT_RESULTS.md). It takes **7.19 seconds / 0.58 GiB**, with zero renders, zero added updates and all parent evidence unchanged. Seven focused tests, Ruff and diff checks pass. An initial preparation-only path lookup incorrectly treated a renderer file as a prediction module; it failed before creating a numerical plan or loading prediction models. The corrected implementation checks the prediction/measurement modules used by this diagnosis, then freezes its [plan](data/correction_choice_diagnosis_plan.json) before execution.

The final sound-trained model retains **every starting unison and filter choice across all 36 inputs**. Unison accuracy remains 12/12 on original inputs, 5/12 on wider-held inputs and 6/12 on wider-alternate inputs. Filter accuracy remains 12/12, 11/12 and 12/12 respectively. These are authored-label diagnostics, not sound-quality acceptance or a requirement to recover an acoustically unique patch.

| Saved native observations | Lower-cost unison edit available | Average sound-cost reduction when available |
|---|---:|---:|
| Original input | 0 / 48 | — |
| Wider-held input | 11 / 24 | 10.99% |
| Wider-alternate input | 16 / 24 | 12.17% |

Those **27 / 48 wider-input observations** already contain a measured sound signal favoring a different unison choice. They are repeated observations along optimization, with continuous controls held at each recorded anchor, not 48 independent recordings or final-model guarantees. Costs average input and reciprocal playback under the frozen objective. The parent audit reconstructs the measurements against accepted native receipts; this report does not independently rescore every probe waveform or claim gains from joint edits.

The sound-trained model mostly makes small continuous changes: mean normalized control movement is **1.26%** of range on original inputs, **0.75%** on held inputs and **0.66%** on alternate inputs. Active-control label error changes from 4.25% to 4.51%, 31.58% to 31.71%, and 32.46% to 32.16% respectively. Its category connectivity checks passed, but useful category decisions have not been learned.

The parameter control changes more controls and some categories, yet original-input control error rises from 4.25% to 16.33% and its actual sound result is much worse. This does not isolate the cause: score margins, shared feature correction, optimization duration and competing objectives remain possible contributors. Positive gradients alone do not distinguish them. The evidence does not establish that the codec, reconstruction loss or task is unsolvable.

Next, qualify an **independent unison-choice correction** that can change category scores while the earlier predictor, continuous controls and filter choice remain fixed. Train it from measured sound costs for the three legal choices and inspect whether it can realize the available improvements. Keep the final codec interface at 14 outputs, verify starting hard presets/export and score calibration, and freeze its smaller fitting protocol before native execution. This isolates one failed component; it is not a larger shared-correction retry and does not replace or weaken any full A2/A3 reconstruction gate. No trained model is admitted by this diagnosis.
