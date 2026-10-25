# Small correction-only sound fit: quality gate failed

Completed 2026-10-24 under the [frozen card](CONDITIONED_SHORT_FIT_CARD.md), plan `95869556df2179579993f905113c54bcfa715bd9086ac53d7ec2d077062fd799`. Both arms finish **48 paired updates** on 12 existing training patches / 36 recordings. All **4,184 native requests** are accepted: 3,768 training/source requests and 416 final-audit requests. There is no interruption or recovery. No development/test data is scored and no model is admitted.

The [independent verification](data/conditioned_short_fit_verification.json) reconstructs every update, both final models, Adam and RNG exactly; it reconstructs all hard patch/performance requests, recomputes source targets/features and rescores **312 candidate comparisons** from retained waveform bytes across 104 unique input/playback cases. Frozen predictor gradients remain absent and its 355,806 weights stay byte-identical through optimization and export. Every correction group receives a finite nonzero gradient after the first projection update, in both arms. The correction is connected to learning, but useful reconstruction has not been established.

| Training group | Change in combined sound loss | Change in MRSTFT | Fit gate |
|---|---:|---:|---|
| Original input, own performance | −0.68% | +1.01% | Fail |
| Original input, other performances | +3.71% | +2.60% | Fail |
| Wider held input, own performance | −0.15% | −0.74% | Fail |
| Wider held input, original performance | −0.01% | −0.34% | Fail |
| Wider alternate input, own performance | −1.75% | −1.63% | Fail |
| Wider alternate input, original performance | −0.77% | −0.76% | Fail |

Changes compare the sound-trained model with initialization; lower is better. The two original-input transfers are averaged within patch before averaging patches. Each group requires at least 10% lower combined sound loss, no worse combined sound loss than the matched parameter control, and strictly lower MRSTFT than initialization. **None of the six groups passes.** These are repeated training examples; they provide no fresh-family or held-out confirmation.

The separate original-32 preservation panel also fails: combined sound loss rises **8.68%**, mean MRSTFT rises **9.41%** (`0.323321 → 0.353729`) and MRSTFT p90 rises **23.91%** (`0.543779 → 0.673781`). Keeping the old weights frozen does not keep their outputs unchanged once the learned correction changes their inputs. The parameter control performs substantially worse on this panel (`1.033177` mean MRSTFT); outperforming that control does not satisfy the initialization comparison.

Native execution takes **7,749.75 seconds (2 h 9 min)** and peaks at **1.26 GiB** process-tree RAM; all resource gates pass. Retained evidence before final audit publication is **1.02 GiB**, below the calculated 1.23-GiB bound and 2-GiB cap. The independent audit takes **221.12 seconds / 1.09 GiB**, adding zero renders or training updates. [Native service](data/conditioned_short_fit_service_001.json) and [audit service](data/conditioned_short_fit_service_002.json) both exit successfully, release the scoped active-work request and record no RSS sampling errors.

The accepted initializer remains unchanged. The final sound-trained tensor digest is `e2d3407024be3084974c722e3b2663b86228bb8a2723812d8208aecfc068cba7`; the parameter control is `b782e705ee741573377c2cc09432a28844b0b8b088b2d773379d710de3731125`. Both remain diagnostic artifacts with 14 outputs and no release qualification.

Do not advance to a larger correction fit or expand prediction heads on this evidence. Next, inspect the saved per-input predictions and update effects to distinguish what this branch actually changed from what the sound-loss measurements asked it to change. Use that diagnosis to choose a structural repair; no learning-rate sweep, extra fit budget, parameter-only initialization or weaker acceptance gate is authorized by this result. A2/A3 and later stages remain open.
