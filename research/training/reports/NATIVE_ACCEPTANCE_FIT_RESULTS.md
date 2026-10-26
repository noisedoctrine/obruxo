# Measured-acceptance basic fit: completed comparison

The saved 48-update fit **fails the frozen reconstruction gates**. It accepts 44/48 weight changes using the actual rendered batch objective. This establishes the recorded training behavior, with the original reconstruction and preservation requirements still controlling admission. No model is qualified for release.

## Reconstruction

Positive gain means lower combined sound error than initialization. A positive control gap means the sound-trained arm is worse than the unchanged parameter-trained control. Each group must improve at least 10% from initialization, match or beat the control, and reduce MRSTFT.

| Input and playback group | Gain from initialization | Error versus control | Frozen gate |
|---|---:|---:|---|
| original input own | 11.28% | -46.10% | Pass |
| original input transfer | 10.51% | -43.64% | Pass |
| wide alternate input own | 17.07% | -15.22% | Pass |
| wide alternate input transfer | 13.08% | -4.83% | Pass |
| wide held input own | 13.66% | +8.91% | Fail |
| wide held input transfer | 13.74% | +11.11% | Fail |

Earlier 32-sound gains must also match or beat the stronger previous sound model. Negative changes below are improvements.

| Measure | New sound model | Previous sound model | Change |
|---|---:|---:|---:|
| Mean combined error | 0.108106234 | 0.110012800 | -1.73% |
| Mean MRSTFT | 0.284369507 | 0.293334691 | -3.06% |
| MRSTFT p90 | 0.428812426 | 0.426978835 | +0.43% |

Earlier-sound preservation: **Fail**. Accepted-update minimum: **Pass** (44/48, minimum 36). Original-regression gate: **Pass**. Full numerical decisions and all group means are in the [summary](data/native_acceptance_completion_summary.json).

Accepting a change that helps its current training batch does not guarantee better reconstruction from another recording of the same patch. That transfer gap remains substantial: 8.91% and 11.11% worse than the control in the two held-recording groups. Keep the failure gates unchanged. Next work stays in Stage A2 and must address performance transfer while preserving earlier sounds; this result does not justify more output heads or simply extending the same fit.

## Completion and evidence

The original process stopped after saving all 48 updates and 201/208 final comparisons. It left one charged qualification request without a receipt. Its cause is unknown, its original terminal resource report is absent and its six-hour budget is not certified. Original peak RAM remains unknown.

The [separate frozen completion](FINAL_FIT_COMPLETION_CARD.md) replays the saved model/Adam/RNG exactly and rescores all 201 retained results before launch. Preparation takes 44.35 seconds / 0.86 GiB. It then adds **only seven native requests and zero experimental optimizer updates** in the second originally allocated native execution. Original files stay immutable; copied ledgers preserve their byte prefixes. Only checkpoint plan identifiers are rebound.

The absent receipt is explicitly annotated as unobserved process loss, with no audio or QA claim. Its exact request is accepted when repeated. Overall: **5,801 training + 209 qualification = 6,010 charged requests**, including the unobserved attempt; 6,009 have observed accepted receipts. No timeout is invented. The original 7,584-request / 49 charged-update / two-execution caps remain intact.

The detached completion and independent audit take 407.00 seconds / 1.14 GiB, with zero RSS sampling errors. Aggregate retained evidence before final reporting is 1.13 GiB across both directories, within 2 GiB. These new resources do not establish the missing original resource measurements.

The [independent audit](data/native_acceptance_completion_verification.json) reconstructs all 48 updates per arm, model/Adam/RNG, every one of the 5,801 retained scalar probes, the early checkpoint and matched parameter control. It independently rescores all **312** source/initial/final comparisons and checks cold model exports. Transient training-probe audio is not claimed to be independently rescored. Nine focused local tests pass, plus Ruff and diff checks. Actions remains disabled; no paid/cloud work.

The 14-output scope, source panels and final checkpoint selection are unchanged. No new held-out/test data are consumed. A2/A3 admission, broader fitting, fresh families/performances, second seed, listening and deployment remain open.
