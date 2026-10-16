# Direct-render selection and learning

Vital confirmed alternative patch guesses for **13/32 training recordings**. Learning from those extra targets changed mean held-out native error by **+12.56%** relative to the matched source-supervised control. This fixed single-seed authored-alpha comparison does not promote a release model.

[Protocol](../RENDER_SELECTED_TRAINING.md) · [Frozen screen](data/render_selected_plan.json) · [Collection plan](data/render_selected_data_plan.json) · [Aggregate evidence](data/render_selected_summary.json)

## What was compared

Thirty-two training patches were selected outside the source inverse checkpoint's retained training states. Proposals used its ordinary prediction, training-only audio retrieval, one categorical alternative and a bounded continuous change. The source patch label and known notes never entered the proposer. Actual training MIDI was used only to render and score proposals.

The best proposal and ordinary prediction each received an independent confirmation render. An extra target was retained only when the candidate's worse draw beat the ordinary prediction's better draw by at least 5%. Ground-truth parameter supervision remained an anchor in both training arms.

Both arms received identical warm-start weights, optimizer initialization, old/new feature minibatches and draw counts for 1500 updates. The sole objective difference was confirmed-target weight zero versus 0.5. Validation used ordinary single predictions, not evaluator best-of-four.

| Model | Mean reference/transfer MR-STFT |
|---|---:|
| Matched source-supervised control | 1.5600 |
| Additional confirmed targets | 1.7559 |

Candidate-minus-control paired mean: **+0.1959**; descriptive 95% state-bootstrap interval: **[0.048180687700708734, 0.34148854165648435]**. Candidate wins: **2/10** states; ties: 0. Both performances of a patch stay in the same resampling unit. The validation patches are reused development cases, not a fresh final test.

## Native evidence and compute

Training selection accepted 220/220 native attempts, with 32 accepted sources. Validation accepted 120/120 renders. Failed and unscored cases retain their planned denominators. Confirmed target methods: {'retrieval': 12, 'category': 1}.

Selection took 16.06 CPU minutes after source loading. The matched learning arms took 0.99 minutes; native validation took 9.07 minutes after checkpoint loading. These segments exclude plan/pool and repeated contract verification, and some work overlapped the local software suite. They are not a dedicated throughput benchmark.

Retained feature storage including the original bank was 5.83 MiB. Sampled selection peak RSS was 0.83 GiB. Execution used local CPU float32 and four threads; no paid compute or audio archive was used.

The full local training/renderer suite passes **173 tests**. New checks cover legal proposals, training isolation, independent confirmation, QA rejection, feature/cache integrity, exact optimizer recovery, matched minibatches, portable audio-only exports and render-attempt caps including recovery. GitHub Actions stays disabled.

## Remaining plan

The next independent capability is deployed audio-only candidate ranking, evaluated separately from oracle best-of-four. A conditioning comparison, broader performance/control coverage, measured data scaling and release gates remain open in the completion ledger. A successful training-case alternative does not itself establish improved inverse generalization, and a small development comparison does not complete the finished-model plan. Final test states remain reserved.
