# Match both performances with each predicted patch

Prepared before native work on 2026-10-23. **Fifty local checks pass**, covering both historical transfer-only and new input/transfer execution, exact model/Adam/RNG recovery, independent waveform rescoring, changed performance rejection and failure in the second component before either model advances. Ruff passes. Freeze the implemented input/compute plan before execution. This is a new run, not another continuation of the failed paired fit.

The [paired result](PAIRED_NATIVE_FIT_RESULTS.md) loses original-input reconstruction despite wider-input gains. It also leaves a gap against the training plan: an original-input prediction is scored only under the wider performance, and a wider-input prediction only under the original performance. Correct that objective before further changes to model capacity or training data. The regression does not establish this gap as its sole cause.

For each of the two independent predictions in an update, reuse the exact same legal candidate patch under **both its input performance and the reciprocal performance**. Compare each render with the corresponding real source recording, and return the arithmetic mean of the two sound losses to the native estimator. Both performances have weight one half for that prediction; the two predictions also have equal weight in the update. MIDI and source settings remain outside the deployed network. The sound candidate receives no parameter loss.

Keep the accepted initializer, 64 training patches, original 32-state subset, three input recordings, reciprocal schedule, 128 paired updates per arm, singleton model forwards, raw-logit parameter control, production argmax, fixed backward scales, optimizer and temperature schedule. Keep **14 outputs / 355,806 parameters**, fixed generator phases, the settled 32,768-frame lead, CPU float32, two Torch threads and one native worker. This isolates the added input-performance sound term from further architecture, input or optimizer changes.

Record each component render separately with its performance, candidate, measured loss, request ID and audio hash. The two component rows must refer to the same candidate and ordered input/reciprocal views. The independent auditor recomputes their mean before reconstructing the gradient and update. A partial pair cannot advance either optimizer. Seed Python and NumPy explicitly as well as Torch for this new experiment; preserve and independently replay all three RNG states.

Save both models and Adam after every update. The first four real updates are the small end-to-end check within this run: validate both performance components, finite sound gradients and exact model/Adam/RNG replay before continuing. They consume the main allocation and do not select a quality checkpoint. Score only the final 128-update models. All full-64/original-32 fit gates and conditional development gates remain unchanged; a fit failure skips development and admits neither model.

| Native work | Worst-case planned requests |
|---|---:|
| Three source recordings for each of 64 patches | 192 |
| 128 updates × two predictions × two performances × 22 control probes | 11,264 |
| Final fit comparisons | 768 |
| Conditional existing development panels | 504 |
| **Planned total** | **12,728** |

Cap cumulative requests at **13,120**: 11,840 training and 1,280 qualification attempts, including failed or interrupted work. Cap charged update attempts at 132 per arm. Four complete extra paired attempts plus source requests require at most 11,808 training requests; the phase cap leaves 32 additional attempts. This explicitly replaces the previous single-performance request allocation. No intermediate quality sweep or automatic budget extension is permitted.

Use a new **ten-hour** deadline from the original start of this new run, **8 GiB** process-tree RAM, **512 MiB** replay and **5 GiB** evidence. These are ceilings, not promises of runtime. The completed prior run's last 500 accepted calls average 1.418 seconds wall time for 3.975 seconds of audio; across all accepted calls the mean is 2.183 seconds. At that latter rate the request ceiling alone costs about eight hours. Preparation retains a separately measured ten-minute allowance. The independent no-render audit retains twenty minutes / 4 GiB. Recompute the conservative storage bound with an additional 256-MiB allowance for the doubled ledgers and reject any estimate above the evidence ceiling before execution.

Render sources and probes on demand; discard intermediate probe audio under the existing contract, retaining component losses and provenance for exact optimizer replay. Keep every final comparison waveform for independent rescoring. A scoped active-work request may prevent automatic idle sleep without changing persistent power settings. Any interruption keeps the original deadline, failed attempt, checkpoint and ledger prefix. No extra launch is inferred from remaining optimizer attempts.

Entry point: `reports/scripts/run_paired_native_fit.py --input-and-transfer` prepares the new plan; add `--execute` to run or `--execute --resume` for explicit recovery. `--verify` derives the objective from the saved plan and independently reconstructs both performance components. A local durable operator records its own hash, requires the frozen plan-file hash, and permits at most three explicit native executions; final no-render verification is separate. Local only. GitHub Actions remains disabled and Colab is deferred. A2 admission, fresh-family confirmation, a second seed, blinded listening and release remain open until their existing quality gates pass.
