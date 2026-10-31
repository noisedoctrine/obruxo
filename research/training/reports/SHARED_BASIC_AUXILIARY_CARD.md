# Does removing pitch and note length help the small model reconstruct sounds?

This is the next Stage A2 experiment. Train one shared audio-to-patch model on the three sounds whose direct control fits passed (training indices 2, 8 and 11; held MIDI 36, 61 and 83). Compare native sound training alone with the same training plus a pitch/duration removal loss on its timbre branch. The separate proposed encoder screen is deferred; it is not a prerequisite for this comparison.

Both arms start from the same qualified untrained input bridge, with identical Adam states, data order, fixed preprocessing, original frozen predictor, 14 outputs, inward control derivative, native gradient estimator, production argmax, sound objective and reference rehearsal. Use eight seeded passes over the six reciprocal original/alternate and original/held pairs: 48 updates per arm. Keep the prior 0.001 learning rate, norm cap five, 0.25/0.5 paired weights and two round-robin references at 0.125 each. Keep the existing 0.025 control cap, at most three measured trials and actual sound-decrease acceptance. Refused model weights roll back while Adam advances once. The auxiliary term affects the proposed gradient; acceptance measures sound alone in both arms.

The candidate alone receives the added training loss. Before every update, fit a standardized ridge predictor of pitch and duration on the full 768-value timbre representation of the existing 259-patch auxiliary training split. Preserve its whole-patch exclusions. For diagnostic probes, start from the existing 145-patch split and exclude any patch exposed through sound training or reference rehearsal; freeze the exact remaining identities in the plan. Fit statistics use training recordings only. Use the full representation, including its mean and scale. The ridge head has no optimizer and adds no inference outputs. Train the encoder against its reversed prediction gradient on 32 deterministic patch pairs per update. MIDI labels supply only this temporary loss and rendering; deployed inference receives audio alone.

At update two, set one fixed coefficient to equal the sound and unscaled auxiliary gradient norms in the timbre encoder. Update one uses sound alone because the zero final correction layer initially blocks sound gradients to the encoder. Require finite nonzero norms and a coefficient between 0.001 and 1000; otherwise stop and report the calibration failure. Ramp the fixed coefficient over the first eight updates. Do not tune coefficients, heads, step counts or evaluation thresholds after viewing results.

Report all 21 existing comparisons for the three sounds, including all nine input recordings. Report MIDI 36, 61 and 83 individually, each input/playback combination, the original 32-patch preservation panel, and the unchanged six-group full-panel gates. The three-sound learning prerequisite requires each sound's mean combined loss to improve at least 25% from initialization, each of its comparisons to improve combined loss at least 10%, MRSTFT to improve on every comparison, and its two held-input playback losses to beat the saved matched settings-trained control. This demanding prerequisite is reported separately for each arm; a failure stays visible.

For a directional benefit, the auxiliary arm must reduce mean combined loss and MRSTFT by at least 5% against sound-only on the selected 21 comparisons, with no individual sound's mean combined loss or MRSTFT more than 5% worse. It must preserve the original 32 reference means and MRSTFT tail under the existing limits. Fresh linear and nonlinear pitch/duration probes and patch retrieval remain secondary diagnostics; their scores cannot qualify a worse reconstruction. These reused training/development sounds provide no fresh validation. A small or negative result applies to this setting and budget; it does not settle all possible separation losses.

Use local CPU float32, two Torch threads and one isolated native worker. Preserve fixed oscillator/sampler phases, the qualified renderer identity and 32,768-frame preroll. At most 12,672 gradient requests, 1,728 trial requests, 208 final requests and 12 exact-repeat requests are planned: 14,620 total, with a hard cap of 15,360 including recovery. Allow ten hours, 4 GiB process-tree RAM, 2 GiB evidence and 512 MiB resident replay. Allow twenty minutes and 4 GiB for the separate offline audit. Keep the original deadline across one permitted identical timeout recovery. Checkpoint both models, Adam, calibration, trace and RNG after every update. Record all attempts, failures and refusals. Independently replay the trajectory and rescore retained endpoint audio before drawing conclusions. Run focused software checks and a real-initializer software check before native execution; replay the first two native updates exactly before proceeding with the long job.

The prediction space stays at 14 values. This experiment advances the shared-learning task and tests the auxiliary loss together. Full A2/A3 reconstruction, fresh families, second seed, level-matched human listening and release requirements remain open. The 192 reserved roots remain unscored; GitHub Actions stays disabled and Colab remains deferred.

## Local execution

Run the focused tests for the comparison, ridge head, inward gradients and native acceptance, then Ruff and `git diff --check`. The installed Ruff is 0.16.2, differing from the repository's 0.12.2 contract pin; record the actual version with results.

From this worktree, use the existing local evidence and this command template:

```powershell
$pyExe = 'C:/Users/angert/miniforge3/envs/py312/python.exe'
$outputsRoot = 'C:/Users/angert/Documents/projects/OBRUXO/research/training/outputs'
$env:PYTHONPATH = 'C:/Users/angert/Documents/projects/OBRUXO/research/data_generation/outputs/smoke-deps'
$runArgs = @('research/training/reports/scripts/run_shared_basic_auxiliary.py',
  '--parent', "$outputsRoot/conditioned-short-native-fit-checked-20261024",
  '--bridge', "$outputsRoot/trainable-basic-input-bridge-20261026",
  '--features', "$outputsRoot/mixed-basic-features-20261022/features.pt",
  '--pool', "$outputsRoot/authored-expanded-20261013/pool.json",
  '--fit', "$outputsRoot/trainable-basic-sound-fit-20261026",
  '--auxiliary', "$outputsRoot/branch-note-removal-screen-20261028",
  '--output', "$outputsRoot/shared-basic-auxiliary-20261031")
& $pyExe @runArgs --mode prepare
& $pyExe @runArgs --mode check
& $pyExe @runArgs --mode execute
& $pyExe @runArgs --mode verify
```

Omit the `PYTHONPATH` assignment when dependencies are installed normally. Native execution runs under a hidden durable driver with a recorded process identity, progress/checkpoints and completion status. Use `--mode resume` only for the single permitted identical timeout, preserving the original output and deadline. Verification replays from initialization with no native worker.
