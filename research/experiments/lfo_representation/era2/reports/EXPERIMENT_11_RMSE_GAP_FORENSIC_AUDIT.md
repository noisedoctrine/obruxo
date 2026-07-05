# Experiment 11 RMSE Gap Forensic Audit

## Main Findings

The large Era 1 vs Era 2 RMSE gap should be read first as a reconstruction-pipeline problem, not as a budget problem.
The primitive array checks clear the basic RMSE calculation, circular shift behavior, and Experiment 9 decode replay.
The strongest concrete difference is residual-layer gain: Era 1's `phase_only` rows still saved and applied a gain scalar for every residual layer, while the canonical Era 2 flat path did not.
The audit also fixed an Era 2 circular-shift precision bug: tiny near-zero phases now stay near the identity transform instead of occasionally wrapping into a bad interpolation fraction.

In the replayed Era 1 W8D16 checkpoint sample, using the saved residual-layer gains gives P95 RMSE `0.0254555`.
Forcing those same residual layers to gain `1.0` gives P95 RMSE `0.43257`.
That is the cleanest current explanation for why the Era 2 quality numbers look wildly worse: the supposedly comparable Era 1 rows were not fixed-amplitude residual atoms.

Changing only residual-layer gains changes the reconstructed curve by P95 RMSE `0.432016` between decodes.
The saved residual-layer gain absolute P95 is `0.661099`, with nonzero rate `0.584473`.

## What Passed

- `rmse_same_array`: pass (max_abs_delta `0`). Same target/prediction arrays produce identical RMSE.
- `circular_shift_parity`: pass (max_abs_delta `5.96046e-08`). Era 1 and Era 2 circular shift agree on endpoint-excluded sampled curves.
- `tiny_phase_identity`: pass (max_abs_delta `0`). Era 2 circular shift keeps near-zero phases near the identity transform.
- `era1_decoder_replay_parity`: pass (max_abs_delta `2.38419e-07`). Era 2-side artifact decoder reproduces the Experiment 9 final-only decode from the same saved encoding.

These checks mean the first-order failure is not the RMSE formula, ordinary circular shift behavior, or a decoder replay mismatch. The gap is higher in the stack: which residual scalars are applied, how paths are searched, and how atoms are constructed.

## Still Open

- `exact_phase_gain_alignment_parity`: unsupported (max_abs_delta `n/a`). Could not import Era 1 exact phase/gain alignment: boolean index did not match indexed array along axis 2; size of axis is 64 but size of corresponding boolean axis is 1

Exact phase/gain alignment parity still needs a cleaner cross-era probe. That does not weaken the residual-gain finding above, because the artifact replay uses saved Era 1 indices, phases, and gains rather than re-solving alignment.

## Implication For Experiment 11

The next meaningful Experiment 11 candidate should include optimized residual-layer gain as a model-facing scalar family. Beam width 4 and offline topology-aware construction are still useful, but neither explains the huge gap on its own.

Do not compare Era 2's fixed-amplitude flat path against Era 1's `phase_only` rows as if both use the same decoder degrees of freedom. In Era 1, `phase_only` means no extra modifier/base gain family; it does not mean residual atom gains were absent.

## Method Notes

- Era 1 checkpoint: `C:\Users\angert\Documents\projects\OBRUXO\research\experiments\lfo_representation\era1\artifacts\additive_finalization_9_screen\checkpoints\9_screen_9B_phase_only_final_only_phase_only_phase_only_raw_final_only_none_W8D16_bw4_eval120_sample33_seed7267`
- Era 1 catalog: `C:\Users\angert\Documents\projects\OBRUXO\research\experiments\lfo_representation\era1\artifacts\lfo_catalog.csv`
- Replay row limit: `256`
- `W` remains residual-layer atom choices, not grid subdivisions.
- Topology-runtime replay is forensic only. It is not an Era 2 deployable contract.
- Detailed probe values are in `era2/artifacts/experiment_11/rmse_gap_audit/probe_summary.csv`.
