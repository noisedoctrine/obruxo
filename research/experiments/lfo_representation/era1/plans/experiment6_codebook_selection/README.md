# Experiment 6: Codebook-Generation Approach Selection

Experiment 6 is a design-selection experiment. It does not precompute or freeze the final production LFO codebook.

The goal is to decide the recipe we should later use to generate the production codebook:

- sampling grid;
- candidate codebook family;
- fitting objective;
- alignment oracle;
- complexity accounting;
- learnability gate;
- decision metrics and plots.

The detailed plan is in [EXPERIMENT_6_PLAN.md](EXPERIMENT_6_PLAN.md).

## Current status

Runnable implementation is available.

- Module: [experiment6.py](../../code/lfo_experiment/experiment6.py)
- Worker: [experiment6_worker.py](../../code/lfo_experiment/experiment6_worker.py)
- Background runner: [run_experiment6_background.cmd](../../code/run_experiment6_background.cmd)
- Output directory: [artifacts/codebook_selection/](../../artifacts/codebook_selection/)
- Completion marker: `../../artifacts/codebook_selection/COMPLETED_EXCPERIMENT_6.txt`

The implementation evaluates oracle reconstruction, factor-of-3 direct-grid baselines, phase-residual candidate families, threshold coverage, editor-node preservation, complexity accounting, Pareto plots, and pseudo-AIC/BIC diagnostics. Neural predictor training is intentionally not part of the default run; this run is for choosing the codebook-generation approach before we spend more time on inferability.

## Runtime behavior

Experiment 6 now checkpoints each candidate independently under:

```text
../../artifacts/codebook_selection/checkpoints/
```

Restarting the experiment skips completed checkpoints and aggregates whatever is complete at the end. Candidate-level parallelism is available with `--parallel N`; use `--parallel 2` as the cautious laptop default.

Progress is written to:

```text
../../artifacts/codebook_selection/progress.json
```

The status command reports completed candidates, estimated workload progress, elapsed time, and ETA.

## Key idea

Experiments 1-5 showed that the best direction is not a pure discrete library. The current best family is:

```text
discrete canonical atoms
+ circular phase offsets
+ clipped residual gains
+ residual stacking
+ exact no-op paths
+ shared/topology correction branches
```

Experiment 6 will compare ways of constructing that family and decide which approach should be used for final precomputation.

## Main design pressure

We need enough data to choose together, not a single automatic winner. The experiment should plot quality against complexity:

- reconstruction error;
- near-exact reconstruction share;
- editor-node preservation;
- output dimensions;
- number of codes;
- effective index bits;
- stored dictionary size;
- learnability from dense LFO curves.
