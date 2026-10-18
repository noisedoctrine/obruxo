# Mixed-source small-batch fit

This check asks whether the complete patch model can learn eight retained real-audio examples before scaling the [mixed-source pipeline](MIXED_SOURCES.md). It is a training sanity check; it does not measure generalization or finish the approximately 32-example proof in the [training plan](../modelling/TRAINING_PLAN.md).

The source batch contains five authored and three original corpus presets. Its nine acknowledged attempts include one clipping rejection. The new run reuses all eight accepted features and their original labels, resets the model/optimizer/learning RNGs, and preserves the original recipe numbers and cursor. Audio remains generated on demand during collection and is discarded after feature extraction. No new source rendering is needed for fitting this saved batch.

`--seed-replay` checks the sibling `renders.jsonl` when a directly collected checkpoint has rejected/error attempts. The history must cover the exact acknowledged prefix; recipes, outcome counters, renderer identity and accepted audio hashes must match. Every accepted feature must still be present. A missing/changed history or an evicted accepted record fails before training. Resuming the seeded run uses its own checkpoint and requires no external source log. To seed another fresh model, retain the original collection checkpoint and log.

## Frozen decoder decision

Keep the existing immutable factory vocabulary and train-only Era2 LFO decoder for the first mixed-data baseline. Measure the reconstruction error obtained with perfect saved labels against the original curves; this separates decoder limits from learning error. Do not fit a dictionary on validation/test states or silently change targets. If that error prevents a required quality gate, propose a separately identified train-only representation change. The sampler, effects, curves, seeds and full connection destinations remain in the model throughout.

## Run

Use the qualified local Python 3.12/Vital environment. Substitute private artifact paths; they are not included in inference exports.

```sh
python research/training/run.py train --pool MIXED_POOL/pool.json \
  --output NEW_TRAINING_DIRECTORY --steps 1000 --replay-capacity 8 --batch-size 8 \
  --render-budget 32 --checkpoint-every 100 --cpu-threads 2 \
  --sampling-policy source_balanced --seed-replay SOURCE_COLLECTION/checkpoint.pt

python research/training/reports/scripts/qualify_mixed_overfit.py \
  --pool MIXED_POOL/pool.json --checkpoint NEW_TRAINING_DIRECTORY/checkpoint.pt \
  --source-checkpoint SOURCE_COLLECTION/checkpoint.pt --output NEW_QUALIFICATION_DIRECTORY
```

The unchanged strict fit gate requires continuous MAE below 0.05, categorical accuracy exactly 1, and final loss below 5% of initial loss. The fixed limit is 1,000 updates with the existing learning rate, model and loss. A failed gate remains a failure. Collection and optimization timings are reported separately.

The initial sigmoid-output run failed the categorical gate. A focused gradient test demonstrates that a sigmoid-constrained phase can become trapped near zero when the shortest route to the target crosses the phase boundary. The opt-in `--cyclic-head periodic` predicts unbounded linear phase and wraps it modulo one. Phase biases start at their normalized defaults; all other continuous and categorical heads keep their original initialization and behavior. The matched corrected run adds only this flag to the command above. The model specification, training contract, checkpoint recovery, inference loader, evaluation and optional candidate learner preserve this choice. Earlier model/training specifications omit the new field and continue to use sigmoid outputs. This changes model outputs, not original source phases, labels, codec identities or the frozen decoder.

The qualification checks frozen features, labels, decoder identity and export weights, then loads every predicted patch in real Vital. Each prediction is repeated in a fresh process with all three oscillator random phases and the sampler random phase fixed. Other phase controls, seeds and routes remain as predicted. These diagnostics do not change the training source or claim regenerated training waveforms are identical. The process-tree limit is 8 GiB. Final-test audio remains reserved; Linux/Colab native rendering remains unqualified.
