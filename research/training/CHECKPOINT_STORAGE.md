# Durable collection checkpoints

During initial audio collection, the model, optimizer and frozen frontend have not changed. The trainer now writes those values and the run contract once to `checkpoint.initial.pt`. Each acknowledged render writes a smaller `checkpoint.pt` containing the replay features and original labels, RNG states, cursor, renderer identity and counters. Accepted and rejected attempts are both persisted; feature extraction failures remain unacknowledged. Audio is still rendered on demand and is not archived.

The immutable initial file is written, flushed, verified and hashed before any collection checkpoint references it. New snapshots are flushed and verified before promotion; the previous committed snapshot is retained as `checkpoint.previous.pt`. Backup copies are flushed before their promotion too. The collection pointer accepts only the exact sibling filename and expected fields. Recovery verifies the initial file's hash before loading it and refuses missing, altered or incompatible dependencies.

Once initial collection and its initial metrics are complete, checkpoints return to the existing self-contained `obruxo_pilot_training_v1` format. Optimizer updates, exports and inference bundles keep their existing formats. Earlier checkpoints remain loadable. Run, sampler, split, renderer and learning contracts do not change because of the storage representation.

## Resume and inspect

The training CLI continues accepting `--resume path/to/checkpoint.pt`. Use the existing training configuration and a new output directory. To explicitly recover the preceding committed state, pass `checkpoint.previous.pt`. The trainer, feature-seeding path, paired feature recovery and evaluation loader all resolve collection checkpoints through the shared loader; evaluation still refuses a checkpoint with no optimizer updates.

```python
from pathlib import Path
from obruxo_training.checkpoints import load_training_checkpoint

saved = load_training_checkpoint(Path("run/checkpoint.pt"))
print(saved["step"], saved["cursor"]["next_sequence_id"], len(saved["replay"]))
```

During collection, raw `torch.load` exposes the versioned collection envelope rather than a complete training dictionary. Use the shared loader for training-state inspection. It restores the exact Python numeric labels and tensors, without requantizing targets. The existing contract, replay and cursor checks then run as before.

Keep `checkpoint.initial.pt` beside both collection snapshots. Copying a collection pointer alone is insufficient. For persistent storage, copy and verify the immutable initial file first, then promote verified copies of the latest and previous collection snapshots. A completed self-contained checkpoint can still be moved by itself. Keep the initial file while any retained collection snapshot depends on it; the trainer retains it so earlier recovery points stay usable. Storage is bounded by one initial file and the current/previous snapshots, in addition to the separate inference export.

The trainer summary reports initial, collection and full write counts plus serialized bytes. That byte count includes each primary serialization; it excludes backup-copy traffic and is not a disk-quota estimate. Full snapshots during ongoing online training remain necessary when model/optimizer state has changed. This increment removes repeated initial-state serialization during warmup without reducing recovery frequency.

## Local checks

```powershell
python -m pytest research/training/tests/test_checkpoints.py research/training/tests/test_training.py research/training/tests/test_paired.py -q
python research/training/reports/scripts/benchmark_checkpoint_io.py --checkpoint path/to/64-example-checkpoint.pt --output research/training/outputs/checkpoint-io
python research/training/reports/scripts/run_checkpoint_integration.py --pool path/to/pool.json --native-qualification path/to/combined-native --output research/training/outputs/checkpoint-native
```

The paired I/O benchmark saves identical growing replay prefixes with both methods and checks exact resolved state. It alternates execution order and counts the one-time initial write. Its fixed existing weights are an I/O fixture, not a training comparison. The native integration collects 64 distinct training states, snapshots before initial metrics and update one, continues training, and resumes the collection snapshot against the uninterrupted result. Exact retained features permit exact model/optimizer/RNG/replay/cursor comparison without new native renders during recovery. Native collection keeps the requested source phases; it does not claim a regenerated waveform match. The previously qualified source renderer uses fixed random phases and fresh-process repeats for that separate correctness claim.

Final-test audio remains reserved, GitHub Actions stays disabled and free local CPU remains the qualified platform. This storage change does not establish model reconstruction quality or Linux/Colab compatibility.
