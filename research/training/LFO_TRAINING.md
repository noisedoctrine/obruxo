# Learning modulation curves from audio

The curve model adds eight LFO shapes to the [scalar and factory-asset outputs](FACTORY_ASSETS.md). Training audio comes from the **original authored curves**, with three to eight control points in the current curriculum. The model predicts the existing Era2 codes, which rebuild each shape as 97 legal Vital points. The [train-only curve bank](LFO_BANK.md) stays fixed. Only audio-derived timbre and frozen Basic Pitch features enter the model.

| Output | Budget |
|---|---|
| Existing full scalar controls and factory choices | 424 continuous values, 410 categorical heads |
| Eight LFO codes | 264 continuous phase/gain values, 136 categorical heads |
| Total | 688 continuous values, 546 categorical heads |

The LFO part adds 1,544 prediction outputs: 193 per slot. All eight numbered shapes remain distinct. Phases wrap around a cycle, gains stay within `[-2,2]`, every residual layer retains its zero atom, and only the final reconstructed curve is clipped to `[0,1]`. The existing scalar decoder still serializes controls on Vital's native float32 grid.

## Original sound and training labels

The pool stores legal code targets and hashes the original source patch. A versioned authoring policy reconstructs the original shapes for rendering. Pool reload verifies that each saved code label matches its original source; changing the label without changing the source fails. The sampler renders those originals inside the training function. It does not replace them with code-reconstructed audio.

In addition to code/parameter losses, a differentiable Era2 decoder compares predicted curves with the original curves sampled at 97 points. Those original values are **supervision**, retained separately from the model's input features. Missing curve labels fail rather than falling back to already-compressed targets. The labels are included in the bounded replay byte count, checkpointed, and verified against the exact source recipe on recovery.

Code supervision alone can penalize different codes for a similar curve. The reconstruction term provides a direct curve error alongside it. Its differentiable soft choice is a training approximation; exported patches use hard choices and the existing NumPy decoder. The native evaluator measures the resulting sound separately. No topology or oracle code is an inference input.

## Connections and activity

This stage declares sixteen fixed, ordered connections. LFO `i` feeds oscillator-1 level in slot `2*i-1` and sampler level in slot `2*i`; slots 17–64 remain empty. The authored curriculum activates one LFO per patch through the audible generator. Levels are scaled by 0.25 throughout this new curriculum to leave room for modulation. This is a declared distribution change; previous factory artifacts retain their contracts.

The numbered LFOs rotate within each frozen split's root order. Coverage does not require moving a state between splits. Unused curves receive no learning loss, and zero residual atoms do not create arbitrary phase/gain targets. Their stored codes survive serialization. LFO rates and the fixed routes' scalar controls use the existing full inventory. Other route structures, custom warps and changed seeds fail explicitly. Learning arbitrary ordered connections is the next capability step.

The bank's fitting manifest remains authoritative. Authored roots outside that population, or roots assigned to different splits, are refused. A larger or different population needs an explicitly fitted bank and new training lineage. The default run preserves all 256 original scalar roots, including the 24-state final reserve.

## Run locally

Use the qualified factory bank and a curve bank built with the corrected periodic decoder in the pinned Windows environment. The interpolation repair changes the shared decoder fingerprint, so banks from before this repair must be rebuilt with `run_lfo_bank_check.py`; the old files are refused by the source checks.

```powershell
python research/training/run.py lfo-pool --asset-bank research/training/outputs/factory-bank/bank.json --lfo-bank research/training/outputs/lfo-bank-check/bank.json --output research/training/outputs/lfo-curves --count 256
python research/training/run.py codec-check --pool research/training/outputs/lfo-curves/pool.json --output research/training/outputs/lfo-serialized-check
python research/training/run.py train --pool research/training/outputs/lfo-curves/pool.json --output research/training/outputs/lfo-training --steps 1000 --replay-capacity 64 --render-budget 128 --sampling-policy state_round_robin --cpu-threads 4 --worker-jobs 4
```

Bank creation is documented in [FACTORY_ASSETS.md](FACTORY_ASSETS.md) and [LFO_BANK.md](LFO_BANK.md). The source generator is checked against the bank's fitted curve hashes before the fixed integration. Generated banks, pools and models remain in ignored local outputs. No waveform archive is required.

The Python trainer also supports `stop_after=0` to collect its bounded warm-up features without optimizer updates. A same-pool checkpoint can seed a fresh model using `--seed-replay`; original source labels and the frozen frontend are rechecked. This does not allow reuse across changed bank/pool contracts. Exact interruption recovery separately restores model, optimizer and random-number state.

Raw curve encoding is lossy and need not be idempotent. The new `codec-check` therefore proves the declared **saved-target contract**: target serialization and codec reload preserve the decoded patch exactly, followed by exact native audio with oscillator and sampler random phases fixed. It does not relabel raw-curve projection as a lossless round trip. Earlier lossless scalar/asset checks keep their original behavior.

Run the fixed integration in a new directory:

```powershell
python research/training/reports/scripts/run_lfo_training_check.py --asset-bank research/training/outputs/factory-bank/bank.json --lfo-bank research/training/outputs/lfo-bank-check/bank.json --output research/training/outputs/lfo-training-check
```

It pins eight validation families, 1,000 CPU updates, 64 retained examples and native budgets of 24 saved-target checks, at most 128 training attempts and 96 validation attempts. Validation separately renders the original patch, its code reconstruction and the audio-only prediction. It measures original-to-code, original-to-prediction and code-to-prediction sound differences. The code reconstruction uses labels only for evaluation, after inference; it never enters the model input. Source-repeat calibration renders the literal original patch without applying the lossy encoder again.

The [integration report](reports/LFO_TRAINING_RESULTS.md) records the actual gates and resource use. Pipeline correctness, tiny-set learning, generalization and release qualification remain distinct. The implementation does not establish broad quality, learn all 775 controls, or complete arbitrary routing. Broader coverage/scaling, custom warps/seeds, performance variation and release work remain in the [completion ledger](IMPLEMENTATION_STATUS.md). Free local CPU is the qualified path; Linux/Colab and accelerator qualification remain open. GitHub Actions stays disabled.
