# Train-only LFO reconstruction assets

`lfo-fit` rebuilds the existing Era2 curve representation under the inverse model's frozen split manifest. It accepts registered training states only. Validation/test input, unknown identities and duplicate preset states fail before dictionary construction begins. It samples all eight numbered LFO slots, deduplicates sampled shapes, and selects a bounded, deterministic set by curve hash. Input-file order cannot change the fitted bank.

The fixed representation uses the existing W8D16 phase-and-residual-gain policy: 32 base atoms, 16 residual layers with eight atoms each, 17 phases and 16 residual gains. That is 193 model outputs per slot, or 1,544 for eight slots. Each residual layer keeps a zero NoOp atom. Phase wraps in `[0,1)`; gains lie in `[-2,2]`; reconstruction clips only the final curve to `[0,1]`. The grid remains 97 inclusive points. The selected existing Beam4/BestOverallRepair policy is reused without a new search over representations.

```powershell
python research/training/run.py lfo-fit --presets path/to/train-a.vital path/to/train-b.vital --split-manifest path/to/splits.json --output research/training/outputs/lfo-bank --maximum-curves 256
```

`splits.json` is the frozen split manifest, including inherited variant assignments when applicable. Each supplied file must already be registered as a training state. This command does not relabel or silently exclude held-out files. The output directory must be new. A fit is bounded to 4,096 source presets and at most 4,096 unique curves; the ordinary default selects at most 256 curves. Shape parsing currently requires at least two control points; a one-point line fails explicitly. No audio is needed to fit the dictionaries.

The saved bank includes the full split manifest, selected state/lineage/slot identities, source and curve hashes, policies, source-code hashes and fitted arrays. Its ID covers that content. Source hashes normalize line endings and use portable relative paths. Loading verifies the current Era2 implementation, policy, dimensions, finite values, zero atoms and training-only membership. Execution timing is excluded. Runtime source files from the repository are still required; a separately packaged release runtime remains open work.

The connected training work fixes a periodic interpolation boundary in the shared curve helper. Banks made before that repair have a different decoder fingerprint and must be rebuilt; the loader refuses them. The newer [training report](reports/LFO_TRAINING_RESULTS.md) records qualification under the corrected implementation.

```python
import json
from pathlib import Path
from obruxo_training.lfo_bank import LfoBank, LfoCode, curve_to_lfo

bank = LfoBank(json.loads(Path("research/training/outputs/lfo-bank/bank.json").read_text()))
code = LfoCode.from_dict(saved_or_predicted_code)
curve = bank.decode([code])[0]
vital_shape = curve_to_lfo(curve)
```

Decoding takes the saved bank and legal codes. It has no source curve, topology label, preset identity or MIDI input. `curve_to_lfo` exports a legal 97-point straight-segment Vital shape with float32 coordinates. Encoding a known curve is an offline target-construction operation; its error is a representation ceiling, not evidence of predicting codes from audio.

The [bounded native report](reports/LFO_BANK_RESULTS.md) separates original-to-grid sound changes from grid-to-code changes and checks all eight slots under two performances with fixed oscillator and sampler phases. Reproduce it with:

```powershell
python research/training/reports/scripts/run_lfo_bank_check.py --output research/training/outputs/lfo-bank-check
```

The diagnostic retains the original 256 scalar roots and all split assignments. It fits at most 64 curves from 205 training presets, selects an eight-case train/validation panel before fitting, and caps native work at 80 attempts. No final-test audio, waveform archive, paid compute or Actions run is involved. It measures correctness and reconstruction effects; no perceptual threshold is calibrated here.

The [eight-curve training path](LFO_TRAINING.md) connects these outputs, reconstruction supervision and activity masks to original-audio training. Custom warps, random seeds and learned arbitrary modulation routing remain separate capability work. The [completion ledger](IMPLEMENTATION_STATUS.md) tracks those requirements.
