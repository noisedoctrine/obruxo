# Learning ordered modulation connections

The model can now predict the connection in each of Vital's 64 numbered slots, alongside the [eight LFO curves](LFO_TRAINING.md), [scalar controls](FULL_SCALAR_TRAINING.md) and [registered factory assets](FACTORY_ASSETS.md). Every slot retains its number. References such as `modulation_32_amount` remain literal; connections are never sorted or matched to different slots.

Each slot adds four categorical outputs: connection present, source, destination and mapping present. The full contract supplies 32 source names and 431 destination names. A separate presence output makes an empty connection atomic. The five existing indexed controls—amount, power, bipolar, stereo and bypass—continue through the full scalar outputs. Optional mapping curves add 97 continuous values per slot.

| Output | Size |
|---|---|
| Scalar controls, registered assets and eight LFO codes | 688 continuous values, 546 categorical heads |
| 64 mapping curves | 6,208 continuous values |
| 64 connections | 256 categorical heads |
| Total | 6,896 continuous values, 802 categorical heads |

The current network has 5,668,542 trainable parameters with the qualified banks. All 64 positions and the complete name vocabulary are represented. The initial authored curriculum uses five patterns: one direct connection, four parallel connections, an indexed amount reference, a pair of LFO frequency connections forming a feedback loop, and sixteen parallel connections. This establishes training integration for those patterns; it does not establish that every possible graph or destination has been learned.

## Source audio and learning

Training renders the original authored LFOs and three-point mapping curves during the training loop. It retains their sampled shapes as supervision alongside bounded audio-derived features. The model receives only timbre and frozen Basic Pitch features. Connection labels, original curves, source patches and slot identities are never inference inputs.

LFO outputs retain the fixed train-only Era2 bank. Mapping curves use direct 97-point values, with a presence flag distinguishing an absent mapping from an explicit curve. Their original smoothness, point count, name and power parameters are approximated by a canonical 97-point linear curve. Saved targets reload exactly; encoding arbitrary raw curves remains lossy. Native evaluation separately measures the sound of the original patch, its representation and the prediction.

The presence loss gives connected and empty slots equal total weight per example. Empty slots therefore cannot dominate simply because they are numerous. Structure and curve losses apply to connections that can reach an audible destination. The activity calculation follows literal amount references and LFO feedback; bypassed or disconnected paths are masked. Stored inactive choices survive in saved targets. This graph analysis is a conservative training mask, not a proof of every native signal dependency.

The authored curriculum rotates slot positions within each frozen split, keeps all original root assignments, and uses up to sixteen active connections. Its master volume is stored as `3844` (the documented native display transform gives -18 dB), leaving headroom before rendering. This is a declared source-distribution change; rendered PCM is not normalized to conceal clipping.

## Native boundary

The full native check covers every source name, every destination name, all 64 positions with mapping and indexed controls, and a dense 64-slot state. Self-amount references, LFO feedback and amount cycles have dedicated repeated fixtures. Oscillator and sampler random phases are fixed for correctness checks; musical curve phases remain intact.

Vital 1.6.4 drops a mapping attached to an empty connection. The raw encoder explicitly refuses that input. Decoding emits a mapping only when the slot is connected; latent mapping predictions for an empty slot remain stored but are not emitted or supervised. The check records the empty-slot example as an expected state-verification rejection. It does not weaken the renderer's exact nested-state verification.

Vocabulary acceptance proves that the requested state loads and valid audio is returned. Some source types require performances not exercised here, and some destinations are inactive in these fixtures. This check does not certify audible effectiveness for every source/destination or all possible feedback graphs. [The results report](reports/ORDERED_ROUTING_RESULTS.md) records the actual outcomes and resource use.

## Run locally

Use the already qualified factory and curve banks, including the corrected periodic curve decoder:

```powershell
python research/training/run.py routing-pool --asset-bank research/training/outputs/factory-bank/bank.json --lfo-bank research/training/outputs/lfo-bank-check/bank.json --output research/training/outputs/routing --count 256
python research/training/reports/scripts/run_routing_check.py --output research/training/outputs/routing-native
python research/training/reports/scripts/run_routing_training_check.py --asset-bank research/training/outputs/factory-bank/bank.json --lfo-bank research/training/outputs/lfo-bank-check/bank.json --output research/training/outputs/routing-training-check
```

`--probe` limits the native check to eight attempts for the four edge cases. The fixed integration plans 400 CPU updates, 64 retained examples within 8 MiB, at most 128 training renders, fifteen saved-target correctness renders, and sixty validation renders over five connection families and two performances. It is an integration check with no model promotion. Separate map labels count toward replay memory and are reverified against source recipes on recovery.

Free local CPU is the qualified execution path; Linux/Colab remains open. GitHub Actions stays disabled. The final-test reserve is untouched. Custom warps, random seeds, broader combined coverage and release qualification remain in the [completion ledger](IMPLEMENTATION_STATUS.md).
