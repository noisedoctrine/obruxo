# Custom oscillator curves and random seeds

The full nested-state model extends [ordered connections](ORDERED_ROUTING.md) with three custom oscillator curves and three exact seed values. The custom curves affect the installed Vital 1.6.4 spectral modes 13, 14 and 15. The seeds affect oscillator random-amplitude morphing in mode 6; the four `random_i` modulation sources remain separate controls. These associations were measured with fixed-phase native comparisons in every oscillator slot.

| Outputs | Size |
|---|---|
| Scalar controls, registered assets, eight LFOs and 64 connections/maps | 6,896 continuous values, 802 categorical heads |
| Three custom curves | 291 continuous values: 97 per oscillator |
| Three seeds | 96 binary categorical heads: 32 per oscillator |
| Total | 7,187 continuous values, 898 categorical heads |

With the qualified asset banks, the model has 5,730,849 trainable parameters. Seeds are assembled from 32 binary outputs as unsigned integers in `[0, 4294967295]`. Float32 regression cannot preserve every integer in that range. Bit accuracy is a state metric; nearby numeric seeds do not imply similar sound, and this representation alone does not establish seed predictability or sound quality.

## Original sources and approximation

Training renders the original authored custom curves, LFOs, connection maps and seed values on demand. Original curve samples remain separate supervision tensors. Only timbre and frozen Basic Pitch features enter the model. The custom curves have direct sampled-value loss alongside existing parameter, LFO and mapping losses. Missing or altered original labels fail recovery checks; all label tensors count toward bounded replay memory.

Each custom curve decodes to 97 float32 points with a canonical name, zero segment powers and linear interpolation. This approximates original point counts, powers and smoothness. Saved targets reload exactly; arbitrary raw-curve reencoding is lossy. The evaluator renders both the original and the representation to measure that approximation separately from prediction error. The current encoder supports two to 100 points through the shared curve parser; one-point records fail explicitly and remain a coverage gap.

Unused curves and seed bits retain their saved values but receive no learning loss. Curves are active when their oscillator is enabled in modes 13–15; seeds are active in mode 6. Native checks verify that changing another oscillator's curve or seed leaves the isolated active oscillator's audio unchanged. Oscillator and sampler random phases are fixed during correctness checks. Musical seed values are preserved.

The initial curriculum has thirteen families: each of three custom-curve modes in each oscillator, each oscillator's seed mode, and one combined case. All three oscillators use a registered Basic Shapes asset (the structural test fixture uses its first authored registered wave), level `0.06` and master volume `3844`. Custom curves have three to six original points, varying powers and smoothness; original seeds span all 32 bits. Existing ordered-connection patterns remain present. The source distribution is explicit and preserves all frozen root assignments; it does not claim broad coverage of every scalar control or graph.

## Exact seed bits in the native profile

Vital accepts an unsigned seed such as `4294967295`, then saves it as signed `-1`. Both encode the same 32 bits. A direct native check confirmed identical repeated audio and readback for signed/unsigned pairs, including non-silent cases.

The new `full-1.5.5-state-verified-seed-bits-v2` renderer profile converts only negative integer readback values in the signed 32-bit range back to the corresponding unsigned bits, then requires an exact match. Incorrect bits, other nested changes and scalar changes still fail. Its renderer, request and training-contract identities differ from the older full profile. Old codecs and bank qualifications keep their previous identities and exact readback behavior; the new model uses the new profile explicitly. The registered factory payloads are unchanged, and the new native check exercises them under the new profile.

Several legal seeds yield exact silence in the isolated Basic Shapes fixture at frame 128, mode 6 and amount 0.6. The native check records these as valid loaded state with rejected audio. Silence remains excluded from training; thresholds are unchanged. This observation is fixture-specific, not a claim about every waveform or morph amount.

## Run locally

Use the qualified factory bank and corrected periodic LFO bank:

```powershell
python research/training/run.py nested-pool --asset-bank research/training/outputs/factory-bank/bank.json --lfo-bank research/training/outputs/lfo-bank-check/bank.json --output research/training/outputs/nested --count 256
python research/training/reports/scripts/run_nested_check.py --asset-bank research/training/outputs/factory-bank/bank.json --output research/training/outputs/nested-native
python research/training/reports/scripts/confirm_seed_bit_alias.py --asset-bank research/training/outputs/factory-bank/bank.json --output research/training/outputs/seed-bit-alias
python research/training/reports/scripts/run_nested_training_check.py --asset-bank research/training/outputs/factory-bank/bank.json --lfo-bank research/training/outputs/lfo-bank-check/bank.json --native-qualification research/training/outputs/nested-native --output research/training/outputs/nested-training
```

The fixed integration verifies the completed native qualification against its source hashes, asset bank and renderer identity before training. It plans 400 CPU updates with 64 retained examples within 8 MiB, at most 128 training renders, 39 saved-target correctness renders and 156 validation renders across thirteen states and two performances. It never opens the final-test reserve and does not promote a model.

The [results report](reports/CUSTOM_CURVES_AND_SEEDS_RESULTS.md) records native state acceptance, audible effects, approximation error, learning, failures and resources. Free local CPU remains the qualified path. Linux/Colab, broader combined control/performance coverage, useful model quality and release gates remain open in the [completion ledger](IMPLEMENTATION_STATUS.md). GitHub Actions stays disabled.
