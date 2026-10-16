# Full scalar training integration

Completed 2026-10-16 on the local CPU. The pipeline now trains, exports, restores and natively evaluates a model with outputs for all **775 scalar fields** in the reviewed full preset inventory. The fixed integration passed all eight pipeline gates. The six nested structures remain fixed, and the authored curriculum varies 25 scalar controls; this is not evidence that all 775 controls have been learned.

| Measurement | Result |
|---|---|
| Model | 803,823 trainable parameters; audio input only; frozen Basic Pitch |
| Output budget | 424 continuous outputs, 406 categorical heads, 3,063 categorical logits |
| Authored population | 256 states: 205 train, 27 validation, 24 reserved test |
| Learning run | 400 CPU updates, batch 16, 32 retained training examples |
| Training native attempts | 34 attempted, 32 accepted, two clipping rejections, zero errors |
| Retained features | 1,237,120 bytes; no stored audio corpus |
| Training/collection time | 210.37 seconds, including 7.79 seconds of feature extraction |
| Peak training/process memory | 896.93 MiB, below 8 GiB |
| Fixed-phase codec qualification | 15/15 renders accepted; all 775 controls equal at float32; exact repeated/decoded audio |
| Enabled-effect order equivalence | 2/2 renders accepted; exact fixed-phase audio |
| Native validation | 50/50 renders accepted across five validation families and reference/transfer performances |
| Inference artifact | Restored weights and codec identity match the trained checkpoint |

The training losses measured over the complete 32-example replay changed as follows:

| Metric | Initial | Final |
|---|---:|---:|
| Combined loss | 0.253044 | 0.002311 |
| Family-averaged continuous MAE | 0.035448 | 0.007879 |
| Categorical accuracy | 96.2343% | 99.8954% |

The strict tiny-overfit gate remains **false**, because it requires 100% categorical accuracy. Loss reduction and continuous-MAE thresholds passed. High aggregate accuracy includes many fixed controls and does not establish broad control recovery. No model is promoted and no weight/seed sweep was added to force this gate through.

The five-state native panel is an integration diagnostic, with no claim of a reliable generalization gain:

| Method | Reference mean spectral distance | Reserved-transfer mean spectral distance |
|---|---:|---:|
| Model | 1.01318 | 1.03856 |
| Training-only retrieval | 1.31101 | 1.48691 |
| Fixed template | 7.61532 | 8.84459 |
| Source repeat | 0 | 0 |

All ten model/performance scores are complete. Only 32 training states supplied retrieval candidates. The [plan](data/full_scalar_plan.json) pins model/training configuration, source hashes, frozen lineages, validation cases and attempt budgets. The [summary](data/full_scalar_summary.json) records training/resource metrics, varied controls, all gates and the rejected prototype. The [codec report](data/full_scalar_codec_summary.json) contains each fixed-phase comparison.

The first prototype accepted 40/50 validation attempts and rejected all ten model renders on a strict scalar readback check: a requested reverb gain of `-0.0006000995635986328` returned as `-0.0005998611450195312`. The repair serializes controls on the plugin's float32 grid. It leaves the native readback threshold unchanged, and the final run accepts all fifty validation attempts. The earlier failed run remains recorded rather than being counted as a pass.

That repair changed all 256 serialized state hashes. Their complete original root manifest is preserved exactly: every corrected state inherits its original split and lineage, including all 24 reserved test states. Pool loading reconstructs the expected lineage mapping and rejects reparenting or omitted parents. Checkpoint recovery separately reproduces final model weights, optimizer state and Torch/minibatch RNG exactly.

Local software verification covers 224 distinct checks. The complete training-suite run had 219 passes and three failures from CLI test stubs that did not accept the added renderer-profile argument. After updating those stubs, all three focused reruns passed. The separate modern CLI/profile and exact optimizer/RNG recovery checks also passed. Counts combine the complete run and focused checks. Ruff and branch whitespace checks passed; the native qualification source hashes still match.

The two rejected training renders had clipping peaks of 1.018 and 1.248. Both attempts remain charged in the stream history and cursor; the fixed 64-attempt training budget allowed collection to continue to 32 accepted examples. Validation failures would fail the integration gate. Clipping/readback tolerances were not weakened, and final-test audio was never rendered.

Broader asset/curve/routing outputs, wider authored/corpus coverage, performance variation, data scaling and native quality selection remain open. Existing pilot/alpha contracts stay unchanged. GitHub Actions remains disabled, and no paid compute was used.

Reproduction commands and representation details are in [FULL_SCALAR_TRAINING.md](../FULL_SCALAR_TRAINING.md).
