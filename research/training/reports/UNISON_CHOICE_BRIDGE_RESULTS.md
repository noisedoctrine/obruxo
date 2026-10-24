# Independent unison correction: initialization and recovery pass

Completed 2026-10-24 under the [frozen card](UNISON_CHOICE_BRIDGE_CARD.md), plan `e598c1b72414faf7d639fde3bf5c80c9d4d4015c4487f46ebde367aff0955a2a`. This qualifies an untrained correction and its software recovery path. It is not a reconstruction-quality result, a completed fitting check or model admission.

The [bound summary](data/unison_choice_bridge_summary.json) and [per-input identities](data/unison_choice_bridge_identity.jsonl) establish:

- All **1,616 scheduling entries / 1,212 distinct recordings / 404 training patches** retain their starting hard choices, preset bytes and cold restored outputs. No development/test input is used.
- Unison soft probabilities exactly match the prior calibrated predictor at temperatures 1.0 and 0.5. Raw unison score units intentionally change; no raw-score identity is claimed.
- All nine continuous outputs and both filter scores remain bit-identical to the frozen predictor, including after simulated correction learning and cold restoration on the entire cache.
- At all **24 cached native points**, measured category costs reach the direct unison correction and match the analytical softmax derivative. Frozen predictor gradients remain absent; initial weights and Torch RNG stay unchanged by these gradient checks.
- The first backward pass reaches the zero-initialized output layer. The subsequent simulated step reaches the hidden layer too. A checkpoint after step one reproduces step two's model, Adam, gradients/loss and Python/NumPy/Torch RNG exactly; the exported simulated model reloads identically.

The model has **382,561 parameters**: 355,806 frozen and 26,755 trainable. It retains **14 output values** and the same 11 codec selections. Only the three unison scores can change. Fixed phases and the qualified 32,768-frame lead survive export. The earlier model, codec and renderer modules remain unchanged.

Qualification uses **113.21 seconds / 0.73 GiB** process-tree RAM, within five minutes / 4 GiB; evidence stays below 64 MiB. [Preparation](data/unison_choice_bridge_preparation.json) takes 0.55 seconds / 0.47 GiB. The [execution service](data/unison_choice_bridge_service.json) exits successfully with zero RSS sampling errors and releases its scoped active-work request. Twelve local checks, Ruff and diff checks pass; implementation `8083e9a` and its [numerical plan](data/unison_choice_bridge_plan.json) were frozen before execution.

There are **zero native requests and zero experimental training updates**. Recovery verification applies **three explicitly counted simulated optimizer steps**: two sequential steps and one replay, using a retained measured three-choice cost vector. This verifies the connection and recovery; it does not show that training across recordings improves sound.

The untrained artifact is `model.pt`, file SHA-256 `dcc17b0203684b513467e4d2b85d97b59db40f397425a063b7ac7449d7882eaf`, tensor SHA-256 `c7011f3a55fbc7fb4b577d9fdad9b71632766e26dca0969bd70311da5318d66a`. It uses the explicit `restore_unison` wrapper; this bridge does not claim release CLI/bundle support. The simulator artifacts are separate and are not fitting initializers.

Next, implement and freeze the [small unison sound-cost fit](UNISON_CHOICE_FIT_CARD.md). It must establish useful category decisions against retained actual sound costs and fresh final renders while preserving original reconstruction. All full A2/A3 reconstruction gates remain required.
