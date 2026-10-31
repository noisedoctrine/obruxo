# Shared reconstruction comparison: execution record

The [three-sound comparison](SHARED_BASIC_AUXILIARY_CARD.md) started locally at **04:41 on 31 October 2026, Kuala Lumpur time** and is now complete and audited. Its [result](SHARED_BASIC_AUXILIARY_RESULTS.md) finds a 0.235% auxiliary improvement, below the fixed directional target. Both models fail reconstruction admission. The launch and software evidence below remain unchanged.

The implementation keeps the 14 outputs and existing native sound objective. Both versions use the same untrained initializer, update order, optimizer, inward control derivative, reference rehearsal and measured-sound acceptance. The added loss reads the complete timbre representation through a temporary ridge head. The proposed extra encoder screen was deferred.

**18 focused tests pass**, covering matched initial states, the identical first update, gradient reversal, continuous bounds, and exact recovery of both models, Adam, calibration and RNG. Ruff **0.16.2** passes; the contract file pins 0.12.2. GitHub Actions remains disabled.

The [real-initializer software check](data/shared_basic_auxiliary_software_check.json) also passes: 42 cold-export comparisons, exact simulated recovery, zero native requests and zero experimental fitting. It takes **8.45 seconds / 0.75 GiB**. Its coefficient is for the analytic software check only; native training independently measures and freezes its actual coefficient at update two.

The [execution declaration](data/shared_basic_auxiliary_declaration.json) records plan **83bcfff742c12f2f33b799676beb2168300b8810333c37e81d289533fe70a387**, source identities, schedule, renderer settings and resource bounds. Diagnostic probes use **125 patches** untouched by this run's sound training or rehearsal; 20 of the previous 145 diagnostic patches are excluded because they receive sound supervision. These are development diagnostics, and the baseline encoder had already seen this population during its earlier initialization.

The hidden process records its identity, resource heartbeat and terminal status under the local output directory. It checkpoints after every update, checks an exact replay after the first two native updates, and automatically replays the completed trajectory and rescores endpoint audio. The cap is **48 updates per arm, 15,360 native requests, ten hours and 4 GiB RAM**, plus a separate twenty-minute offline audit. Existing battery authorization applies.

At the [first-two-update checkpoint](data/shared_basic_auxiliary_prefix_verification.json), both models, Adam, calibration and RNG replay exactly with zero added renders. Both arms accept both updates. The actual native calibration is **0.026674**, with nonzero sound and auxiliary encoder gradients; the coefficient used at update two is 0.0066685 under the frozen ramp. This checks the live training path and recovery. The fit and endpoint comparisons were unfinished at that checkpoint; their completed result is linked above.

The completed result publishes all three sounds separately, the full comparison groups, earlier-32 preservation, refusals and the auxiliary effect on real sound. Full A2/A3, fresh confirmation, second seed, clean human listening and release remain unfinished.
