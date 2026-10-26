# Inspect the saved learning mechanism before choosing another model change

Use only the completed trainable-input fit, its independent verification and exact frozen feature cache. Verify plan/probe/final file identities before reading them. No new audio, model inference, optimization or confirmation scoring is needed. Allow five minutes / 4 GiB process-tree RAM / 8 MiB evidence, with zero native requests and zero optimizer applications.

Check whether the strongest note in each held recording's saved Basic Pitch feature matches its known single source note. Average the existing eight time bins, with note offset 21 and three contour bins per semitone. This is a descriptive pooled-feature check, not a new note transcription or use of MIDI as an inverse-model input. Count primary update exposures separately for each patch/input type.

For each of twelve held and twelve alternate inputs, collect previously verified native probes measured under both its own and original performances. A candidate is eligible only when the exact same complete patch was measured under both. Reject inconsistent repeated costs, report incomplete candidates and keep every patch in the analysis. Compare the best joint candidate with the final sound model and matched parameter control. Also retain a final sound prediction if it beats all its earlier probes. This post-hoc best choice is an oracle diagnostic, not a deployable model or an independent quality result. Training probe waveforms were discarded and cannot be independently rescored again.

This distinguishes missing note evidence, limited exposure and failure to retain measured alternatives without asserting any single cause. It does not authorize more epochs, parameter-only initialization, wider outputs, fresh confirmation or a second seed. Use the result and actual listening feedback before selecting the next mechanism test.

```powershell
python -I research/training/reports/scripts/audit_basic_fit_mechanism.py --fit research/training/outputs/trainable-basic-sound-fit-20261026 --features research/training/outputs/mixed-basic-features-20261022/features.pt --output research/training/outputs/basic-fit-mechanism-20261027
```
