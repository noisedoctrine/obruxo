# Per-control training scope data

`stage_a_controls.jsonl` is the exact current Stage A inventory. It has one record per scalar Vital field and one record per structured LFO curve, modulation-route source/destination, modulation mapping slot, wavetable/custom-curve slot, and sample asset. Fixed and unmodeled controls are included.

Each record identifies the experiment/version, exact control key and Vital data path, control family, instance, whether the model can change it, head width, update method, training signal/gradient path, range or choices, default/fixed value, active mask, and reason when fixed. JSON Lines keeps the source diffable and can be loaded directly into SQLite using one row per JSON object.

Regenerate the Stage A snapshot from the pinned Vital schema, init preset, and current `basic_profile.py` with:

```powershell
python research/modelling/control_scope/build_inventory.py
```

This inventory describes the basic-codec scope; it is not evidence that the shared model has learned sound reconstruction. Add a separate experiment/version record set for each experiment that changes the control inventory or update path. Keep the report table grouped by capability family and reconcile its enabled/total counts to this file.
