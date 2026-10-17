# Match performance length to amplitude attack

The opt-in `--sampling-policy attack_compatible` visits every selected training state once per epoch and rotates through its compatible reference performances. It requires at least one note, including an already engaged sustain pedal, to last through the stored amplitude delay and attack. Vital's stored envelope values are converted to seconds using their fourth power. Tempo changes and canonical MIDI event order are respected. A pedal pressed after note-off cannot extend that note.

This changes the observation schedule. Source presets, targets, controls, gain, audio QA and split assignments stay unchanged. The sampler has a separate identity that includes the rule's source fingerprint; incompatible saved schedules fail recovery checks. Earlier sampling policies and their identities retain their behavior. A state with no compatible performance fails explicitly. The policy requires the complete nested-state codecs; it does not guess missing envelope controls for the narrow pilot codec.

Compatibility means enough time to observe the attack, not guaranteed audibility or identification of every control. A long note can still be silent because of filtering or modulation. Native QA remains mandatory before a sample enters training. Existing short-note performances remain available through the other policies; this policy should not replace explicit evaluation of transients or transfer performances.

## Why this is needed

The combined pipeline accepted 64 of 67 source attempts. The three rejected attempts are retained in the [original report](reports/COMBINED_CURRICULUM_RESULTS.md). Their causes were inspected before changing the sampling policy:

| Original sequence | Cause | Treatment |
|---|---|---|
| 6 | A strong oscillator high-pass setting leaves no audible signal in the selected register | Preserve the setting and retain the silent view as an expected audio-QA rejection |
| 15 | A strong oscillator high-pass setting leaves no audible signal in the selected excitation | Preserve the setting and retain the silent view as an expected audio-QA rejection |
| 45 | A sampler patch has about 0.143 seconds of amplitude delay plus attack, but its notes last 0.125 seconds | Choose a longer reference performance of the identical patch |

For the first two sources, disabling spectral processing or reducing its amount restores signal. Longer held notes alone remain silent. For the third, changing oscillator spectral controls has no effect because the oscillators are off; a longer held note of the unchanged sampler patch restores audible output. These are valid source/performance interactions. They do not justify deleting high-pass synthesis, shortening the envelope, raising the silence threshold or excluding these state IDs from the sampler.

The portable [native regression runner](reports/scripts/run_source_eligibility_check.py) repeats the three original negative views and the sampler-selected longer view. All eight gates pass in the [completed result](reports/SOURCE_VIEW_RESULTS.md). Correctness requests fix the four generator random phases and use a fresh process per request, retaining the patch's musical phase controls, seeds and random modulation. Original labels and the unchanged preset between the short and longer views are checked separately. The complete training epoch is checked for state coverage, compatible reference views and split isolation. Only metadata is published; audio is rendered in memory.

```powershell
python research/training/reports/scripts/run_source_eligibility_check.py --pool research/training/outputs/combined-curriculum-headroom-pool-20261017/pool.json --output research/training/outputs/source-view-check
python research/training/run.py train --help
```

Use the existing training entry point with `--sampling-policy attack_compatible` and the frozen combined pool. The native runner deliberately requires that original pool so the negative fixtures cannot silently change. This panel qualifies source-view behavior and recovery, not reconstruction quality. Broader source robustness, graph/range coverage, input layouts, data scaling and the model's quality gates remain open. Final-test audio stays reserved and GitHub Actions stays disabled.
