# Blind local listening

The staged plan requires actual listener judgments after a promising model comparison. `reports/scripts/prepare_listening.py` prepares reference/anonymous A/B comparisons and audits returned judgments. **Tone comparisons default to integrated loudness matching.** Original-level comparisons require an explicit level-accuracy purpose. The [first twelve diagnostic responses](reports/LOUDNESS_MATCHED_LISTENING_RESULTS.md) are preserved, but their tone preferences are loudness-confounded. No model is selected from them.

Prepare inputs from a frozen native comparison or separately costed collection. Supply complete 44.1-kHz stereo float32 WAV observations at their original level, with identical paired frame counts. Pin model/render/phase/case identities and acceptance rules before collecting judgments. The utility generates no synthesis or training and cannot recover discarded audio. The completed current fit retains its comparison waveforms; its training-probe waveforms remain discarded.

The input JSON uses paths relative to its own directory and byte hashes of each original WAV:

```json
{
  "format": "blind_pairwise_listening_inputs_v1",
  "comparison": {"candidate": "FROZEN_CANDIDATE_ID", "control": "FROZEN_CONTROL_ID"},
  "native_plan_id": "FROZEN_NATIVE_PLAN_ID",
  "cases": [{
    "family_id": "INDEPENDENT_FAMILY_ID", "role": "reference",
    "reference": {"path": "source.wav", "sha256": "FILE_HASH"},
    "candidate": {"path": "candidate.wav", "sha256": "FILE_HASH"},
    "control": {"path": "control.wav", "sha256": "FILE_HASH"}
  }]
}
```

Include every planned family/performance pair once. Experiment metadata remain in the private key. The utility verifies hashes, finite samples within [-1, 1], stereo format and aligned lengths before creating the packet. Tone matching needs at least 400 ms. Default matching uses FFmpeg's BS.1770/EBU R128 meter, one constant gain per clip toward -23 LUFS, and common attenuation of the triple if needed for -1-dBTP headroom. Require a maximum 0.1-LU spread after float32 conversion. Original files remain unchanged; envelopes, timing and stereo balance are preserved without compression or limiting. Missing metering fails instead of silently falling back to original levels.

```powershell
python -m pip install -r research/training/requirements-listening.txt
python -I research/training/reports/scripts/prepare_listening.py prepare --manifest INPUTS.json --output NEW_PACKET_DIRECTORY
python -I research/training/reports/scripts/verify_matched_listening.py --key NEW_PACKET_DIRECTORY/answer-key.json --manifest INPUTS.json --output NEW_METER_REPORT.json
```

The independent verifier uses pyloudnorm's DeMan BS.1770 implementation, checks meter agreement/spread and exact static-gain sample round trips. An existing FFmpeg binary can be supplied with `--ffmpeg`. Meter identity, original/processed LUFS, RMS, peaks and all gains stay in the private key. These review extras are not inverse-model runtime dependencies.

This is a controlled condition, not neutral normalization: peak matching equalizes maxima, RMS matching equalizes unweighted energy, and LUFS applies frequency weighting and quiet-block gating. Gain changes absolute peaks while preserving relative dynamics. Assess original level error separately; no preset, training target or raw sound-quality gate is gain-corrected by this listening transformation. To test level accuracy specifically, use `prepare --purpose level_accuracy`; its original samples are preserved and its questions concern reference level. Historical packets without a condition remain readable and are marked unmatched in new summaries.

The packet contains `listen/index.html`, anonymous WAVs and a public manifest, with the answer key outside `listen/`. A private random seed shuffles order and counterbalances A/B. Freeze packet/key identities and share **only `listen/`**. The page declares its condition, preloads clips, restarts on every play click, uses answer pills and saves browser drafts. Audio URLs contain waveform hashes to avoid stale cache reuse. A new condition receives a new identity and never inherits answers from the previous test. Ties/neither remain valid; the listener saves the completed JSON.

```powershell
python research/training/reports/scripts/prepare_listening.py summarize --key PACKET/answer-key.json --responses RETURNED_JUDGMENTS.json --output NEW_REPORT.json
```

The audit checks packet/assignment identity and all bundled page/audio hashes. Every planned trial must have exactly one complete response; missing, duplicated, foreign and invalid answers fail instead of disappearing from the denominator. It decodes A/B privately, reports all preference counts and usefulness judgments, and weights each independent family equally across its performances. Results are descriptive. The predeclared listening protocol, numeric confirmation, seed replication and remaining bundle checks still determine stage completion. Synthetic test responses must never be reported as listener evidence.

Local checks cover packet integrity, metered matching, static-gain/stereo preservation, peak headroom, explicit purposes, processing-identity tampering, draft recovery and instant replay. Install the review extras and set `OBRUXO_TEST_FFMPEG` for the real-meter tests; run page checks with `node --test research/training/tests/test_listening_page.cjs`. The current page is also verified through its localhost preview. [Protocol](reports/LOUDNESS_MATCHED_LISTENING_CARD.md) and [measured result](reports/LOUDNESS_MATCHED_LISTENING_RESULTS.md). A representative protocol, numeric confirmation and actual human responses remain required; synthetic responses never satisfy the listening gate.
