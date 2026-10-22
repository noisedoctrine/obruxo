# Blind local listening

The staged plan requires actual listener judgments after a promising model comparison. `reports/scripts/prepare_listening.py` prepares a local page with a reference and anonymous A/B reconstructions, then audits the returned judgments. It does not render audio, train a model, certify native QA or decide stage qualification. No listener results have been collected for the new spectral model yet.

Prepare inputs from a frozen, accepted native comparison or separately costed listening collection. Keep complete 44.1-kHz stereo float32 WAV observations at their original level; every reference/candidate/control triple must have the same frame count. Pin the candidate/control model identities, renderer profile, phase policy, case membership and listening acceptance rules in the experiment card before collecting judgments. The current A2 run discards candidate audit audio: this tool does not recover that missing audio or authorize extra native requests.

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

Include every planned family/performance pair once. `comparison`, `native_plan_id` and other experiment metadata remain in the private answer key. The utility verifies file hashes, finite sample values within [-1, 1], stereo format and paired lengths. It rejects altered or unsuitable input before creating the packet. It holds only one comparison's audio at a time. Anonymous WAVs strip container metadata while retaining every float32 sample, including level, stereo and the complete tail.

```powershell
python research/training/reports/scripts/prepare_listening.py prepare --manifest INPUTS.json --output NEW_PACKET_DIRECTORY
```

This creates `listen/index.html`, anonymous WAVs and a public packet manifest, with `answer-key.json` outside `listen/`. A new random 128-bit seed is recorded only in the private key; `--seed PRIVATE_SEED` reproduces a packet. Case order and A/B assignment are reproducible from that seed, and candidate placement is counterbalanced across A/B with at most one trial difference. Freeze the packet and answer-key hashes before listening. Give the listener **only the `listen/` folder**; keep the answer key, seed, source names, scores and model identities private until responses are saved. The listener opens `index.html`, compares the three clips at unchanged playback volume, answers which is closer and which is useful, and saves the JSON. Ties and “neither” remain valid observations. Reloading clears unsaved answers.

```powershell
python research/training/reports/scripts/prepare_listening.py summarize --key PACKET/answer-key.json --responses RETURNED_JUDGMENTS.json --output NEW_REPORT.json
```

The audit checks packet/assignment identity and all bundled page/audio hashes. Every planned trial must have exactly one complete response; missing, duplicated, foreign and invalid answers fail instead of disappearing from the denominator. It decodes A/B privately, reports all preference counts and usefulness judgments, and weights each independent family equally across its performances. Results are descriptive. The predeclared listening protocol, numeric confirmation, seed replication and remaining bundle checks still determine stage completion. Synthetic test responses must never be reported as listener evidence.

Five Python tests cover label hiding, exact sample preservation, deterministic assignment, input/hash rejection, family weighting, ties/neither, response completeness and page/audio/answer-key changes. Three Node tests execute the actual inline page code with local DOM/download boundaries: all comparisons and required questions appear, playback pauses other clips, and saved JSON retains every anonymous judgment. Run them with `node --test research/training/tests/test_listening_page.cjs`. CLI smoke, JavaScript syntax, Ruff and whitespace checks pass. Browser visual/audio-playback verification remains a manual limitation: this environment's browser policy rejects local-file navigation. The Node tests run code only; they do not open a browser or circumvent that restriction. No listener result or stage completion is claimed.
