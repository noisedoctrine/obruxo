# Inspect saved input failures and prepare blind listening

**Historical original-level listening protocol.** The listener found severe loudness confounds in five comparisons. Tone-listening preparation is superseded by the [loudness-matched condition](LOUDNESS_MATCHED_LISTENING_CARD.md); retain the original audio and submitted responses as diagnostic history.

Use only the completed trainable-input fit's verified final predictions and retained native audio. Require its complete independent audit and unchanged plan/final-file identities. Add no renders, optimizer updates, model changes or confirmation scores. The 192 new validation roots stay reserved.

Describe unison/filter selections and source-active normalized control errors separately for original, held-note and alternate input recordings. Count each patch/input once across playback performances; reject different inferred patches for the same recording. Parameter errors are explanatory diagnostics, not reconstruction acceptance rules or proof that exact parameter recovery is necessary.

Prepare 12 listening trials, two from each of the six reconstruction groups. Within each group, average candidate-minus-control combined loss across each family's playback performances. Choose the family nearest the median gap, then the largest-gap remaining family, with stable identity tie breaks. Use the first declared playback for each selected family; do not select a more extreme waveform. This deliberate diagnostic selection is not a random confirmation sample.

For every trial, export the retained source, sound-trained candidate and matched parameter control as full-length stereo float32 WAVs. Check native file/receipt hashes and exact sample round trips. Reuse the existing blinded-listening tool to randomize A/B and trial order, remove identifying metadata and keep the answer key outside the listener directory. Do not normalize loudness, trim tails or generate preferences. Actual human judgments remain a separate required input.

Allow five minutes / 4 GiB process-tree RAM / 128 MiB output, zero native requests and zero experimental optimizer applications. Retain the selection, source hashes, packet identity and measured cost. Keep weights unqualified and report no listening result until real responses arrive.

```powershell
python -I research/training/reports/scripts/diagnose_basic_inputs.py --fit research/training/outputs/trainable-basic-sound-fit-20261026 --output research/training/outputs/basic-input-listening-20261027
```
