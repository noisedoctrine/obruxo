# Frozen audio encoders — visual summary

**Main finding:** CIR is strongest at **recognising the exact preset**. CLAP, DAC and EnCodec are competitive at **finding a different preset that sounds similar**; CLAP is substantially faster than DAC. Those are different strengths, not a single overall ranking.

**The three-MIDI test in one example:** Start with preset **A** playing MIDI 1. Each encoder searches alternative presets playing MIDI 2, selecting one, say **B**. Then compare **A and B playing MIDI 3**, unseen during selection. The closest possible catalog alternative and a random alternative provide reference points.

## 1. Is recognising a preset the same as finding a good substitute?

![Recognition accuracy against the sound error of alternative presets](figures/summary_recognition_vs_substitution.svg)

CIR finds the **same preset** most reliably (81.4% Recall@1). But CLAP and DAC mean select **different presets** with lower average sound error (5.15–5.16 versus 5.56 for CIR). CLAP's measured advantage over CIR is not conclusive on this sample.

## 2. What does acoustic similarity cost in inference time?

![Speed versus alternative-preset sound error](figures/summary_speed_vs_substitution.svg)

CLAP combines low measured sound error with **56 ms** median extraction time. CIR takes **289 ms** and DAC **1,466 ms** on the same benchmark recordings.

## 3. Does matching loudness change which encoder looks best?

![Rank changes after matching the overall loudness of candidate recordings](figures/summary_loudness_ranking.svg)

Yes. CLAP leads with the recordings' original levels, while **DAC temporal and CIR lead after gain matching**. Level accuracy and timbral similarity are not identical.

## 4. Does sensitivity to preset changes guarantee recognition?

![Preset-versus-MIDI separation plotted against exact-preset identification](figures/summary_sensitivity_vs_recognition.svg)

No. DAC mean has the **largest normalized preset-versus-MIDI separation**, yet CIR identifies the correct preset much more often. Detecting that something changed is different from recognising what it is.

## 5. What is the memory tradeoff?

![Peak XPU memory usage versus alternative-preset sound error](figures/summary_memory_vs_substitution.svg)

CIR uses about **0.40 GiB** peak XPU allocation, CLAP **0.79 GiB** and DAC **0.92 GiB**. MuQ-MuLan uses **2.68 GiB** but has a worse substitute score on this test.

## Practical conclusion

**Use CIR as the reference for preset identification; keep CLAP as the fast catalog-substitution candidate.** DAC is informative for timbre analysis but expensive for direct inference. None of these tests demonstrates that a representation can predict a new playable Vital patch.

*Source data:* [full technical report](FROZEN_ENCODER_RETRIEVAL_RESULTS.md) · [substitution results](issue161_substitution_results.json) · [preset/MIDI sensitivity](issue157_sensitivity_results.json) · [standardized efficiency benchmark](model_efficiency_benchmark.json).

*Scope:* Seven frozen encoder families, 11 representations, 129 preset families and 50 MIDI performances. The substitute test uses 12 three-MIDI triplets; latency and memory use 128 matched recordings, batch 1, FP32 on Intel Arc XPU. Acoustic similarity is measured with MR-STFT rather than listening judgments. No model training or new rendering was done for this summary.
