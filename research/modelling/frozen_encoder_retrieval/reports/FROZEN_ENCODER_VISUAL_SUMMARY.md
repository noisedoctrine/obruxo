# Frozen audio encoders — visual summary

**Main finding:** CIR is strongest at **recognising the exact preset**. CLAP, DAC and EnCodec are competitive at **finding a different preset that sounds similar**; CLAP is substantially faster than DAC. Those are different strengths, not a single overall ranking.

**The three-MIDI test in one example:** Start with preset **A** playing MIDI 1. Each encoder searches alternative presets playing MIDI 2, selecting one, say **B**. Then compare **A and B playing MIDI 3**, unseen during selection. The closest possible catalog alternative and a random alternative provide reference points.

## 1. Is recognising a preset the same as finding a good substitute?

![Recognition accuracy against the sound error of alternative presets](figures/summary_recognition_vs_substitution.svg)

CIR finds the **same preset** most reliably (81.4% Recall@1). But CLAP and DAC mean select **different presets** with lower average sound error (5.15–5.16 versus 5.56 for CIR). CLAP's measured advantage over CIR is not conclusive on this sample.

Basic Pitch is omitted from all charts; MuQ-MuLan appears only in the first comparison. Subsequent charts compare five encoder families, and Pareto frontiers use only those five.

## 2. How do recognition and sound similarity trade off against inference time?

![Twin plots: Recall@1 and acoustic substitute error against encoder latency, with measured Pareto frontiers](figures/summary_speed_vs_substitution.svg)

**Left:** CIR gives the highest exact-preset recall; CLAP is much faster, with lower recall. **Right:** CLAP has both the lowest measured acoustic error and the fastest inference (56 ms). The dashed frontier marks nondominated choices among the five displayed families; the acoustic advantage is not conclusive.

## 3. Does matching loudness change which encoder looks best?

![Rank changes after matching the overall loudness of candidate recordings](figures/summary_loudness_ranking.svg)

Yes. CLAP leads with the recordings' original levels, while **DAC temporal and CIR lead after gain matching**. Level accuracy and timbral similarity are not identical.

## 4. Does preset sensitivity help recognition or sound similarity?

![Twin plots: Recall@1 and alternative-preset sound error against preset-versus-MIDI sensitivity](figures/summary_sensitivity_vs_recognition.svg)

**Left:** DAC mean has the strongest preset-versus-MIDI separation but far weaker exact-preset recall than CIR. **Right:** DAC's strong separation comes with competitive substitute-sound error. Sensitivity is informative, but not a substitute for measuring either outcome directly.

## 5. What are the recognition and sound-similarity tradeoffs in memory?

![Twin plots: Recall@1 and alternative-preset sound error against peak XPU memory, with measured Pareto frontiers](figures/summary_memory_vs_substitution.svg)

**Left:** CIR reaches the best recall at about **0.40 GiB** peak XPU allocation. **Right:** CLAP's lower measured substitute error comes at roughly twice CIR's memory (**0.79 GiB**); EnCodec uses less memory (**0.24 GiB**) with slightly higher acoustic error.

## Practical conclusion

**Use CIR as the reference for preset identification; keep CLAP as the fast catalog-substitution candidate.** DAC is informative for timbre analysis but expensive for direct inference. None of these tests demonstrates that a representation can predict a new playable Vital patch.

*Source data:* [full technical report](FROZEN_ENCODER_RETRIEVAL_RESULTS.md) · [substitution results](issue161_substitution_results.json) · [preset/MIDI sensitivity](issue157_sensitivity_results.json) · [standardized efficiency benchmark](model_efficiency_benchmark.json).

*Scope:* Seven frozen encoder families, 11 representations, 129 preset families and 50 MIDI performances. The substitute test uses 12 three-MIDI triplets; latency and memory use 128 matched recordings, batch 1, FP32 on Intel Arc XPU. Acoustic similarity is measured with MR-STFT rather than listening judgments. No model training or new rendering was done for this summary.
