# Separate loudness error from sound differences in the completed fit

The complete saved panel contains **104 observations and 312 prediction recordings**. All six training groups retain twelve independent patches; the earlier panel retains all 32, including twelve patches also present in the training groups. Each patch receives equal weight within its group. Original scores come from the independently rescored fit. The new analysis changes audio gain for measurement only, leaving presets, original recordings and model-quality gates unchanged.

The shared sound-trained model still trails its matched settings-trained control on both held-note input groups after integrated-LUFS matching. Loudness is a substantial additional error, rather than the sole explanation for that failure. Matching also preserves envelope, timing and stereo differences; the resulting numbers are not listener judgments.

| Group | Original combined error vs control | Level-matched combined error vs control | Level-matched MRSTFT vs control |
|---|---:|---:|---:|
| Original input, same performance | -41.97% | -39.58% | -38.46% |
| Original input, other performance | -40.16% | -37.43% | -38.30% |
| Held-note input, same performance | **+8.06%** | **+7.30%** | **+5.18%** |
| Held-note input, original performance | **+11.08%** | **+9.50%** | **+7.45%** |
| Alternate input, same performance | -13.80% | -8.08% | -1.63% |
| Alternate input, original performance | -3.80% | -0.70% | +4.73% |
| Earlier 32 patches | -26.47% | -25.42% | -24.83% |

Negative means lower error. This table compares the two arms of the current fit. The separate earlier-32 preservation gate compares against the stronger prior sound model and retains its original result. For held-note inputs, original absolute loudness errors are **3.37 LU** on the same performance and **3.48 LU** on the original performance, versus 3.06/2.84 LU for the settings-trained control.

All four recordings in each observation share one loudness target and peak-headroom allowance. Each receives one constant gain applied directly to its original samples. No compression or limiting is used. Shared targets range from -23 to -26.816 LUFS. The maximum independently measured spread is **0.00738 LU**; maximum FFmpeg/pyloudnorm disagreement is **0.00926 LU**, below the unchanged 0.1/0.2-LU tolerances.

The first attempt stops after 97 cases because amplification changes the absolute gate for a quiet initial prediction: one-pass matching leaves it 0.480 LU below the shared target. The bounded correction solves the final constant gains in at most four passes; only that case needs a second pass. An intermediate processing check retained only twenty non-overlapping reference patches in its group summary. The final implementation explicitly retains the twelve overlapping training/reference patches, checks every expected group denominator, and preserves both earlier outputs. [Attempt history](data/basic_level_diagnosis_attempts.json) and [per-case measurements](data/basic_level_diagnosis_cases.jsonl).

Implementation `7f4e166` precedes the [final frozen analysis](data/basic_level_diagnosis_plan.json). Seven software tests, Ruff and diff checks pass. The [complete result](data/basic_level_diagnosis_summary.json) takes **148.90 seconds / 0.61 GiB RAM / approximately 0.4 MiB evidence**; all three analysis attempts total 498.71 execution seconds. Every source/prediction hash is checked again after processing. There are **zero native renders, inference calls or optimizer applications**, and no fresh confirmation or human preference evidence. The shared model remains unqualified. [Protocol](BASIC_LEVEL_DIAGNOSIS_CARD.md).
