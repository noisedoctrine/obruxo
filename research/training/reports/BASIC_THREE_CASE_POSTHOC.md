# Post-fit look at the three direct-control sounds

The completed 12-patch [shared sound fit](TRAINABLE_BASIC_FIT_RESULTS.md) already contained the three patches from the [direct-control check](DIRECT_BASIC_CONTROL_RESULTS.md). This note records a descriptive slice of its saved final comparisons to help plan the next focused shared fit.

## Selection

From the saved `fit_audit_cells`, select training indices **2, 8 and 11**, input view **1** (the held-note recording), and playback views **0 and 1**. These are the six held-input comparisons for the MIDI 36, 61 and 83 source patches. Average the two playback scores within each patch; report the three patches separately. The final column compares the sound-trained arm with the matched parameter-trained arm.

The same saved plan has **21 audit cells across nine unique input recordings** for these three patches. The table above deliberately shows only the six held-input rows; a later focused fit should keep its full per-patch results visible.

| Training index / MIDI | Parameter MRSTFT | Sound MRSTFT | Change | Parameter spectral | Sound spectral | Change |
|---|---:|---:|---:|---:|---:|---:|
| 2 / 36 | 0.9825 | 1.3909 | +41.6% | 0.2691 | 0.5329 | +98.0% |
| 8 / 61 | 1.1395 | 1.1521 | +1.1% | 0.4686 | 0.4747 | +1.3% |
| 11 / 83 | 1.7921 | 1.8184 | +1.5% | 0.7203 | 0.7246 | +0.6% |
| All six rows, equal weight | 1.3047 | 1.4538 | +11.4% | 0.4860 | 0.5774 | +18.8% |

Positive changes mean higher error for the sound-trained arm. The MIDI 36 result accounts for most of the gap; the other two are close to the parameter control.

## Limit

These comparisons were selected after the 12-patch fit and use sounds that were already in training. They are not a predeclared test, independent validation, or a new pass/fail gate. They do not establish why the low-note result is worse. Keep all three cases visible in the next shared fit and do not let their mean hide a failing sound.

The saved source is `outputs/trainable-basic-sound-fit-20261026/plan.json` and `finals.jsonl`; its plan ID is `efb0885f7e0763d72fd82202798edba0fd312bdec7e5367cfa28969abcb2d5f0`. Their SHA-256 hashes are recorded in the machine-readable note to identify the exact inputs used.

The [machine-readable row values and calculation](data/basic_three_case_posthoc.json) are included with this note so the averages can be checked without the local ignored run outputs.
