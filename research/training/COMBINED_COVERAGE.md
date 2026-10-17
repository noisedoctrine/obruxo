# Combined training coverage

The model has outputs for all 775 scalar controls and all six nested state families. This does not mean the training recipes teach every capability. The [coverage inventory](reports/data/combined_capability_coverage.json) counts variation in the complete frozen expressive recipe pool, using training states only. It separates controls masked out of supervision, controls with one supervised value, and controls with multiple supervised values.

These are recipe and supervision counts. A mask is the codec's declared eligibility rule; it does not prove that changing the control affects the sound, that its full range is covered, or that the model learns it. The audit makes zero render requests and zero optimizer updates. Native acceptance and useful reconstruction require separate checks. Validation/test recipes remain excluded from the coverage totals, and their audio stays reserved.

## Current recipe matrix

The 2026-10-17 audit covers all 205 training states in the frozen 256-state expression pool. The other 27 validation and 24 test states do not contribute. The earlier 32-state integration prefix and its root/split assignments are preserved. Across all scalar controls, 214 have multiple supervised values, 374 have one supervised value and 187 are masked throughout this pool. Of the varied controls, 192 are connection settings; 22 belong to the other families. Curves, assets, seeds and connection endpoints are counted separately below.

| Scalar family | Varied under supervision | Constant under supervision | Always masked |
|---|---:|---:|---:|
| Oscillators | 7 | 83 | 0 |
| Sampler | 0 | 1 | 10 |
| Filters | 3 | 14 | 15 |
| Envelopes | 4 | 5 | 45 |
| LFO/random playback controls | 0 | 96 | 32 |
| Connection settings | 192 | 128 | 0 |
| Effects/order | 4 | 24 | 85 |
| Voice/global | 4 | 23 | 0 |
| **Total** | **214** | **374** | **187** |

| Structured or performance capability | Current authored coverage | Next requirement |
|---|---|---|
| Wavetables | One supervised choice in each oscillator; vocabulary has twenty choices including init | Combine registered choices with varied oscillator modes, levels, scanning and unison |
| Sampler | Zero enabled states; eleven choices including init available | Enable and vary content, playback and its interactions with oscillators/filters |
| Eight LFO shapes | Every slot supervised in 76–78 states, with original 3–8 point shapes | Vary playback/sync controls and combinations; qualify additional source layouts |
| Three custom curves | Supervised in 63/48/63 states, original 3–6 point shapes | Broader oscillator/asset combinations and input layouts |
| Three seed slots | 16/30/15 supervised states with distinct exact uint32 labels | Broader active contexts and measured useful prediction; exact labels alone do not establish learning |
| Ordered connections/maps | All 64 positions used; fourteen of 32 sources and a limited destination subset; 205 stored positional topologies | Broader sources, destinations, keyed controls, mappings and graph behavior under active synthesis |
| Effects | Chorus and distortion enabled together in forty states; two enabled-effect sets including none | Activate the remaining effect families and vary their settings/order in combined contexts |
| MIDI expression | Eight reference/eight transfer performances; native transport and ingestion qualified separately | Train combined controls across the reference bank and measure held-out performance transfer |
| Raw preset layouts | Authored sources only; corpus layout coverage not audited | Qualify one-point curves and other eligible layout gaps explicitly |

The final audit took 115.35 seconds on free local CPU, including saved-pool validation. All 52 recorded runtime source hashes matched at completion. Curve display names are excluded from shape-diversity counts. It wrote no audio or model weights. Four focused checks pass, covering inventory accounting, exclusion of held-out sources, refusal of a narrow codec and curve-name invariance; Ruff and whitespace checks pass. The parent expression PR records the broader 344-test regression run. Static recipe coverage cannot replace its native checks or the later quality gates.

## Reproduce

Use a compatible frozen nested-state or expression pool and the same asset/curve banks as that pool. The auditor validates the saved pool before counting coverage and refuses an existing output directory.

```powershell
python research/training/reports/scripts/audit_capabilities.py --pool research/training/outputs/expression/pool.json --output research/training/outputs/combined-coverage
```

The JSON report pins the pool, codec, split manifest, scalar inventory and normalized runtime source hashes. It reports every scalar by name, per-slot asset/curve/seed/connection coverage and the available reference/transfer bank. It does not publish presets, asset payloads, state membership or weights. Performance-bank availability is separate from actual exposure: the previous [native integration](reports/EXPRESSIVE_MIDI_RESULTS.md) retained two examples per reference phrase, rather than training every possible pair.

## Next implementation and acceptance criteria

Build a combined curriculum that retains the original LFO, mapping and custom-curve source bindings while varying the sampler, registered assets, voice behavior, envelopes, filters, effects and positional connections together. Preserve the frozen root assignments. Track each control's intended prerequisites and declared range; record deliberately fixed or approximated cases explicitly. Merely increasing pool size while the same controls stay constant does not close those gaps.

Before scaling, require native state/readback and audible-effect checks for newly activated families, fixed-phase correctness repeats, source/label integrity, explicit headroom/tail accounting, and bounded training with coverage of every reference performance. Report rejected renders in the full denominator. Recipe variation alone must never pass the native, learning or release gates.

Keep raw input-layout coverage separate. One-point curve inputs and other accepted-layout gaps need explicit parser/codec qualification; the current authored-source audit does not inspect the corpus. Broader graph behavior and range/combination coverage also remain open. Measured data scaling and useful held-out sound reconstruction follow the combined capability work, before any final-test or release decision.
