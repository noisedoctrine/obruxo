# Basic models work through the audio-to-preset command

The existing `infer` command now restores both basic correction formats, including copied trainable audio features, and exports their full predictions. The output remains 14 dense values / 11 codec selections. Historical pilot loading keeps its previous behavior.

Fourteen local tests pass. They exercise the actual command with nonzero saved corrections, both model formats, the fixed phase policy, legacy CPU settings, invalid artifacts and existing-file preservation. Direct models use the same two CPU threads as their native-fit audits. Saved diagnostic models keep their unqualified release status in the JSON result.

The [real-audio check](data/basic_direct_inference_summary.json) compares six exports against exact preset hashes already retained by native audits: two saved model formats, each with three performances of one previously used training patch. All six match byte for byte, including oscillator/sampler phase controls. The direct artifact contains actual sound-trained weights; the trainable-input artifact is the qualified untrained initializer. This verifies deployment correctness, without measuring new reconstruction gains.

The check takes **3.81 seconds / 0.48 GiB**, writes 5.1 MiB of evidence and adds **zero renders or optimizer updates**. Source model files stay unchanged. Implementation `2e9f704` precedes the check. The running fit's 42 frozen source files also remain unchanged.

[Usage and supported artifacts](../BASIC_INFERENCE.md). The local fit continues; its final quality decision, fresh confirmation, listening and fresh-environment release bundle remain open.
