Goal: fully local, open-source system that infers Vital synthesizer presets from audio clips. 

A pipeline that takes an audio signal (optionally with a MIDI-derived representation in parallel), analyzes its timbre and musical structure, and outputs a valid .vital preset. That preset should approximate the original sound by recovering the underlying synthesis configuration inside Vital-oscillators, filters, envelopes, effects, and modulation routing. The system is trained entirely offline by generating our own dataset: randomly sample Vital presets, render them into audio, and use those paired examples to learn the inverse mapping. During training, also explore whether adding auxiliary structure - like audio to MIDI transcription or contrastive “same preset, different MIDI input” augmentations - improves robustness to polyphony and musical variation.
The core challenge is not just parameter regression, but learning a representation that separates: musical content (notes, chords, polyphony) from synthesis identity (the preset configuration that produces the timbre and motion) The end goal is a model that generalizes across different note patterns while still recovering the same underlying synth preset, and can export that back into a valid Vital .vital file without relying on cloud services or proprietary systems.

No code, we are writing up a framework. Conduct literature review. Topics include (but not limited to):
1. What is the structure of .vital files? How can we represent them for input/output into the model
2. What is the best FOSS option for audio -> midi
3. Audio -> spectrogram: any prebuilt layers we can use?
4. Should we use negative sampling with triplet/JEPA style?

Might be useful:
http://github.com/DBraun/Vita/blob/main/src/headless/bindings.cpp=