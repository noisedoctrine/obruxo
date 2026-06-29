# OBRUXO

OBRUXO is a FOSS research project to train a machine learning model that takes raw audio and turns it directly into playable `.vital` patches for [Vital](https://vital.audio/), the synthesizer by Matt Tytel.

OBRUXO is named in tribute to Hermeto Pascoal.

The core problem is audio-to-synth-parameter inference. The primary objective is perceptual similarity between the reference and audio rendered from a predicted patch; exact recovery of the source parameters is secondary because multiple patches can produce perceptually equivalent sounds.

The system is intended to remain fully local and open source: no cloud dependency, no proprietary inference service, and no rent-seeking middlemen enclosing the work behind a paywall.

## Ground rules

- **Your outputs are yours.** Presets created by the system and music made with them belong to you. Use them, share them, or sell your music without royalties owed to OBRUXO.
- **Keep OBRUXO free.** Hack it, remix it, and use it to build more FOSS tools but do not turn the project, model, or dataset into a private clone or closed commercial service.
- **Commercialise the music, not the enclosure.** The point is to give musicians a tool, not manufacture another middleman.

## Repository map

- [`research/`](research/) — current research framing, modelling decisions, architecture, Vital schema work, experiments, and archived historical plans.
- [`community/`](community/) — outward-facing community and data-contribution material.
- [`datasets/`](datasets/) — dataset sources, collection tooling, analysis, tracked derived results, and local raw data.

## Current research documents

- [Project brief](research/PROJECT_BRIEF.md)
- [Modelling research tracker](research/modelling/RESEARCH_TRACKER.md)
- [Model architecture](research/modelling/MODEL_ARCHITECTURE.md)
- [Vital preset schema](research/vital/PRESET_SCHEMA.md)

## Community

Community participation and the planned data-submission format are documented in the [community README](community/README.md).

## Data policy

Large downloaded and generated datasets are local working data and are excluded from version control. Collection and analysis code, canonical reference data, methodology, and derived corpus summaries remain in the repository for reproducibility.
