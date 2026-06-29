# Neural Proxy Experiment: Synthetic Data Generation Findings

## Overview
Research findings for generating synthetic training data to pretrain a possible neural proxy for `Vital(patch, MIDI) → audio`. The proxy is one candidate mechanism for propagating perceptual training signals through non-differentiable Vital; it is not the core audio-to-parameter system.

## MIDI Format Options

### Option 1: Standard .mid Files
- **Pros**: Industry standard, compatible with all DAWs and tools
- **Cons**: Requires parsing for programmatic use, binary format
- **Tools**: `pretty_midi`, `mido`, `music21` libraries for parsing

### Option 2: JSON Note Sequences (Recommended for Synthetic Generation)
- **Pros**: Human-readable, easy to generate programmatically, no parsing needed
- **Cons**: Requires conversion to .mid for DAW compatibility
- **Format**:
```json
{
  "bpm": 120,
  "notes": [
    {"pitch": 60, "velocity": 0.7, "start_time": 0.0, "duration": 1.0},
    {"pitch": 64, "velocity": 0.8, "start_time": 1.0, "duration": 0.5}
  ]
}
```

## Vital Interface Options

### Headless Vital
- **Command-line interface**: `vital --headless --render -m <pitch> -l <length> -b <bpm>`
- **Input**: Individual note parameters (not MIDI files)
- **Example**: `vital --headless --render -m 60 -l 1.0 -b 120`
- **Status**: Reported segmentation faults on Mac, may be unstable

### Python Vita Package
- **Library**: `pip install vita`
- **Input**: Individual note parameters via function calls
- **Example**: `synth.render(pitch=60, velocity=0.7, note_dur=1.0, render_dur=3.0)`
- **Advantages**: More stable, programmatic control, JSON preset support
- **Documentation**: Browse `bindings.cpp` for API details

**Key Finding**: Both tools require individual note parameters, not MIDI files directly.

## MIDI Parsing Workflow (For Existing MIDI Files)

If you have existing .mid files (e.g., from user submissions):

```python
import pretty_midi as pm

# Load MIDI file
midi = pm.PrettyMIDI('file.mid')

# Extract notes
notes = []
for instrument in midi.instruments:
    for note in instrument.notes:
        notes.append({
            "pitch": note.pitch,
            "velocity": note.velocity / 127.0,  # normalize to 0-1
            "start_time": note.start,
            "duration": note.end - note.start
        })

# Convert to JSON
json_output = {
    "bpm": midi.get_tempo_changes()[1][0],  # get tempo
    "notes": notes
}
```

**Libraries**:
- `pretty_midi`: Recommended for performance and ease of use
- Alternative: `midi-json-parser` (JavaScript/TypeScript)

## .vital Preset Format

**Key Finding**: .vital preset files are JSON format.

### Python Vita Preset Operations
```python
# Export preset to JSON
json_text = synth.to_json()
with open("preset.vital", "w") as f:
    f.write(json_text)

# Load preset from JSON
with open("preset.vital", "r") as f:
    json_text = f.read()
    synth.load_json(json_text)
```

### Implications for Synthetic Generation
- Presets can be generated programmatically as JSON
- No conversion needed - .vital files are already JSON
- Structure contains all synth parameters: oscillators, filters, envelopes, mod matrix, etc.

## Recommended Synthetic Data Generation Approach

### For Pure Synthetic Generation (No Existing Files)
1. **Generate JSON note sequences directly** - skip .mid files entirely
2. **Generate JSON preset files directly** - .vital format is already JSON
3. **Feed to Python Vita** - use `synth.render()` with note parameters
4. **Benefits**: No parsing overhead, full programmatic control

### For Mixed Synthetic + Real Data
1. **Parse existing .mid files** using `pretty_midi`
2. **Convert to JSON note sequences**
3. **Generate synthetic presets** as JSON
4. **Combine** with real user-submitted presets
5. **Feed to Python Vita** for audio generation

## Data Requirements for Neural Surrogate

### MIDI/Performance Data

#### Curriculum Stages

**Stage 1: Single Note Patterns (Simplest)**
- **Duration**: 1-3 seconds
- **Patterns**:
  - Single held notes at various pitches (C2-C7 range)
  - Single repeated notes with different rhythms
  - Single notes with velocity variations
- **Diversity**: 1000 files across pitch range, durations, and velocities
- **Tags**: mono, tonal

**Stage 2: Monophonic Melodies**
- **Duration**: 2-6 seconds
- **Patterns**:
  - Simple ascending/descending scales (major, minor, pentatonic)
  - Simple arpeggios (triads, 7th chords)
  - Random walks within pitch constraints
  - Repeated melodic motifs (2-4 note patterns)
- **Diversity**: 2000 files across scales, tempos, and contour types
- **Tags**: mono, tonal, melody, lead

**Stage 3: Complex Patterns**
- **Duration**: 2-10 seconds
- **Patterns**:
  - **Polyphonic chords**: Static chords, chord progressions, different voicings
  - **Rhythmic patterns**: Staccato/legato, varying gate times, rhythmic density
  - **Arpeggios**: Up/down patterns, different subdivisions, tempos
  - **Extended techniques**: Pitch bends, velocity swells, sustained pads, drones
  - **Atonal/noise**: Random pitch clusters, microtonal intervals, percussive patterns
  - **Register variations**: Sub/bass/mid/high patterns, cross-register
- **Diversity**: 6000 files across all pattern types
- **Tags**: mono/poly, tonal/atonal, chords/arp/pad/lead/drum, staccato/legato, sub/bass/mid/high

#### Dataset Structure
- **Total Target**: ~9,000 MIDI files across 3 stages
- **Distribution**: Weighted toward simpler stages (Stage 1: 1000, Stage 2: 2000, Stage 3: 6000)
- **Validation**: 10% holdout from each stage
- **File Format**: JSON note sequences (recommended) or .mid files
- **Metadata**: Each file paired with JSON containing stage identifier, tags, duration, note count, pitch range, tempo, pattern type

#### File Structure
```
synthetic_midi_dataset/
├── stage1_single_notes/
│   ├── file_001.json (or .mid)
│   ├── file_001_metadata.json
│   └── ...
├── stage2_melodies/
│   └── ...
└── stage3_complex/
    └── ...
```

### Preset Data

#### Parameter Coverage
- **Oscillators**: Various wavetable shapes, samples, noise types
- **Filters**: Different filter types (lowpass, highpass, bandpass, etc.), cutoff frequencies, resonance
- **Envelopes**: ADSR parameters for amplitude, filter, and pitch envelopes
- **LFOs**: Rate, depth, shape, routing to various destinations
- **Effects**: Reverb, delay, distortion, compression, chorus, etc.
- **Mod Matrix**: Variable-length modulation routing (0-32 connections)

#### Generation Strategy
- **Realistic combinations**: Musically sensible parameter relationships (e.g., filter cutoff related oscillator pitch)
- **Diversity**: Cover full parameter space with systematic exploration
- **Extremes**: Include both subtle and extreme parameter settings
- **Common patches**: Focus on frequently used synth types (leads, basses, pads, plucks)

#### File Format
- **Format**: JSON (.vital files are JSON format)
- **Structure**: All synth parameters in structured JSON
- **Generation**: Programmatic generation using Python Vita's parameter control API

## Next Steps
1. Implement JSON note sequence generator (3-stage curriculum)
2. Implement JSON preset generator (parameter space exploration)
3. Set up Python Vita pipeline for batch audio rendering
4. Generate synthetic dataset
5. Validate with spot-checking and quality metrics
