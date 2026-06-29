# OBRUXO Data Submission Guide

OBRUXO is a FOSS project to train a machine learning model that takes in raw audio files and converts them directly into playable `.vital` patches for the Vital synthesizer by Matt Tytel.

OBRUXO is named in tribute of Hermeto Pascoal.

# Ground Rules

- **Your outputs are yours**: The presets created by this tool and any music made with them belong to you. Use the presets in your music, give them to friends, or sell them as allowed by the laws in your country. No royalties, no restrictions.
- **The data stays here**: Your submissions are used purely to develop and improve this tool for the community.
- **No corporate cloning**: No rent-seeking middlemen allowed.
  - **DO NOT commercialise** this project, the code, the model, or the dataset
  - **DO commercialise** the music you create with the tool
  - **DO NOT** scrape this data for private models or hide this tool behind a paywall
  - **DO** hack and remix this code to build more **FOSS** tools for the community - just keep it free

# Data Submission Guide

Submissions are not open yet; this guide describes the planned format.

Each submission is basically just a Vital preset paired with a couple of recordings of you playing it. This gives the model the raw material it needs to learn how the synth actually sounds in action.

*(I am open to feedback on this plan, so let me know if you have any thoughts on improving the submission process. My priority is keeping the process simple to encourage more submissions.)*

## Summary

- One `PresetName` folder per preset
- One `PresetName.vital` file inside each folder
- At least 2 WAV + MIDI pairs per preset, named `PresetName_1`, `PresetName_2`, etc.
   - Each recording should be 1-10 seconds long
   - No external fx on the recording
- *Optional: mix context recordings attached to clean recordings*
- *Optional: MP3 versions included if available (256kbps+, same recording as WAV)*
- *Optional: `tags.txt` for describing recordings*

## What to Submit

For each preset you want to contribute:

1. **The preset file** - `YourPresetName.vital`
2. **At least 2 recordings** of that preset, each with its MIDI

Each recording should be a **DIFFERENT** musical performance of the same preset. Vary how you play it to give the model a broader understanding of the sound. You can submit multiple takes of the same idea, or entirely different ideas.

Here are some suggestions for variations:
- **Takes**: Separate recordings of the same core idea with a slight variation - different pitch, octave, articulation, phrasing, or a small flourish.
- **Ideas**: Separate recordings of entirely different musical ideas using the same preset - one file could be a melody, another chord stabs; one a single held note, another an arpeggio run.

## What makes a good recording

Good recordings give the model a clear example of what the preset sounds like in use.

- Leave enough space at the end to hear the full decay, reverb, echo, or release tail.
- Submit a mix of short, medium, and longer recordings. Do not make every recording the same length.
- Try different playing styles where it makes sense: single notes, melodies, chords, stabs, arps, basslines, pads, drums, or sound effects.
- Use the preset naturally. If the sound is meant to be played fast, play it fast. If it is meant to bloom slowly, give it room.
- Avoid clipping. The recording should be loud enough to hear clearly, but not distorted by the export.
- Keep silence at the start short. A little breathing room is fine, but the sound should begin quickly.
- Do the sound design INSIDE of Vital. Do not add EXTERNAL EQ, reverb, compression, saturation, limiting, or mastering.

## Requirements

- Record directly from Vital in your DAW. **No external effects** - EQ, reverb, compression, etc. must not be applied to the Vital track. Vital's own internal effects are fine and should stay on.
- Audio must be **WAV**. No MP3 as a primary format.
- **Export the MIDI** you used for each recording alongside the audio.
- Each recording must be between **1 and 10 seconds** long, including any natural decay, reverb, or echo tail.
- At least **2 clean recordings** per preset. Maximum of 8.

## File Structure

Create one folder per preset. Name the audio and MIDI files using the preset name as a prefix, followed by `_1`, `_2`, etc.

```text
submission.zip
├── gfunklead/
│   ├── gfunklead.vital
    ├── tags.txt              [4. optional tags]
│   ├── gfunklead_1.mid
│   ├── gfunklead_1.wav
│   ├── gfunklead_1.mp3       [3. optional mp3]
│   ├── gfunklead_2.mid
│   ├── gfunklead_2.wav
│   └── gfunklead_2.mp3       [3. optional mp3]
└── UKGBassline/
    ├── UKGBassline.vital
    ├── UKGBassline_1.mid
    ├── UKGBassline_1.wav
    ├── UKGBassline_1_mix.wav [2. optional mix context]
    ├── UKGBassline_2.mid
    ├── UKGBassline_2.wav
    ├── UKGBassline_3.mid     [1. optional additional midi+recording]
    └── UKGBassline_3.wav     [1. optional additional midi+recording]
```

- One folder per preset
- Folder name can be anything
- The `.vital` filename is what links everything together
- Clean audio and MIDI files must share the same prefix and number
- Mix context files should use the same prefix and number, followed by `_mix`

Zip the whole thing and submit.

## Optional

### 1. Additional recordings

You can include up to 8 recordings (takes/ideas) per preset.

### 2. Mix context recordings

Sometimes the sound you want to recreate is a synth line inside a fuller mix, where other instruments, drums, vocals, or effects are partly obscuring it.

You can optionally include a mix context recording for this. This is useful, but it should not replace the clean recording.

Think of this as an extra version of a clean recording, not a separate required take/idea.

For each mix context recording:

- Include the matching clean WAV recording.
- Include the matching MIDI.
- Use the same musical performance as the clean version if possible.
- Name it with `_mix`, for example `PresetName_1_mix.wav`.
- Keep it between 1 and 10 seconds long.

### 3. MP3 versions

If you can easily export an MP3 of each recording from your DAW, please include it. It must be the same recording as the WAV - just a compressed version. Minimum 256kbps.

This helps the model learn to work with real-world audio that has been compressed.

### 4. Tagging your recordings

Since the same synth preset can sound completely different depending on how you play it, you can optionally tag your recordings to help the model learn better. This will help us structure the training curriculum more effectively.

If you want to do this, just create a text file named `tags.txt` inside your preset folder. In it, write the recording number, a colon, and your tags separated by commas. You can list as many tags as you want in any order.

<details>
<summary>Suggested tag format and tags</summary>

#### Format

```text
1: mono, lead, dry, genre-gfunk
2: poly, chords, pad, genre-synthwave, shimmer, reverb, wide, chorus
3: atonal, genre-techno, drum, snare
4: monophonic, arp
```

#### Suggested Tags

To keep things consistent for our dataset, try to use these standard tags when describing your recordings:

1. **Core Type (Pick one):**
   * `monophonic` / `mono` - Playing one note at a time
   * `polyphonic` / `poly` - Playing multiple overlapping notes
   * `atonal` - Sounds with no clear pitch center

2. **Function (List as many as apply):**
   * `arp` - Arpeggiators or fast sequenced patterns.
   * `lead` - Main melodic lines.
   * `pad` - Long, evolving, sustained sounds.
   * `chords` - Harmonic block chords.
   * `stabs` - Short, punchy, decaying notes/chords.
   * `drum` / `perc` / `snare` / `...` - Synthesized drum hits or acoustic-like percussion elements.

3. **Genres:**
   * Prefix any genre with `genre-` (like `genre-gfunk`, `genre-house`, `genre-lofi`, `genre-synthwave`)

4. **Any** other tags you can think of.

</details>

# Open Questions

### [TBD] - Where should .zip files be sent?

### [TBD] Community Validation?

I am kicking around the idea of setting up a community vote so you can help pick the favorite submissions. I would use these top picks as my main validation set - basically a benchmark to check if the model is learning the sounds people actually care about. I might also reach out later for your qualitative feedback on how you think the model is shaping up.
