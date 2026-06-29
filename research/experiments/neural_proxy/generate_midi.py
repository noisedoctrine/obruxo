# %%
# Pretrain MIDI Generator

# TODO:
# - CRITICAL BUG: file durations not following expected distribution
#     This is due to blank space in midi files being ignored by miditoolkit when calculating max_tick.
#     The fix is to use a CC123 event at the desired end time to make sure max tick reflects the intended duration, even if there are no notes in the trailing silence
# - Showing stats about duration of generated files. every 1 second bin (index) from MIN-MAX duration (exactly =MIN and =MAX should have their own row), how many files fall into that bin (values), broken down by stage (columns)
# - KIV - no long files for arpeggios - why are some patterns in shorter files than others? there should be no pattern-based bias in duration distribution, since duration is sampled independently of pattern choice.

import random
import os
from pathlib import Path

from miditoolkit import MidiFile, Instrument, Note, TempoChange, ControlChange, PitchBend

# %%
# User defined parameters
OUTPUT_DIR = Path(__file__).resolve().parent / "generated_midi"
STAGES_TO_GENERATE = [1, 2, 3]
NUM_FILES_PER_STAGE = {
    1: 2000,
    2: 2000,
    3: 2000,
}
RANDOM_SEED = 1
BPM_MIN = 70
BPM_MAX = 170
PITCH_MIN = 36  # C2
PITCH_MAX = 96  # C7
PITCH_BIAS_LOW = 48  # C3
PITCH_BIAS_HIGH = 84  # C5
SUBBASS_MIN = 24  # C1
SUBBASS_MAX = 48  # C3
VELOCITY_RANGE = (60, 110)
MIN_DURATION = 0.1
DEFAULT_TICKS_PER_BEAT = 480
PAN_PROB = 0.10
PITCH_BEND_PROB = 0.05
MAX_POLYPHONY = 7
TOTAL_DURATION_MIN = 1.0
TOTAL_DURATION_MAX = 10.0
TRAILING_SILENCE_PROB = 0.35
MIN_TRAILING_SILENCE = 1.0
MAX_TRAILING_SILENCE = 3.0

FILE_DURATION_HARD_MIN = 1.0
FILE_DURATION_HARD_MAX = 10.0
MAX_GENERATION_ATTEMPTS = 100

# Naming scheme: s<stage>_<index>_<pattern>_<tags>.mid
# Example: s1_001_single-held_mono-tonal-mid_tail.mid

# %%
# Utility functions

def ensure_output_dir():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def validate_config():
    if not (FILE_DURATION_HARD_MIN <= TOTAL_DURATION_MIN <= TOTAL_DURATION_MAX <= FILE_DURATION_HARD_MAX):
        raise ValueError(
            "TOTAL_DURATION_MIN/MAX must stay within the hard "
            f"{FILE_DURATION_HARD_MIN:.1f}-{FILE_DURATION_HARD_MAX:.1f}s file duration limits."
        )
    if MIN_DURATION <= 0:
        raise ValueError("MIN_DURATION must be positive.")
    if MIN_DURATION > FILE_DURATION_HARD_MIN:
        raise ValueError("MIN_DURATION cannot exceed the minimum file duration.")
    if BPM_MIN <= 0 or BPM_MAX < BPM_MIN:
        raise ValueError("BPM_MIN/MAX must define a positive tempo range.")
    if MIN_TRAILING_SILENCE < 0 or MAX_TRAILING_SILENCE < MIN_TRAILING_SILENCE:
        raise ValueError("MIN_TRAILING_SILENCE/MAX_TRAILING_SILENCE must define a non-negative range.")


def sanitize_token(token: str) -> str:
    return token.lower().replace(" ", "-").replace("/", "-")


def build_filename(stage: int, index: int, pattern_name: str, tags: list[str]) -> str:
    pattern_token = sanitize_token(pattern_name)
    tag_token = "-".join(sanitize_token(tag) for tag in tags)
    return f"s{stage}_{index:03d}_{pattern_token}_{tag_token}.mid"


def sample_bpm() -> int:
    return random.randint(BPM_MIN, BPM_MAX)


def sample_total_duration() -> float:
    virtual_min = TOTAL_DURATION_MIN - 0.5  # Virtual range
    virtual_max = TOTAL_DURATION_MAX + 1
    raw_sample = random.betavariate(2, 2.4)  # Continuous flat beta sample
    duration = virtual_min + raw_sample * (virtual_max - virtual_min)  # Scale to the extended virtual range
    clipped = max(TOTAL_DURATION_MIN, duration)  # clip to system minimum
    return min(clipped, TOTAL_DURATION_MAX, FILE_DURATION_HARD_MAX)  # Hard clip to system maximum and absolute ceiling


def sample_trailing_silence(total_duration: float) -> float:
    if random.random() >= TRAILING_SILENCE_PROB:
        return 0.0

    max_silence = min(MAX_TRAILING_SILENCE, total_duration - MIN_DURATION)
    if max_silence < MIN_TRAILING_SILENCE:
        return 0.0

    return random.uniform(MIN_TRAILING_SILENCE, max_silence)


def sample_pitch() -> int:
    return int(random.triangular(PITCH_MIN, PITCH_MAX, (PITCH_BIAS_LOW + PITCH_BIAS_HIGH) / 2))


def sample_subbass_pitch() -> int:
    return int(random.triangular(SUBBASS_MIN, SUBBASS_MAX, (SUBBASS_MIN + SUBBASS_MAX) / 2))


def sample_velocity(base: int | None = None, vary: bool = True) -> int:
    if not vary or base is None:
        return random.randint(*VELOCITY_RANGE)
    delta = random.randint(-20, 20)
    return max(1, min(127, (base + delta)))


def clamp_duration(value: float) -> float:
    return max(MIN_DURATION, value)


def jitter_time(value: float, grid: bool = True) -> float:
    if grid:
        return round(value, 2)
    jitter = random.uniform(-0.08, 0.08)
    return max(0.0, value + jitter)


def seconds_to_ticks(seconds: float, bpm: int, ticks_per_beat: int) -> int:
    return int(round(seconds * ticks_per_beat * bpm / 60.0))


def get_notes_end_time(notes: list[dict]) -> float:
    return max((note["start_time"] + note["duration"] for note in notes), default=0.0)


def fit_notes_to_duration(notes: list[dict], content_duration: float) -> list[dict]:
    """Scale and clip notes so that all note-offs land at or before content_duration.
    Trailing silence is not a concern here.. it is simply the gap between the last
    note-off and the end of the file, which is produced naturally by writing nothing."""
    notes = [note.copy() for note in notes]
    if content_duration < MIN_DURATION:
        return []

    raw_end = get_notes_end_time(notes)

    if raw_end > content_duration:
        scale = content_duration / raw_end
        for note in notes:
            note["start_time"] *= scale
            note["duration"] *= scale

    fitted_notes = []
    for note in notes:
        start_time = max(0.0, note["start_time"])
        end_time = min(content_duration, start_time + note["duration"])
        duration = end_time - start_time
        if duration >= MIN_DURATION:
            note["start_time"] = start_time
            note["duration"] = duration
            fitted_notes.append(note)

    return fitted_notes


def write_midi_file(notes: list[dict], bpm: int, filepath: Path, pan: bool = False, pitch_bend: bool = False) -> None:
    mid = MidiFile(ticks_per_beat=DEFAULT_TICKS_PER_BEAT)
    instrument = Instrument(program=0, is_drum=False, name="generated")

    mid.tempo_changes = [TempoChange(bpm, 0)]

    pan_note = random.choice(notes) if pan and notes else None
    pitch_bend_note = random.choice(notes) if pitch_bend and notes else None

    last_tick = 0
    for note in notes:
        start_tick = seconds_to_ticks(note["start_time"], bpm, mid.ticks_per_beat)
        end_tick = seconds_to_ticks(note["start_time"] + note["duration"], bpm, mid.ticks_per_beat)
        instrument.notes.append(Note(velocity=note["velocity"], pitch=note["pitch"], start=start_tick, end=end_tick))
        last_tick = max(last_tick, end_tick)

        if pan_note is note:
            pan_value = random.randint(0, 127)
            instrument.control_changes.append(ControlChange(10, pan_value, start_tick))
            instrument.control_changes.append(ControlChange(10, 64, end_tick))

        if pitch_bend_note is note:
            pitch_bend_value = random.randint(-4096, 4096)
            instrument.pitch_bends.append(PitchBend(pitch_bend_value, start_tick))
            instrument.pitch_bends.append(PitchBend(0, end_tick))

    instrument.notes.sort(key=lambda midi_note: (midi_note.start, midi_note.pitch, midi_note.end))
    instrument.control_changes.sort(key=lambda control_change: control_change.time)
    instrument.pitch_bends.sort(key=lambda pitch_bend_event: pitch_bend_event.time)
    mid.instruments.append(instrument)
    # max_tick is the last note-off; trailing silence is simply the gap between
    # that tick and the end of the file... produced by writing nothing after it.
    mid.max_tick = last_tick

    mid.dump(filepath)

# %%
# Scale and chord helpers

def generate_scale(root: int, size: int) -> list[int]:
    intervals = sorted(random.sample(range(0, 12), k=size))
    return [root + interval for interval in intervals]


def make_chord(root: int, notes_count: int) -> list[int]:
    chord_steps = sorted(random.sample(range(0, 24), k=notes_count))
    return [root + step for step in chord_steps]


def make_polyphonic_texture(voices: int, duration: float) -> list[dict]:
    notes = []
    for voice_index in range(voices):
        pitch = sample_pitch()
        start = random.uniform(0.0, 2.0)
        notes.append({
            "pitch": pitch,
            "velocity": sample_velocity(vary=random.choice([True, False])),
            "start_time": clamp_duration(start),
            "duration": clamp_duration(random.uniform(duration * 0.8, duration * 1.2)),
        })
    return notes

# Arpeggio speed multipliers: binary subdivisions and triplet subdivisions.
# Each tuple is (label_suffix, notes_per_beat_multiplier).
# At 120 BPM, x1 = one note per beat = 0.5 s/note.
# x2 = eighth notes, x4 = sixteenth notes, x8 = 32nd notes, x16 = 64th notes.
# x3 = triplet quarter, x6 = triplet eighth, x12 = triplet sixteenth.
_ARP_SPEED_OPTIONS: list[tuple[str, float]] = [
    ("x2",  2.0),
    ("x4",  4.0),
    ("x8",  8.0),
    ("x16", 16.0),
    ("x3",  3.0),
    ("x6",  6.0),
    ("x12", 12.0),
]

# Direction modes for arps
_ARP_DIRECTIONS = ["asc", "desc", "asc-desc", "random"]


def make_arpeggio(bpm: int, total_seconds: float | None = None) -> tuple[str, list[str], list[dict]]:
    """
    Generate a single arpeggio pattern with randomised speed and direction.

    Returns (pattern_name, extra_tags, notes).  The caller supplies bpm so that
    the note-interval can be expressed in real seconds, and optionally a rough
    total_seconds budget so that very fast arps still produce enough repetitions
    to be interesting.
    """
    speed_label, multiplier = random.choice(_ARP_SPEED_OPTIONS)
    direction = random.choice(_ARP_DIRECTIONS)
    is_triplet = speed_label.startswith("x") and int(speed_label[1:]) % 3 == 0 and int(speed_label[1:]) % 2 != 0

    chord_size = random.choice([3, 4, 5])
    root = sample_pitch()
    chord_pitches = sorted(make_chord(root, chord_size))

    # interval between successive notes in seconds
    beat_seconds = 60.0 / bpm
    interval = beat_seconds / multiplier

    # Build the sequence of pitches for one cycle
    if direction == "asc":
        cycle = chord_pitches
    elif direction == "desc":
        cycle = list(reversed(chord_pitches))
    elif direction == "asc-desc":
        cycle = chord_pitches + list(reversed(chord_pitches[1:-1]))
    else:  # random
        cycle = random.sample(chord_pitches, len(chord_pitches))

    # How many notes to generate: fill ~2–4 full cycles, capped by budget
    budget = total_seconds if total_seconds is not None else 4.0
    max_notes_by_budget = max(len(cycle), int(budget / interval))
    num_cycles = random.randint(2, 4)
    num_notes = min(len(cycle) * num_cycles, max_notes_by_budget)

    base_velocity = random.randint(*VELOCITY_RANGE)
    use_jitter = random.choice([True, False])
    notes = []
    for i in range(num_notes):
        pitch = cycle[i % len(cycle)]
        start = jitter_time(i * interval, grid=not use_jitter)
        notes.append({
            "pitch": pitch,
            "velocity": sample_velocity(base=base_velocity, vary=random.choice([True, False])),
            "start_time": max(0.0, start),
            "duration": clamp_duration(interval * random.uniform(0.70, 0.95)),
        })

    tags = [speed_label, direction]
    if is_triplet:
        tags.append("triplet")
    pattern_name = f"arpeggio-{speed_label}-{direction}"
    return pattern_name, tags, notes


def make_stage1_patterns(stage: int) -> list[tuple[str, list[str], list[dict], int]]:
    patterns = []
    tags = ["mono", "tonal"]

    pitch = sample_pitch()
    velocity = sample_velocity()
    duration = clamp_duration(random.uniform(1.0, 2.5))
    notes = [{"pitch": pitch, "velocity": velocity, "start_time": 0.0, "duration": duration}]
    patterns.append(("single-held", tags + ["sustained"], notes, sample_bpm()))

    pitch = sample_pitch()
    steps = random.randint(3, 6)
    base_velocity = random.randint(*VELOCITY_RANGE)
    notes = []
    for i in range(steps):
        notes.append({
            "pitch": pitch,
            "velocity": sample_velocity(base_velocity, vary=(i % 2 == 1)),
            "start_time": jitter_time(i * 0.4, grid=random.choice([True, False])),
            "duration": clamp_duration(0.35 + random.uniform(-0.05, 0.05)),
        })
    patterns.append(("repeated-note", tags + ["rhythm"], notes, sample_bpm()))

    subbass_pitch = sample_subbass_pitch()
    notes = [{
        "pitch": subbass_pitch,
        "velocity": sample_velocity(base=random.randint(*VELOCITY_RANGE), vary=random.choice([True, False])),
        "start_time": 0.0,
        "duration": clamp_duration(random.uniform(1.2, 3.0)),
    }]
    patterns.append(("subbass-one-shot", tags + ["subbass", "low"], notes, sample_bpm()))

    pitch = sample_pitch()
    notes = []
    for i in range(4):
        notes.append({
            "pitch": pitch,
            "velocity": sample_velocity(base=None, vary=True),
            "start_time": jitter_time(i * 0.6, grid=random.choice([True, False])),
            "duration": clamp_duration(0.5 + random.uniform(-0.1, 0.1)),
        })
    patterns.append(("single-note-variation", tags + ["dynamic"], notes, sample_bpm()))

    return patterns


def make_stage2_patterns(stage: int) -> list[tuple[str, list[str], list[dict], int]]:
    patterns = []
    tags = ["mono", "melody"]

    root = sample_pitch()
    scale_notes = generate_scale(root, random.randint(3, 9))
    notes = []
    for i, pitch in enumerate(scale_notes[:6]):
        notes.append({
            "pitch": pitch,
            "velocity": sample_velocity(base=random.randint(*VELOCITY_RANGE), vary=random.choice([True, False])),
            "start_time": jitter_time(i * 0.4, grid=random.choice([True, False])),
            "duration": clamp_duration(0.4 + random.uniform(-0.05, 0.1)),
        })
    patterns.append(("melodic-walk", tags + ["scale"], notes, sample_bpm()))

    notes = []
    scale = sorted(scale_notes)
    for i in range(min(5, len(scale))):
        notes.append({
            "pitch": scale[-1 - i],
            "velocity": sample_velocity(base=90, vary=random.choice([True, False])),
            "start_time": jitter_time(i * 0.5, grid=random.choice([True, False])),
            "duration": clamp_duration(0.45 + random.uniform(-0.05, 0.05)),
        })
    patterns.append(("melody-desc", tags + ["interval"], notes, sample_bpm()))

    bpm_for_arp = sample_bpm()
    arp_name, arp_tags, arp_notes = make_arpeggio(bpm_for_arp, total_seconds=TOTAL_DURATION_MAX)
    patterns.append((arp_name, tags + arp_tags + ["triad"], arp_notes, bpm_for_arp))

    return patterns


def make_stage3_patterns(stage: int) -> list[tuple[str, list[str], list[dict], int]]:
    patterns = []

    chord_notes = make_chord(sample_pitch(), random.choices([2, 3, 4, 5, 6, 7], weights=[5, 20, 30, 25, 15, 5])[0])
    notes = [
        {"pitch": pitch, "velocity": sample_velocity(base=90, vary=random.choice([True, False])), "start_time": 0.0, "duration": clamp_duration(random.uniform(2.0, 4.0))}
        for pitch in sorted(chord_notes)
    ]
    patterns.append(("static-chord", ["poly", "chord"], notes, sample_bpm()))

    progression_roots = [sample_pitch(), sample_pitch(), sample_pitch()]
    notes = []
    for chord_index, root in enumerate(progression_roots):
        chord_notes = make_chord(root, random.choices([2, 3, 4, 5], weights=[5, 25, 40, 30])[0])
        for i, pitch in enumerate(sorted(chord_notes)):
            notes.append({
                "pitch": pitch,
                "velocity": sample_velocity(base=90, vary=random.choice([True, False])),
                "start_time": chord_index * 1.2 + i * 0.2,
                "duration": clamp_duration(0.3 + random.uniform(0.0, 0.2)),
            })
    patterns.append(("chord-progression", ["poly", "progression"], notes, sample_bpm()))

    # Arpeggio with randomised speed / direction / triplet feel
    bpm_for_arp = sample_bpm()
    arp_name, arp_tags, arp_notes = make_arpeggio(bpm_for_arp, total_seconds=TOTAL_DURATION_MAX)
    patterns.append((arp_name, ["poly", "arpeggio"] + arp_tags, arp_notes, bpm_for_arp))

    voices = random.randint(3, MAX_POLYPHONY)
    notes = make_polyphonic_texture(voices, duration=2.5)
    patterns.append(("polyphonic-texture", ["poly", "texture"], notes, sample_bpm()))

    cluster_root = sample_pitch()
    notes = []
    for i in range(random.randint(4, 7)):
        notes.append({
            "pitch": cluster_root + random.choice([0, 1, 3, 4, 6, 7, 8, 10, 11]),
            "velocity": sample_velocity(base=80, vary=random.choice([True, False])),
            "start_time": jitter_time(random.uniform(0.0, 1.6), grid=random.choice([True, False])),
            "duration": clamp_duration(random.uniform(0.15, 0.6)),
        })
    patterns.append(("cluster", ["poly", "atonal", "texture"], notes, sample_bpm()))

    return patterns

# %%
# Sequential Generation Worker Function

def generate_single_file(stage: int, file_number: int, index: int) -> dict:
    """Generate a single MIDI file sequentially."""
    
    # Deteministic, predictable seed assignment per file index
    random.seed(RANDOM_SEED + stage * 100_000 + file_number)
    
    stage_generators = {
        1: make_stage1_patterns,
        2: make_stage2_patterns,
        3: make_stage3_patterns,
    }
    
    generator = stage_generators[stage]
    for attempt in range(MAX_GENERATION_ATTEMPTS):
        pattern_name, tags, notes, bpm = random.choice(generator(stage))
        total_duration = sample_total_duration()
        total_duration = max(FILE_DURATION_HARD_MIN, min(total_duration, FILE_DURATION_HARD_MAX))
        trailing_silence = sample_trailing_silence(total_duration)
        content_duration = total_duration - trailing_silence
        fitted_notes = fit_notes_to_duration(notes, content_duration)
        if fitted_notes:
            notes = fitted_notes
            break
    else:
        raise RuntimeError(
            f"Failed to generate a stage {stage} MIDI file with notes >= {MIN_DURATION:.3f}s "
            f"after {MAX_GENERATION_ATTEMPTS} attempts."
        )

    if trailing_silence > 0.0:
        tags = tags + ["tail"]

    pan_file = random.random() < PAN_PROB
    pitch_bend_file = random.random() < PITCH_BEND_PROB
    filename = build_filename(stage, index, pattern_name, tags)
    filepath = OUTPUT_DIR / filename
    write_midi_file(notes, bpm, filepath, pan=pan_file, pitch_bend=pitch_bend_file)

    return {
        "pattern": pattern_name,
        "tags": tags,
        "pan": pan_file,
        "pitch_bend": pitch_bend_file,
        "duration": total_duration,
        "file_number": file_number,
    }


# %%
# Runner

def generate_stage(stage: int, count: int) -> dict:
    stage_generators = {
        1: make_stage1_patterns,
        2: make_stage2_patterns,
        3: make_stage3_patterns,
    }

    if stage not in stage_generators:
        raise ValueError(f"Unsupported stage: {stage}")

    stage_stats = {
        "files": 0,
        "pan_files": 0,
        "pitch_bend_files": 0,
        "duration_sum": 0.0,
        "pattern_counts": {},
        "tag_counts": {},
    }

    # Generate files cleanly in a single-threaded loop
    for file_num in range(count):
        result = generate_single_file(stage, file_num, file_num + 1)
        stage_stats["files"] += 1
        stage_stats["pan_files"] += int(result["pan"])
        stage_stats["pitch_bend_files"] += int(result["pitch_bend"])
        stage_stats["duration_sum"] += result["duration"]
        stage_stats["pattern_counts"][result["pattern"]] = \
            stage_stats["pattern_counts"].get(result["pattern"], 0) + 1
        for tag in result["tags"]:
            stage_stats["tag_counts"][tag] = stage_stats["tag_counts"].get(tag, 0) + 1
        
        # Log progress
        should_log = count < 50 or (file_num + 1) % 1000 == 0 or file_num == count - 1
        if should_log:
            print(f"Stage {stage}: wrote {file_num + 1}/{count} files (pattern={result['pattern']})")

    return stage_stats


# %%
# Run and log
if __name__ == "__main__":

    print("starting pretrain_midi_generator", Path.cwd(), OUTPUT_DIR)
    validate_config()
    random.seed(RANDOM_SEED)
    ensure_output_dir()

    total_stats = {
        "files": 0,
        "pan_files": 0,
        "pitch_bend_files": 0,
        "duration_sum": 0.0,
        "pattern_counts": {},
        "tag_counts": {},
    }

    import pandas as pd

    # Track tag stats per stage
    stage_tags = {}

    for stage in STAGES_TO_GENERATE:
        count = NUM_FILES_PER_STAGE.get(stage, 0)
        if count <= 0:
            continue
        
        print(f"Generating stage {stage}: {count} files")
        stage_stats = generate_stage(stage, count)
        
        # Store stage tags for dataframe conversion
        stage_tags[stage] = stage_stats["tag_counts"]

        # Update totals
        total_stats["files"] += stage_stats["files"]
        total_stats["pan_files"] += stage_stats["pan_files"]
        total_stats["pitch_bend_files"] += stage_stats["pitch_bend_files"]
        total_stats["duration_sum"] += stage_stats["duration_sum"]

    if total_stats["files"] > 0:
        avg_duration = total_stats["duration_sum"] / total_stats["files"]
        print(f"\nFinal generation stats:")
        print(f"  files generated: {total_stats['files']}")
        print(f"  pan files: {total_stats['pan_files']}")
        print(f"  pitch bend files: {total_stats['pitch_bend_files']}")
        print(f"  average duration/file: {avg_duration:.2f}s\n")

        # Tabulate Tags Only
        df_tags = pd.DataFrame(stage_tags).sort_index().fillna(0).astype(int)
        print("Tag Counts Across Stages:")
        print(df_tags.replace(0, "").to_string())
        print()

        # # TODO: Tabulate Files by Length Bins

    print("Done.")
