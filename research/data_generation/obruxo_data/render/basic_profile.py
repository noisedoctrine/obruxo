"""Qualified control domain shared by the staged codec and native render cache."""

CONTINUOUS = (
    ("osc_1_level", .15, .35, .25), ("osc_1_pan", -.7, .7, 0.),
    ("osc_1_unison_detune", .5, 2.5, 1.), ("filter_1_cutoff", 50., 120., 96.),
    ("filter_1_resonance", .1, .6, .25), ("env_1_attack", .08, .6, .15),
    ("env_1_decay", .25, 1., .7), ("env_1_sustain", .25, .9, .7), ("env_1_release", .2, .65, .4),
)
CATEGORICAL = (("osc_1_unison_voices", (1, 2, 4)), ("filter_1_on", (0, 1)))
CONTROL_NAMES = tuple(row[0] for row in (*CONTINUOUS, *CATEGORICAL))
PHASE_CONTROL = {**{f"osc_{index}_random_phase": 0. for index in range(1, 4)}, "sample_random_phase": 0.}
PREROLL_FRAMES = 32768
