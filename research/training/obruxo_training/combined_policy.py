"""Declared, bounded raw-control sampling ranges for combined authored sources."""
import math
import re

from .full_scalar import QUANTIZE, scalar_family

POLICY = "combined_all_scalar_families_original_nested_sources_v1"
FIXED = {"bypass": (0., "non_silent_training_eligibility"), "beats_per_minute": (2., "host_performance_tempo_supplies_the_clock"),
         "view_spectrogram": (0., "display_state"), **{f"osc_{slot}_view_2d": (1., "display_state") for slot in range(1, 4)}}
RANGES = {"volume": (2601., 3136.), "voice_amplitude": (.6, 1.), "portamento_time": (-9., -4.), "portamento_slope": (-2., 2.),
          "voice_tune": (-.25, .25), "pitch_wheel": (-.25, .25), "velocity_track": (-.4, .4),
          "eq_low_cutoff": (24., 60.), "eq_high_cutoff": (84., 120.), "eq_band_cutoff": (48., 108.),
          "delay_frequency": (2., 6.), "delay_aux_frequency": (2., 6.), "reverb_decay_time": (-3., -.25),
          "reverb_delay": (.01, .12), "reverb_pre_low_cutoff": (8., 40.), "reverb_pre_high_cutoff": (90., 120.),
          "reverb_low_shelf_cutoff": (16., 48.), "reverb_high_shelf_cutoff": (85., 118.), "phaser_mod_depth": (4., 28.)}
SUFFIX_RANGES = (("_level", (.08, .16)), ("_attack_power", (-2., 2.)), ("_decay_power", (-2., 2.)), ("_release_power", (-2., 2.)),
    ("_attack", (.1, .6)), ("_decay", (.3, .85)), ("_sustain", (.3, .9)), ("_release", (.3, .8)),
    ("_delay", (0., .6)), ("_hold", (0., .6)), ("_delay_time", (0., .3)), ("_fade_time", (0., .5)),
    ("_smooth_time", (-10., -5.)), ("_frequency", (-3., 2.)), ("_feedback", (-.25, .35)), ("_formant_resonance", (.3, .75)),
    ("_resonance", (.05, .4)), ("_cutoff", (48., 116.)), ("_drive", (0., 6.)), ("_gain", (-6., 6.)),
    ("_dry_wet", (.12, .4)), ("_mix", (.15, .6)), ("_pan", (-.6, .6)), ("_phase", (0., 1.)),
    ("_phase_offset", (0., 1.)), ("_random_phase", (0., 1.)), ("_wave_frame", (0., 256.)),
    ("_frame_spread", (-48., 48.)), ("_spectral_morph_spread", (-.2, .2)), ("_distortion_spread", (-.2, .2)),
    ("_detune_range", (.5, 8.)), ("_detune_power", (-1., 2.)), ("_unison_detune", (.5, 2.)), ("_tune", (-.3, .3)))


def control_policy(scalar):
    result = {}
    for name in sorted(scalar.schema.scalar_names):
        prerequisite = scalar_family(name)
        if name in FIXED:
            value, reason = FIXED[name]
            result[name] = {"kind": "fixed", "value": value, "reason": reason}
            continue
        if name == "effect_chain_order":
            result[name] = {"kind": "permutation_rank", "minimum": 0, "maximum": math.factorial(9) - 1, "prerequisite": "two_or_more_enabled_effects"}
            continue
        if name in QUANTIZE:
            result[name] = {"kind": "pitch_mask", "minimum": 0, "maximum": 8191, "prerequisite": prerequisite}
            continue
        spec = scalar._parameters[name]
        if spec.is_discrete:
            values = list(range(math.ceil(spec.minimum), math.floor(spec.maximum) + 1))
            if name.endswith("_destination"):
                result[name] = {"kind": "choice", "values": list(range(5)), "prerequisite": prerequisite,
                                "excluded_native_values": list(range(5, 15)), "reason": "zero_output_for_5_to_14_with_all_nine_effects_on_or_off_in_native_qualification"}
                continue
            if name.endswith("_unison_voices"):
                values = [1, 2, 4, 8]
            elif name == "polyphony":
                values = [1, 4, 8]
            elif name == "oversampling":
                values = [0, 1]
            elif name == "pitch_bend_range":
                values = [2, 7, 12]
            elif name.endswith("_keytrack_transpose"):
                values = [-24, -12, 0, 12]
            elif name.endswith("_transpose"):
                values = [-12, 0, 12]
            result[name] = {"kind": "choice", "values": values, "prerequisite": prerequisite}
            continue
        lo, hi = RANGES.get(name, next((bounds for suffix, bounds in SUFFIX_RANGES if name.endswith(suffix)),
                                    (spec.minimum + .2 * (spec.maximum - spec.minimum), spec.minimum + .8 * (spec.maximum - spec.minimum))))
        if name.startswith("compressor_"):
            if name.endswith("_gain"):
                lo, hi = -9., 0.
            elif name.endswith("_lower_threshold"):
                lo, hi = -55., -35.
            elif name.endswith("_upper_threshold"):
                lo, hi = -24., -9.
        if re.fullmatch(r"modulation_\d+_amount", name):
            lo, hi = -.03, .03
        lo, hi = max(lo, spec.minimum), min(hi, spec.maximum)
        if lo >= hi:
            raise ValueError(f"combined range does not vary {name}")
        result[name] = {"kind": "uniform_raw", "minimum": lo, "maximum": hi, "prerequisite": prerequisite}
    return result
