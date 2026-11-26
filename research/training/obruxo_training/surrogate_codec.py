"""Versioned control representations for the two-stage Vital surrogate pilot."""

from base64 import b64encode
from copy import deepcopy
from itertools import product
import json

import numpy as np

from obruxo_data.hashing import canonical_json, canonical_sha256

from .codec import PilotCodec, PilotTarget


WAVEFORMS = ("saw", "sine", "triangle", "square")
FILTER_RESPONSES = ("low-pass", "band-pass", "high-pass")
FILTER_BLEND = (0., 1., 2.)
EXTRA_CONTINUOUS = (("osc_2_level", .06, .22, .15), ("osc_2_tune", -.2, .2, 0.),
                    ("reverb_dry_wet", .12, .40, .25), ("reverb_decay_time", -3., 0., 0.))
EXTRA_CATEGORICAL = (("osc_1_waveform", WAVEFORMS), ("filter_1_response", FILTER_RESPONSES),
                     ("osc_2_on", (0, 1)), ("osc_2_waveform", WAVEFORMS),
                     ("osc_2_transpose", (0, -12, 12)), ("reverb_on", (0, 1)))


def waveform_bank(saw):
    phase = np.arange(2048, dtype=np.float64) / 2048
    forms = (np.sin(2 * np.pi * phase), 1 - 4 * np.abs(phase - .5), np.where(phase < .5, 1., -1.))
    waves = [saw]
    for name, values in zip(WAVEFORMS[1:], forms, strict=True):
        wave = deepcopy(saw)
        wave["name"] = f"OBRUXO {name}"
        wave["groups"][0]["components"][0]["keyframes"][0]["wave_data"] = b64encode(
            values.astype("<f4").tobytes()).decode("ascii")
        waves.append(wave)
    return waves


class SurrogateCodec:
    """Use one binary slot for a known category and one-hot slots for larger categories."""

    def __init__(self, basic: PilotCodec, *, expanded: bool):
        self.basic, self.expanded = basic, expanded
        self.continuous = (*basic.continuous, *EXTRA_CONTINUOUS) if expanded else basic.continuous
        self.categorical = (*basic.categorical, *EXTRA_CATEGORICAL) if expanded else basic.categorical
        self.width = len(self.continuous) + sum(1 if len(values) == 2 else len(values)
                                                for _, values in self.categorical)
        self.defaults = tuple((default - lower) / (upper - lower) for _, lower, upper, default in self.continuous)
        seed_preset = json.loads(basic.decode(basic.default_target()))
        self.waves = waveform_bank(seed_preset["settings"]["wavetables"][0]) if expanded else []

    def spec(self):
        return {"format": "surrogate_pilot_controls_v1", "basic_codec": self.basic.to_dict(),
                "continuous": self.continuous, "categorical": self.categorical,
                "width": self.width, "binary_width": 1, "multiclass_representation": "one_hot",
                "waveform_sha256": [canonical_sha256(wave) for wave in self.waves],
                "filter_response_blend": FILTER_BLEND if self.expanded else (),
                "inactive": "canonical_fixed_values"}

    def masks(self, choices):
        unison, filtered = choices[:2]
        continuous = (True, True, bool(unison), bool(filtered), bool(filtered), True, True, True, True)
        if not self.expanded:
            return continuous, (True, True)
        layered, wet = bool(choices[4]), bool(choices[7])
        return (*continuous, layered, layered, wet, wet), (True, True, True, bool(filtered), True, layered, layered, True)

    def canonical(self, target):
        choices, values = target.categorical, target.continuous
        if len(choices) != len(self.categorical) or len(values) != len(self.continuous):
            raise ValueError("surrogate control shape mismatch")
        if any(not 0 <= i < len(options) for i, (_, options) in zip(choices, self.categorical, strict=True)):
            raise ValueError("illegal category index")
        if any(not 0 <= v <= 1 for v in values):
            raise ValueError("continuous controls must be finite and in [0, 1]")
        active, categories = self.masks(choices)
        return PilotTarget(tuple(v if enabled else default for v, default, enabled in zip(values, self.defaults, active, strict=True)),
                           tuple(i if enabled else 0 for i, enabled in zip(choices, categories, strict=True)))

    def from_basic(self, target):
        return self.canonical(PilotTarget((*target.continuous, *self.defaults[9:]),
                                         (*target.categorical, *([0] * (len(self.categorical) - 2)))))

    def decode(self, target):
        target = self.canonical(target)
        raw = self.basic.decode(PilotTarget(target.continuous[:9], target.categorical[:2]))
        if not self.expanded or not any(target.categorical[2:]):
            return raw
        document = json.loads(raw)
        settings = document["settings"]
        wave1, response, osc2, wave2, octave, wet = target.categorical[2:]
        if wave1:
            settings["wavetables"][0] = self.waves[wave1]
        if response:
            settings["filter_1_blend"] = FILTER_BLEND[response]
        if osc2:
            settings.update(osc_2_on=1., osc_2_destination=0., osc_2_transpose=float(self.categorical[6][1][octave]))
            settings["wavetables"][1] = self.waves[wave2]
        if wet:
            settings["reverb_on"] = 1.
        active, _ = self.masks(target.categorical)
        for value, (name, lower, upper, _), enabled in zip(target.continuous[9:], EXTRA_CONTINUOUS, active[9:], strict=True):
            if enabled:
                settings[name] = lower + value * (upper - lower)
        return canonical_json(document).encode()

    def legal_choices(self):
        return tuple(choices for choices in product(*(range(len(options)) for _, options in self.categorical))
                     if all(active or index == 0 for index, active in zip(choices, self.masks(choices)[1], strict=True)))

    def category_vector(self, choices):
        return [value for index, (_, options) in zip(choices, self.categorical, strict=True)
                for value in ([float(index)] if len(options) == 2 else [float(index == i) for i in range(len(options))])]
