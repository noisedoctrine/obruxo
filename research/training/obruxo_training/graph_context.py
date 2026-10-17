"""Declared prerequisite settings for connection destinations; native response is measured separately."""
from copy import deepcopy
import re


def moving_curve(name):
    return {"name": name, "num_points": 3, "points": [0., 1., .5, 0., 1., 1.], "powers": [0., 0., 0.], "smooth": False}


def prepare_destination_context(settings, destination, waves, samples):
    wave = next((value for value in waves if value.get("name") == "Basic Shapes"), waves[min(1, len(waves) - 1)])
    sample = next((value for value in samples[1:] if value.get("length", 0) > 4), samples[-1])
    module = re.match(r"(lfo_[1-8]|random_[1-4]|env_[1-6])_", destination)
    if module:
        prefix = module[1]
        if prefix.startswith("lfo_"):
            settings["lfos"][int(prefix[-1]) - 1] = moving_curve("OBRUXO destination context")
            settings[prefix + "_sync_type"] = 0.
        if not prefix.startswith("env_"):
            settings[prefix + "_sync"] = 4. if "_keytrack_" in destination else 1. if destination.endswith("_tempo") else 0.
            settings[prefix + "_frequency"] = -.5
            settings[prefix + "_tempo"] = 7.
    if destination.startswith("modulation_"):
        settings["lfos"][1] = moving_curve("OBRUXO referenced connection")
        settings.update(lfo_2_sync=0., lfo_2_frequency=-.5, lfo_2_delay_time=0., lfo_2_fade_time=0.)
    oscillator = re.match(r"osc_([123])_", destination)
    if oscillator:
        slot = int(oscillator[1])
        settings.update({f"osc_{slot}_on": 1., f"osc_{slot}_destination": 4., f"osc_{slot}_unison_voices": 2.})
        settings["wavetables"][slot - 1] = deepcopy(wave)
        settings[f"osc_{slot}_wave_frame"] = 128.
        if "_distortion_" in destination:
            settings[f"osc_{slot}_distortion_type"] = 1.
            settings[f"osc_{slot}_distortion_amount"] = .3
        if "_spectral_morph_" in destination:
            settings[f"osc_{slot}_spectral_morph_type"] = 1.
            settings[f"osc_{slot}_spectral_morph_amount"] = .4
    if destination.startswith("sample_"):
        settings.update(sample_on=1., sample_keytrack=1., sample_loop=1.)
        settings["sample"] = deepcopy(sample)
    filter_match = re.match(r"filter_(1|2|fx)_", destination)
    if filter_match:
        prefix = "filter_" + filter_match[1]
        settings[prefix + "_on"] = 1.
        settings[prefix + "_cutoff"] = 69.
        settings[prefix + "_mix"] = .8
        if "_formant_" in destination:
            settings[prefix + "_model"] = 5.
        settings["osc_1_destination"] = 3. if filter_match[1] == "fx" else float(int(filter_match[1]) - 1)
    if destination == "beats_per_minute":
        settings.update(lfo_1_sync=1., lfo_1_tempo=8.)
        settings["lfos"][0] = moving_curve("OBRUXO tempo context")
    if destination.endswith("_tempo") and not module:
        sync = destination.removesuffix("_tempo") + "_sync"
        if sync in settings:
            settings[sync] = 1.
    if destination.endswith("_frequency") and not module:
        sync = destination.removesuffix("_frequency") + "_sync"
        if sync in settings:
            settings[sync] = 0.
