"""Read-only separation of level error and level-matched sound error."""

from collections import defaultdict
from statistics import mean
import math

import numpy as np
import torch

from .evaluate import audio_distance
from .listening_loudness import POLICY
from .spectral import SpectralTargets, spectral_loss

ARMS = ("initial", "parameter", "spectral")


def match_recordings(audio, meter, independent):
    """Use one shared headroom target for reference and all three predictions."""
    if set(audio) != {"reference", *ARMS} or len({value.shape for value in audio.values()}) != 1:
        raise ValueError("level diagnosis requires four complete aligned recordings")
    before = {name: meter.measure(value) for name, value in audio.items()}
    gains = {name: POLICY["target_lufs"] - row["integrated_lufs"] for name, row in before.items()}
    highest = max(before[name]["true_peak_upper_dbtp"] + gain for name, gain in gains.items())
    common = min(0., POLICY["true_peak_ceiling_dbtp"] - highest - .05)
    rounds = []
    for iteration in range(4):
        matched = {name: (value.astype(np.float64) * 10**((gains[name] + common) / 20)).astype(np.float32) for name, value in audio.items()}
        after = {name: meter.measure(value) for name, value in matched.items()}
        levels = [row["integrated_lufs"] for row in after.values()]
        peak = max(row["true_peak_upper_dbtp"] for row in after.values())
        rounds.append({"round": iteration + 1, "individual_gain_db": dict(gains), "common_gain_db": common,
                       "levels_lufs": {name: row["integrated_lufs"] for name, row in after.items()}, "highest_true_peak_upper_dbtp": peak})
        if max(levels) - min(levels) <= .1 and peak <= -1.:
            break
        if iteration < 3:
            # Absolute gating can change when a quiet clip is amplified. Solve
            # the final static gain; every pass still starts from original samples.
            target = POLICY["target_lufs"] + common
            correction = {name: target - row["integrated_lufs"] for name, row in after.items()}
            gains = {name: gains[name] + correction[name] for name in gains}
            common += min(0., -1. - max(after[name]["true_peak_upper_dbtp"] + correction[name] for name in gains) - .05)
    checked = {name: {"before_lufs": float(independent(value.astype(np.float64))),
                      "after_lufs": float(independent(matched[name].astype(np.float64)))} for name, value in audio.items()}
    disagreement = max(abs(checked[name][stage + "_lufs"] - readings[name]["integrated_lufs"])
                       for name in audio for stage, readings in (("before", before), ("after", after)))
    levels = [row["integrated_lufs"] for row in after.values()]
    independent_levels = [row["after_lufs"] for row in checked.values()]
    spread = max(levels) - min(levels)
    independent_spread = max(independent_levels) - min(independent_levels)
    if (not all(math.isfinite(value) for value in (*levels, *independent_levels, disagreement)) or disagreement > .2
            or spread > .1 or independent_spread > .1 or any(row["true_peak_upper_dbtp"] > -1. for row in after.values())):
        raise ValueError("level diagnosis misses the frozen meter, matching or headroom tolerance")
    return matched, {"before": before, "after": after, "independent": checked, "individual_gain_db": gains, "common_gain_db": common,
                     "spread_lu": spread, "independent_spread_lu": independent_spread, "meter_disagreement_lu": disagreement, "rounds": rounds}


def matched_scores(audio):
    measure = SpectralTargets()
    reference = audio["reference"]
    targets = measure(torch.from_numpy(reference.copy()).T[None])
    return {arm: {"spectral": float(spectral_loss(measure(torch.from_numpy(audio[arm].copy()).T[None]), targets)["loss"]),
                  "mrstft": audio_distance(reference, audio[arm])["mrstft"]} for arm in ARMS}


def summarize(rows):
    groups = defaultdict(lambda: defaultdict(list))
    for row in rows:
        groups[row["group"]][row["state_id"]].append(row)
    result = {}
    for group, families in groups.items():
        arms = {}
        for arm in ARMS:
            per_family = []
            for observations in families.values():
                def average(key):
                    if key == "absolute_loudness_error_lu":
                        return mean(abs(row["matching"]["before"][arm]["integrated_lufs"] - row["matching"]["before"]["reference"]["integrated_lufs"]) for row in observations)
                    condition, metric = key.split("_", 1)
                    return mean(row[condition][arm][metric] for row in observations)
                per_family.append({key: average(key) for key in ("raw_spectral", "raw_mrstft", "matched_spectral", "matched_mrstft", "absolute_loudness_error_lu")})
            arms[arm] = {key: mean(row[key] for row in per_family) for key in per_family[0]}
        ratios = {key: arms["spectral"][key] / arms["parameter"][key] if arms["parameter"][key] else None for key in arms["spectral"]}
        result[group] = {"independent_patches": len(families), "observations": sum(map(len, families.values())), "means": arms,
                         "sound_vs_parameter_ratios": ratios}
    return result
