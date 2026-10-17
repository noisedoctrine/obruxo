"""Readback rules for the full 1.5.5 inventory on the reviewed Vital 1.6.4."""
from copy import deepcopy
import base64

import numpy as np


NESTED_FAMILIES = ("lfos", "custom_warps", "random_values", "modulations", "wavetables", "sample")


def _verify_sample(expected, actual):
    if not isinstance(actual, dict) or expected.keys() != actual.keys():
        raise RuntimeError("Vital did not load the requested sample structure")
    payloads = {"samples", "samples_stereo"} & expected.keys()
    if any(expected[key] != actual[key] for key in expected.keys() - payloads):
        raise RuntimeError("Vital did not load the requested sample metadata")
    deltas = {}
    for key in sorted(payloads):
        try:
            before = np.frombuffer(base64.b64decode(expected[key], validate=True), dtype="<i2").astype(np.int32)
            after = np.frombuffer(base64.b64decode(actual[key], validate=True), dtype="<i2").astype(np.int32)
        except (ValueError, TypeError) as error:
            raise RuntimeError("Vital returned an invalid sample payload") from error
        if before.shape != after.shape or np.any(after[:4]):
            raise RuntimeError("Vital sample readback differs from the reviewed four-zero prefix")
        # Native serialization includes its four-sample prefix and omits the final four.
        delta = int(np.max(np.abs(before[:-4] - after[4:]))) if len(before) > 4 else 0
        if delta > 1:
            raise RuntimeError("Vital sample readback exceeds one PCM16 unit after four-sample alignment")
        deltas[key] = delta
    return deltas


def verify_full_nested_state(requested, loaded, ramps, *, signed_seed_bits=False):
    expected, actual = requested["settings"], loaded.get("settings", {})
    for name in ("lfos", "custom_warps", "modulations"):
        if expected[name] != actual.get(name):
            raise RuntimeError(f"Vital did not load the requested {name}")
    seeds = actual.get("random_values")
    converted = 0
    if signed_seed_bits and isinstance(seeds, list):
        seeds = deepcopy(seeds)
        for value in seeds:
            seed = value.get("seed") if isinstance(value, dict) else None
            if type(seed) is int and -2147483648 <= seed < 0:
                value["seed"] = seed + 2**32
                converted += 1
    if expected["random_values"] != seeds:
        raise RuntimeError("Vital did not load the requested random_values")
    waves = actual.get("wavetables")
    if not isinstance(waves, list) or len(waves) != 3 or any(not isinstance(wave, dict) for wave in waves):
        raise RuntimeError("Vital did not load the requested wavetables")
    expected_waves = deepcopy(expected["wavetables"])
    for wave in expected_waves:
        wave["version"] = "1.6.4"
    if expected_waves != waves:
        raise RuntimeError("Vital wavetable readback has unreviewed drift beyond version migration")
    deltas = _verify_sample(expected["sample"], actual.get("sample"))
    if any(actual.get(name) != value for name, value in ramps.items()):
        raise RuntimeError("Vital did not supply the reviewed modulation ramp defaults")
    return {"nested_families_verified": list(NESTED_FAMILIES), "modulation_slots_verified": 64,
            **({"seed_readback_policy": "signed_int32_to_uint32_exact_bits", "signed_seed_slots_converted": converted} if signed_seed_bits else {}),
            "native_ramp_defaults_verified": len(ramps), "wavetable_version_migration": "1.6.4",
            "sample_readback_shift_samples": 4, "sample_aligned_max_pcm16_delta": deltas,
            "sample_readback_byte_exact": False}
