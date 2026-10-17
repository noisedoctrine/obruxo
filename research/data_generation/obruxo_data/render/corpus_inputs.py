"""Declared corpus-only native readback and ignored-empty-map handling."""
from copy import deepcopy
import hashlib
import math
from pathlib import Path

import numpy as np

from obruxo_data.vital.atlas import ParameterSpec


def corpus_native_policy():
    atlas = Path(__file__).parents[1] / "vital/atlas.py"
    return {"policy": "exact_normalized_float32_scalar_readback_and_ignored_empty_maps_v1",
            "source_sha256_lf": {"render/corpus_inputs.py": hashlib.sha256(Path(__file__).read_bytes().replace(b"\r\n", b"\n")).hexdigest(),
                                 "vital/atlas.py": hashlib.sha256(atlas.read_bytes().replace(b"\r\n", b"\n")).hexdigest()},
            "scalar_input_changes": False, "ordinary_relative_tolerance": 1e-6, "ordinary_absolute_tolerance": 1e-7,
            "extra_continuous_acceptance": "loaded_equals_exact_declared_normalized_float32_roundtrip",
            "discrete_acceptance": "exact_requested_ordinal", "empty_connection_map": "omit_before_curve_import_only_when_source_and_destination_empty"}


def prepare_native_corpus(document):
    """Only remove a mapping that native Vital ignores on an empty connection."""
    prepared, changes = deepcopy(document), []
    for slot, route in enumerate(prepared["settings"]["modulations"]):
        if not route["source"] and not route["destination"] and "line_mapping" in route:
            del route["line_mapping"]
            changes.append({"path": f"/settings/modulations/{slot}/line_mapping", "reason": "native_ignores_empty_connection_mapping"})
    return prepared, changes


def native_scalar_grid(name, value, schema):
    """Match the existing full scalar encoder's raw/normalized float32 round trip."""
    raw, baseline = schema.parameters[name], schema.baseline_parameters.get(name, {})
    discrete = raw["is_discrete"] or baseline.get("is_discrete", False) or baseline.get("scale") == "Indexed"
    if discrete or name == "effect_chain_order" or name.endswith("_transpose_quantize"):
        return None
    metadata = {**baseline, "min": raw["min"], "max": raw["max"], "default": value, "is_discrete": False}
    spec = ParameterSpec.from_dict(name, {"scale": "Linear", **metadata})
    position = spec.normalized_from_raw(value)
    requested = np.float32(spec.raw_from_normalized(position))
    minimum, maximum = np.float32(spec.minimum), np.float32(spec.maximum)
    span = np.float32(maximum - minimum)
    normalized = np.float32(np.float32(requested - minimum) / span)
    return min(spec.maximum, max(spec.minimum, float(np.float32(minimum + np.float32(normalized * span)))))


def verify_native_corpus_controls(requested, loaded, schema):
    expected, actual = requested["settings"], loaded.get("settings", {})
    mismatches, roundtrips = [], []
    for name in sorted(schema.scalar_names):
        value, observed = expected[name], actual.get(name)
        if type(observed) not in (int, float) or not math.isfinite(observed):
            mismatches.append(name)
            continue
        grid = native_scalar_grid(name, value, schema)
        if grid is None:
            if observed != value:
                mismatches.append(name)
            continue
        if math.isclose(value, observed, rel_tol=1e-6, abs_tol=1e-7):
            continue
        if observed != grid:
            mismatches.append(name)
            continue
        roundtrips.append({"name": name, "requested": value, "loaded": observed, "exact_native_grid": grid})
    if mismatches:
        raise RuntimeError("Vital did not load the requested scalar controls: " + ", ".join(mismatches[:8]))
    for name in ("lfos", "modulations"):
        if expected.get(name) != actual.get(name):
            raise RuntimeError(f"Vital did not load the requested {name}")
    return len(schema.scalar_names), roundtrips
