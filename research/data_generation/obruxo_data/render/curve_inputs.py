"""Explicit native curve import: finite float32 coordinates and constant singleton lines."""
from copy import deepcopy
import hashlib
import math
from pathlib import Path
import struct

RULE = "float32_curves_singletons_to_constant_endpoints_v1"


def curve_input_policy():
    return {"rule": RULE, "source_sha256_lf": hashlib.sha256(Path(__file__).read_bytes().replace(b"\r\n", b"\n")).hexdigest(),
            "singleton": "constant_y_at_x_0_and_1_zero_powers", "precision": "finite_float32", "raw_source_preserved": True}


def float32(value):
    if type(value) not in {int, float} or not math.isfinite(value):
        raise ValueError("curve coordinates and powers must be finite numbers")
    try:
        result = struct.unpack("<f", struct.pack("<f", value))[0]
    except (OverflowError, struct.error) as exc:
        raise ValueError("curve values must fit finite float32") from exc
    if not math.isfinite(result):
        raise ValueError("curve values must fit finite float32")
    return result


def prepare_curve_inputs(document):
    """Return a separate applied document and a complete changed-curve ledger."""
    result, changes = deepcopy(document), []
    settings = result["settings"]
    curves = [(f"/settings/{family}/{index}", shape) for family in ("lfos", "custom_warps")
              for index, shape in enumerate(settings[family])]
    curves += [(f"/settings/modulations/{index}/line_mapping", route["line_mapping"])
               for index, route in enumerate(settings["modulations"]) if "line_mapping" in route]
    for path, shape in curves:
        before = deepcopy(shape)
        count = shape["num_points"]
        if type(count) not in {int, float} or not float(count).is_integer() or not 1 <= count <= 100:
            raise ValueError("curve requires 1..100 integral points")
        if len(shape["points"]) != 2 * count or len(shape["powers"]) != count:
            raise ValueError("curve lengths differ from num_points")
        if any(type(value) not in {int, float} or not 0 <= value <= 1 for value in shape["points"]) or shape["points"][::2] != sorted(shape["points"][::2]):
            raise ValueError("curve coordinates must be ordered and within [0,1]")
        shape["points"] = [float32(value) for value in shape["points"]]
        shape["powers"] = [float32(value) for value in shape["powers"]]
        if count == 1:
            y = shape["points"][1]
            shape.update(num_points=2, points=[0., y, 1., y], powers=[0., 0.])
        if shape != before:
            changes.append({"path": path, "singleton_expanded": count == 1, "before_points": count, "applied_points": shape["num_points"]})
    return result, changes
