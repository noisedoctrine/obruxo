from __future__ import annotations

import json
import math
from typing import Any

from obruxo_data.errors import Diagnostic, Severity, ValidationReport
from obruxo_data.hashing import canonical_sha256

from .atlas import VitalSchema


CLASSIFIED_RUNTIME_CANONICALIZATIONS = frozenset({"/settings/sample/samples"})


def _error(code: str, message: str, *, pointer: str | None = None, parameter: str | None = None,
           context: dict[str, Any] | None = None) -> Diagnostic:
    return Diagnostic(code, Severity.ERROR, message, pointer=pointer, parameter=parameter, context=context or {})


def _validate_line(value: Any, pointer: str, diagnostics: list[Diagnostic]) -> None:
    if not isinstance(value, dict):
        diagnostics.append(_error("vital.line.type", "line state must be an object", pointer=pointer))
        return
    count = value.get("num_points")
    if not isinstance(count, int) or isinstance(count, bool) or not 1 <= count <= 100:
        diagnostics.append(_error("vital.line.point_count", "num_points must be an integer from 1 to 100", pointer=f"{pointer}/num_points"))
        return
    points = value.get("points")
    powers = value.get("powers")
    if not isinstance(points, list) or len(points) != count * 2:
        diagnostics.append(_error("vital.line.points", "points must contain exactly 2 * num_points numbers", pointer=f"{pointer}/points"))
    if not isinstance(powers, list) or len(powers) != count:
        diagnostics.append(_error("vital.line.powers", "powers must contain exactly num_points numbers", pointer=f"{pointer}/powers"))


def validate_document(document: Any, schema: VitalSchema) -> ValidationReport:
    diagnostics: list[Diagnostic] = []
    if not isinstance(document, dict):
        return ValidationReport((_error("vital.document.type", "preset must be a JSON object", pointer=""),))
    settings = document.get("settings")
    if not isinstance(settings, dict):
        return ValidationReport((_error("vital.settings.type", "settings must be an object", pointer="/settings"),))

    expected_nested = {"sample": dict, "modulations": list, "wavetables": list, "lfos": list}
    for key, expected_type in expected_nested.items():
        if not isinstance(settings.get(key), expected_type):
            diagnostics.append(_error("vital.settings.nested_type", f"{key} must be a {expected_type.__name__}", pointer=f"/settings/{key}"))
    if diagnostics:
        return ValidationReport(tuple(diagnostics))

    if len(settings["wavetables"]) != 3:
        diagnostics.append(_error("vital.wavetables.count", "exactly three wavetables are required", pointer="/settings/wavetables"))
    if len(settings["lfos"]) != 8:
        diagnostics.append(_error("vital.lfos.count", "exactly eight LFO shapes are required", pointer="/settings/lfos"))
    if len(settings["modulations"]) != 64:
        diagnostics.append(_error("vital.modulations.count", "exactly 64 modulation connection objects are required", pointer="/settings/modulations"))

    parameters = schema.parameters
    scalar_names = {name for name in settings if name not in expected_nested}
    for name in sorted(parameters.keys() - scalar_names):
        diagnostics.append(_error("vital.parameter.missing", "required scalar parameter is missing", parameter=name, pointer=f"/settings/{name}"))
    for name in sorted(scalar_names - parameters.keys()):
        diagnostics.append(_error("vital.parameter.unknown", "unknown scalar parameter", parameter=name, pointer=f"/settings/{name}"))
    for name in sorted(scalar_names & parameters.keys()):
        message = parameters[name].validate_raw(settings[name])
        if message:
            diagnostics.append(_error("vital.parameter.invalid", f"{name} {message}", parameter=name, pointer=f"/settings/{name}"))

    for index, lfo in enumerate(settings["lfos"]):
        _validate_line(lfo, f"/settings/lfos/{index}", diagnostics)
    sources = schema.modulation_sources
    destinations = schema.modulation_destinations
    for index, connection in enumerate(settings["modulations"]):
        pointer = f"/settings/modulations/{index}"
        if not isinstance(connection, dict):
            diagnostics.append(_error("vital.modulation.type", "modulation connection must be an object", pointer=pointer))
            continue
        source = connection.get("source", "")
        destination = connection.get("destination", "")
        if not isinstance(source, str) or not isinstance(destination, str):
            diagnostics.append(_error("vital.modulation.identity_type", "source and destination must be strings", pointer=pointer))
            continue
        if bool(source) != bool(destination):
            diagnostics.append(_error("vital.modulation.dangling", "source and destination must both be empty or both be set", pointer=pointer))
        if source and source not in sources:
            diagnostics.append(_error("vital.modulation.source", "unknown modulation source", pointer=f"{pointer}/source", context={"value": source}))
        if destination and destination not in destinations:
            diagnostics.append(_error("vital.modulation.destination", "unknown modulation destination", pointer=f"{pointer}/destination", context={"value": destination}))
        if "line_mapping" in connection:
            _validate_line(connection["line_mapping"], f"{pointer}/line_mapping", diagnostics)

    def walk(value: Any, pointer: str = "") -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                walk(child, f"{pointer}/{key}")
        elif isinstance(value, list):
            for index, child in enumerate(value):
                walk(child, f"{pointer}/{index}")
        elif isinstance(value, float) and not math.isfinite(value):
            diagnostics.append(_error("vital.number.non_finite", "all numeric values must be finite", pointer=pointer))

    walk(document)
    return ValidationReport(tuple(diagnostics))


def _difference_pointers(left: Any, right: Any, pointer: str = "") -> set[str]:
    if type(left) is not type(right):
        return {pointer}
    if isinstance(left, dict):
        differences = {f"{pointer}/{key}" for key in left.keys() ^ right.keys()}
        for key in left.keys() & right.keys():
            escaped = str(key).replace("~", "~0").replace("/", "~1")
            differences.update(_difference_pointers(left[key], right[key], f"{pointer}/{escaped}"))
        return differences
    if isinstance(left, list):
        if len(left) != len(right):
            return {pointer}
        differences = set()
        for index, (left_item, right_item) in enumerate(zip(left, right, strict=True)):
            differences.update(_difference_pointers(left_item, right_item, f"{pointer}/{index}"))
        return differences
    return set() if left == right else {pointer}


def _pointer_value(document: Any, pointer: str) -> Any:
    value = document
    for raw_token in pointer.lstrip("/").split("/"):
        token = raw_token.replace("~1", "/").replace("~0", "~")
        value = value[int(token)] if isinstance(value, list) else value[token]
    return value


def validate_runtime(document_json: str) -> ValidationReport:
    try:
        import vita
    except ImportError as error:
        return ValidationReport((_error("vital.runtime.unavailable", "Vita is not installed", context={"error": str(error)}),))
    try:
        synth = vita.Synth()
        if not synth.load_json(document_json):
            return ValidationReport((_error("vital.runtime.load", "Vita rejected the preset JSON"),))
        round_trip = synth.to_json()
    except Exception as error:
        return ValidationReport((_error("vital.runtime.exception", "Vita runtime validation failed", context={"error": str(error)}),))
    if not round_trip:
        return ValidationReport((_error("vital.runtime.export", "Vita returned an empty round-trip document"),))
    try:
        source = json.loads(document_json)
        exported = json.loads(round_trip)
    except json.JSONDecodeError as error:
        return ValidationReport((_error("vital.runtime.export_json", "Vita returned invalid JSON", context={"error": str(error)}),))
    differences = _difference_pointers(source, exported)
    numeric_canonicalizations = set()
    for pointer in differences - CLASSIFIED_RUNTIME_CANONICALIZATIONS:
        left = _pointer_value(source, pointer)
        right = _pointer_value(exported, pointer)
        if (
            isinstance(left, (int, float)) and not isinstance(left, bool)
            and isinstance(right, (int, float)) and not isinstance(right, bool)
            and math.isclose(float(left), float(right), rel_tol=1e-6, abs_tol=1e-7)
        ):
            numeric_canonicalizations.add(pointer)
    unexplained = sorted(differences - CLASSIFIED_RUNTIME_CANONICALIZATIONS - numeric_canonicalizations)
    diagnostics = []
    if unexplained:
        diagnostics.append(_error(
            "vital.runtime.unclassified_drift", "Vita changed unclassified fields during canonical round trip",
            context={"pointers": unexplained},
        ))
    for pointer in sorted(differences & CLASSIFIED_RUNTIME_CANONICALIZATIONS):
        diagnostics.append(Diagnostic(
            "vital.runtime.canonicalization", Severity.WARNING,
            "Vita re-encoded the deterministic init sampler payload during round trip",
            pointer=pointer,
            context={
                "input_sha256": canonical_sha256(source["settings"]["sample"]["samples"]),
                "output_sha256": canonical_sha256(exported["settings"]["sample"]["samples"]),
            },
        ))
    if numeric_canonicalizations:
        diagnostics.append(Diagnostic(
            "vital.runtime.numeric_canonicalization", Severity.WARNING,
            "Vita quantized numeric controls within the reviewed float32 round-trip tolerance",
            context={"pointers": sorted(numeric_canonicalizations), "relative_tolerance": 1e-6, "absolute_tolerance": 1e-7},
        ))
    return ValidationReport(tuple(diagnostics))
