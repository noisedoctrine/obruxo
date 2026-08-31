"""Strict 1.5.5 preset validation and deterministic scalar-only authoring.

Structural conformance, conservative authoring bounds, and native acceptance are
separate claims. This module does not claim to emulate Vital's DSP or loader.
"""
from __future__ import annotations

import base64
import binascii
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data_generation"))
from obruxo_data.errors import Diagnostic, Severity, ValidationReport  # noqa: E402

DIRECTORY = Path(__file__).with_name("contract_1_5_5")
BASE = Path(__file__).resolve().parents[1] / "data_generation/obruxo_data/vital/schema/vital-1.0.8-vita-0.1.0"


def strict_json(payload):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    def reject_constant(value):
        raise ValueError(f"nonstandard JSON number: {value}")

    return json.loads(payload, object_pairs_hook=pairs, parse_constant=reject_constant)


def json_digest(payload):
    """Pin JSON content independently of Windows/Linux checkout line endings."""
    canonical = json.dumps(strict_json(payload), sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def number(value):
    try:
        return type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        return False


def pointer(parent, key):
    return parent + "/" + str(key).replace("~", "~0").replace("/", "~1")


class FullPresetContract:
    def __init__(self):
        self.spec = strict_json((DIRECTORY / "contract.json").read_bytes())
        raw = (BASE / "parameter_inventory.json").read_bytes()
        if json_digest(raw) != self.spec["baseline_inventory_sha256"]:
            raise ValueError("baseline scalar metadata hash changed")
        self.baseline_parameters = json.loads(raw)["parameters"]
        measured = (DIRECTORY / "renderer_ranges.json").read_bytes()
        if json_digest(measured) != self.spec["renderer_ranges_sha256"]:
            raise ValueError("renderer scalar range hash changed")
        self.parameters = json.loads(measured)["parameters"]
        self.scalar_names = frozenset(self.spec["scalar_names"])
        vocab_raw = (BASE / "modulation_vocab.json").read_bytes()
        if json_digest(vocab_raw) != self.spec["modulation_vocab_sha256"]:
            raise ValueError("modulation vocabulary hash changed")
        vocab = json.loads(vocab_raw)
        self.sources = frozenset(vocab["sources"])
        self.destinations = frozenset(vocab["destinations"]) | (self.scalar_names - self.baseline_parameters.keys())

    def template(self):
        raw = (DIRECTORY / "init.vital").read_bytes()
        if hashlib.sha256(raw).hexdigest() != self.spec["template_sha256"]:
            raise ValueError("factory template hash changed")
        return strict_json(raw)

    def validate(self, document, *, authoring=True):
        diagnostics = []

        def error(code, message, path):
            if len(diagnostics) < 100:
                diagnostics.append(Diagnostic(code, Severity.ERROR, message, pointer=path))

        # Validate Python objects too: bools are not numbers, cycles/deep objects fail closed.
        def finite(value, path="", depth=0):
            if depth > 48:
                error("contract.depth", "maximum nesting depth exceeded", path)
                return
            if len(diagnostics) >= 100:
                return
            if type(value) in (int, float) and not number(value):
                error("contract.finite", "number must be finite and representable", path)
            elif isinstance(value, dict):
                for key, child in value.items():
                    if not isinstance(key, str):
                        error("contract.key", "JSON keys must be strings", path)
                    finite(child, pointer(path, key), depth + 1)
            elif isinstance(value, list):
                for index, child in enumerate(value):
                    finite(child, pointer(path, index), depth + 1)
            elif value is not None and type(value) not in (int, float, bool, str):
                error("contract.json_type", "not a JSON value", path)

        finite(document)
        if diagnostics:
            return ValidationReport(tuple(diagnostics))
        if not isinstance(document, dict) or not isinstance(document.get("settings"), dict):
            error("contract.document", "expected a preset object with a settings object", "")
            return ValidationReport(tuple(diagnostics))
        if document.get("synth_version") != "1.5.5":
            error("contract.version", "this contract requires synth_version 1.5.5; no silent relabeling", "/synth_version")
        settings = document["settings"]
        for name in sorted(self.scalar_names):
            value = settings.get(name)
            path = f"/settings/{name}"
            if not number(value):
                error("contract.scalar", "required scalar must be a finite JSON number", path)
                continue
        if authoring:
            diagnostics.extend(self.validate_ranges(document).diagnostics[:max(0, 100 - len(diagnostics))])
        # Ramps are not even optional model fields. A 1.5.5 export omits them;
        # the reviewed newer renderer supplies -10 for all 128 fields.
        for name in settings.keys() & self.spec["fixed_native_ramps"].keys():
            error("contract.ramp", "ramp controls are excluded from the 1.5.5 model/output contract", f"/settings/{name}")
        nested = {**document, "settings": {k: v for k, v in settings.items() if k not in self.scalar_names}}

        def shape(value, schema, path):
            if len(diagnostics) >= 100:
                return
            if "variants" in schema:
                variant = value.get("type") if isinstance(value, dict) else None
                if not isinstance(variant, str) or variant not in schema["variants"]:
                    error("contract.component", "unknown wavetable component type", path)
                    return
                shape({k: v for k, v in value.items() if k != "type"}, schema["variants"][variant], path)
                return
            tag = "boolean" if isinstance(value, bool) else "number" if number(value) else {
                dict: "object", list: "array", str: "string", type(None): "null"}.get(type(value))
            if tag not in schema.get("types", []):
                error("contract.type", f"expected {schema.get('types', [])}", path)
                return
            if tag == "object":
                properties = schema.get("properties", {})
                for key in sorted(set(schema.get("required", [])) - value.keys()):
                    error("contract.missing", "required field missing", pointer(path, key))
                for key in sorted(value.keys() - properties.keys()):
                    error("contract.unknown", "field outside the pinned schema", pointer(path, key))
                for key in value.keys() & properties.keys():
                    shape(value[key], properties[key], pointer(path, key))
            elif tag == "array":
                if len(value) > 65536:
                    error("contract.array_limit", "array exceeds authoring safety limit", path)
                    return
                for index, child in enumerate(value):
                    shape(child, schema.get("items", {}), pointer(path, index))

        shape(nested, self.spec["shape"], "")
        if diagnostics:
            return ValidationReport(tuple(diagnostics))

        for key, count in {"wavetables": 3, "lfos": 8, "custom_warps": 3, "random_values": 3, "modulations": 64}.items():
            if len(settings[key]) != count:
                error("contract.count", f"requires exactly {count} entries", f"/settings/{key}")
        for index, connection in enumerate(settings["modulations"]):
            path = f"/settings/modulations/{index}"
            source, destination = connection["source"], connection["destination"]
            if bool(source) != bool(destination):
                error("contract.route", "source and destination must both be set or both empty", path)
            if source and source not in self.sources:
                error("contract.source", "unknown modulation source", path + "/source")
            if destination and destination not in self.destinations:
                error("contract.destination", "unknown or excluded modulation destination", path + "/destination")

        def payload(text, path):
            try:
                return base64.b64decode(text, validate=True)
            except (ValueError, binascii.Error):
                error("contract.base64", "invalid base64 payload", path)
                return None

        def relationships(value, path):
            if len(diagnostics) >= 100:
                return
            if isinstance(value, list):
                for index, child in enumerate(value):
                    relationships(child, pointer(path, index))
                return
            if not isinstance(value, dict):
                return
            if {"points", "powers", "num_points"} <= value.keys():
                count = value["num_points"]
                if not number(count) or not float(count).is_integer() or not 1 <= count <= 100:
                    error("contract.line_count", "line requires 1..100 integral points", path)
                elif len(value["points"]) != 2 * count or len(value["powers"]) != count:
                    error("contract.line_length", "points/powers lengths disagree with num_points", path)
                points = value["points"]
                if any(not 0 <= v <= 1 for v in points) or points[::2] != sorted(points[::2]):
                    error("contract.line_points", "line coordinates must be in [0,1] with ordered x positions", path)
            if "seed" in value and (not float(value["seed"]).is_integer() or not 0 <= value["seed"] <= 4294967295):
                error("contract.seed", "seed must be an unsigned 32-bit integer", path)
            if "keyframes" in value:
                for key, upper in {"interpolation_style": 2, "interpolation": 1, "fade_style": 3, "phase_style": 2}.items():
                    if key in value and (not float(value[key]).is_integer() or not 0 <= value[key] <= upper):
                        error("contract.component_enum", f"{key} requires an ordinal in [0,{upper}]", path)
                frames = value["keyframes"]
                if not isinstance(frames, list) or not frames:
                    error("contract.keyframes", "component must contain keyframes; null/empty components are not authorable", path)
                else:
                    positions = [frame["position"] for frame in frames]
                    if any(not float(p).is_integer() or not 0 <= p <= 256 for p in positions) or positions != sorted(set(positions)):
                        error("contract.keyframe_positions", "keyframes require ordered, unique integer positions in [0,256]", path)
            if "wave_data" in value:
                raw = payload(value["wave_data"], path + "/wave_data")
                if raw is not None and (len(raw) != 8192 or not all(math.isfinite(x[0]) for x in struct.iter_unpack("<f", raw))):
                    error("contract.wave_data", "wave_data must encode 2048 finite float32 samples", path)
            if "audio_file" in value:
                raw = payload(value["audio_file"], path + "/audio_file")
                # FileSource serializes raw PCM16, not a WAV/FLAC container.
                if raw is not None and (not raw or len(raw) % 2):
                    error("contract.audio_file", "audio_file requires a nonempty PCM16 byte payload", path)
                if not 0 < value["audio_sample_rate"] <= 768000 or value["window_size"] <= 0:
                    error("contract.audio_window", "audio sample rate and window size must be positive", path)
            for key, child in value.items():
                relationships(child, pointer(path, key))

        relationships(settings, "/settings")
        sample = settings["sample"]
        length = sample["length"]
        if not float(length).is_integer() or not 0 <= length <= 16_777_216:
            error("contract.sample_length", "invalid sample length", "/settings/sample/length")
        if not 0 < sample["sample_rate"] <= 768000:
            error("contract.sample_rate", "sample rate outside authoring safety bounds", "/settings/sample/sample_rate")
        for key in ("samples", "samples_stereo"):
            if key in sample:
                raw = payload(sample[key], f"/settings/sample/{key}")
                if raw is not None and len(raw) != length * 2:
                    error("contract.sample_payload", "PCM16 byte length disagrees with sample length", f"/settings/sample/{key}")
        for index, wavetable in enumerate(settings["wavetables"]):
            if wavetable["version"] not in self.spec["wavetable_versions_observed"]:
                error("contract.wavetable_version", "unreviewed wavetable version", f"/settings/wavetables/{index}/version")
            if not wavetable["groups"] or any(not group["components"] for group in wavetable["groups"]):
                error("contract.wavetable_empty", "wavetable needs nonempty groups and components", f"/settings/wavetables/{index}")
        return ValidationReport(tuple(diagnostics))

    def validate_ranges(self, document):
        """Cheap scalar-only pass for audits that already checked structure."""
        diagnostics = []
        for name, metadata in self.parameters.items():
            value = document["settings"].get(name)
            if not number(value):
                continue  # Structural validation reports malformed/missing numbers.
            lo, hi = metadata["min"], metadata["max"]
            tolerance = max(1e-7, abs(lo) * 1e-6, abs(hi) * 1e-6)
            if not lo - tolerance <= value <= hi + tolerance:
                diagnostics.append(Diagnostic("contract.range", Severity.ERROR, f"outside renderer authoring range [{lo}, {hi}]", pointer=f"/settings/{name}"))
            if metadata["is_discrete"] and not float(value).is_integer():
                diagnostics.append(Diagnostic("contract.ordinal", Severity.ERROR, "categorical control requires an integral ordinal", pointer=f"/settings/{name}"))
        return ValidationReport(tuple(diagnostics))

    def read(self, path, *, authoring=True):
        try:
            with Path(path).open("rb") as stream:
                raw = stream.read(self.spec["max_file_bytes"] + 1)
            if len(raw) > self.spec["max_file_bytes"]:
                raise ValueError("preset exceeds the 64 MiB file safety limit")
            document = strict_json(raw.decode("utf-8-sig"))
        except (OSError, ValueError, RecursionError) as exc:
            return None, ValidationReport((Diagnostic("contract.json", Severity.ERROR, str(exc)),))
        return document, self.validate(document, authoring=authoring)

    def decode_scalars(self, raw_controls):
        """Apply named raw predictions to factory assets; reject all unknown fields."""
        if not isinstance(raw_controls, dict) or raw_controls.keys() - self.scalar_names:
            raise ValueError("predictions must only contain 1.5.5 scalar controls; ramps are excluded")
        document = self.template()
        document["settings"].update(deepcopy(raw_controls))
        self.validate(document).require_valid()
        return document

    def save(self, document, destination):
        """Validate exact serialized bytes and publish atomically without overwriting."""
        self.validate(document).require_valid()
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            raise FileExistsError(destination)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile("wb", dir=destination.parent, delete=False) as stream:
                temporary = Path(stream.name)
                stream.write((json.dumps(document, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode())
                stream.flush()
                os.fsync(stream.fileno())
            self.read(temporary)[1].require_valid()
            os.link(temporary, destination)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
