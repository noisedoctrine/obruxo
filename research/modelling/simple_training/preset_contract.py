"""Strict validator for this experiment's output family, independent of model code."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "data_generation"))

from obruxo_data.errors import Diagnostic, Severity, ValidationReport  # noqa: E402
from obruxo_data.vital import ComponentProfile, VitalPreset  # noqa: E402


def strict_json(text: str):
    def pairs(items):
        document = {}
        for key, value in items:
            if key in document:
                raise ValueError(f"duplicate JSON key: {key}")
            document[key] = value
        return document

    def constant(value):
        raise ValueError(f"nonstandard JSON number: {value}")

    return json.loads(text, object_pairs_hook=pairs, parse_constant=constant)


class PresetContract:
    def __init__(self):
        self.spec = strict_json(Path(__file__).with_name("preset_contract.json").read_text(encoding="utf-8"))
        preset = VitalPreset.init()
        preset.apply_profile(ComponentProfile.only(oscillators=[1], max_active_routes=0))
        for name, value in {"osc_1_on": 1, "osc_1_destination": 4, "osc_1_unison_voices": 1,
                            "osc_1_random_phase": 0, "osc_1_transpose": 0}.items():
            preset.set_raw(name, value)
        document = preset.to_dict()
        document["synth_version"] = self.spec["preset_version"]
        for table in document["settings"]["wavetables"]:
            table["version"] = self.spec["preset_version"]
        self._template = VitalPreset(document, preset.schema)
        digest = hashlib.sha256(self._template.to_json(canonical=True).encode()).hexdigest()
        if digest != self.spec["template_sha256"] or preset.schema.schema_id != self.spec["schema_id"]:
            raise ValueError("preset template changed; review and version the output contract before exporting")
        self._template.validate().require_valid()
        self._expected = self._template.to_dict()

    @property
    def template(self) -> VitalPreset:
        return VitalPreset(self._expected, self._template.schema)

    def validate(self, document, *, runtime: bool = False) -> ValidationReport:
        diagnostics = []

        def reject(code, message, pointer):
            if len(diagnostics) < 20:
                diagnostics.append(Diagnostic(code, Severity.ERROR, message, pointer=pointer))

        def compare(actual, expected, pointer=""):
            if len(diagnostics) >= 20:
                return
            if isinstance(expected, dict):
                if not isinstance(actual, dict):
                    reject("contract.type", "expected an object", pointer)
                    return
                for key in sorted(expected.keys() - actual.keys()):
                    escaped = key.replace("~", "~0").replace("/", "~1")
                    reject("contract.missing", "required field is missing", f"{pointer}/{escaped}")
                for key in sorted(actual.keys() - expected.keys(), key=str):
                    escaped = str(key).replace("~", "~0").replace("/", "~1")
                    reject("contract.unknown", "field is not part of the output contract", f"{pointer}/{escaped}")
                for key in sorted(expected.keys() & actual.keys()):
                    escaped = key.replace("~", "~0").replace("/", "~1")
                    compare(actual[key], expected[key], f"{pointer}/{escaped}")
                return
            if isinstance(expected, list):
                if not isinstance(actual, list) or len(actual) != len(expected):
                    reject("contract.array", "array type or length differs from the template", pointer)
                    return
                for index, (left, right) in enumerate(zip(actual, expected)):
                    compare(left, right, f"{pointer}/{index}")
                return
            if isinstance(expected, (int, float)) and not isinstance(expected, bool):
                try:
                    finite = type(actual) in (int, float) and math.isfinite(actual)
                except OverflowError:
                    finite = False
                if not finite:
                    reject("contract.number", "expected a finite JSON number, not a boolean or string", pointer)
                    return
                name = pointer.removeprefix("/settings/")
                control = self.spec["variable_controls"].get(name)
                if control is not None:
                    valid = actual in control["values"] if "values" in control else control["minimum"] <= actual <= control["maximum"]
                    if not valid:
                        reject("contract.control", "control is outside the supported values or range", pointer)
                    return
                if actual != expected:
                    reject("contract.fixed", "fixed control differs from the pinned template", pointer)
                return
            if type(actual) is not type(expected) or actual != expected:
                reject("contract.fixed", "fixed metadata, shape or embedded payload differs from the pinned template", pointer)

        compare(document, self._expected)
        report = ValidationReport(tuple(diagnostics))
        if report.valid:
            report = report.merge(VitalPreset(document, self._template.schema).validate(runtime=runtime))
        return report

    def read(self, path: Path | str, *, runtime: bool = False) -> tuple[dict | None, ValidationReport]:
        try:
            with Path(path).open("rb") as stream:
                payload = stream.read(self.spec["max_file_bytes"] + 1)
            if len(payload) > self.spec["max_file_bytes"]:
                raise ValueError("file exceeds the output contract's size limit")
            document = strict_json(payload.decode("utf-8-sig"))
        except (OSError, ValueError, RecursionError) as error:
            return None, ValidationReport((Diagnostic("contract.json", Severity.ERROR, str(error)),))
        return document, self.validate(document, runtime=runtime)

    def save(self, preset: VitalPreset, path: Path | str) -> None:
        # Validate both the object and the exact bytes before publishing a destination file.
        self.validate(preset.to_dict()).require_valid()
        destination = Path(path)
        if destination.exists():
            raise FileExistsError(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="\n", dir=destination.parent,
                                             prefix=f".{destination.name}.", suffix=".tmp", delete=False) as stream:
                temporary = Path(stream.name)
                stream.write(preset.to_json())
                stream.flush()
                os.fsync(stream.fileno())
            _, report = self.read(temporary)
            report.require_valid()
            # Same-directory hard link publishes atomically and fails if the destination exists.
            os.link(temporary, destination)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
