from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
import tempfile
from typing import Any

from obruxo_data.errors import Diagnostic, Severity, ValidationError, ValidationReport
from obruxo_data.hashing import canonical_json

from .atlas import VitalSchema
from .components import ComponentRef, component_registry
from .profiles import ComponentProfile
from .validation import validate_document, validate_runtime


def _get_pointer(document: dict[str, Any], pointer: str) -> Any:
    current: Any = document
    for token in pointer.strip("/").split("/"):
        current = current[int(token)] if isinstance(current, list) else current[token]
    return current


def _set_pointer(document: dict[str, Any], pointer: str, value: Any) -> None:
    tokens = pointer.strip("/").split("/")
    current: Any = document
    for token in tokens[:-1]:
        current = current[int(token)] if isinstance(current, list) else current[token]
    final = tokens[-1]
    if isinstance(current, list):
        current[int(final)] = value
    else:
        current[final] = value


class VitalPreset:
    def __init__(self, document: dict[str, Any], schema: VitalSchema):
        self._document = deepcopy(document)
        self.schema = schema

    @classmethod
    def init(cls, schema: VitalSchema | None = None) -> "VitalPreset":
        resolved = schema or VitalSchema.load()
        return cls(resolved.load_init_document(), resolved)

    @classmethod
    def load(cls, path: Path | str, *, schema: VitalSchema | None = None) -> "VitalPreset":
        resolved = schema or VitalSchema.load()
        document = json.loads(Path(path).read_text(encoding="utf-8-sig"))
        return cls(document, resolved)

    def get_raw(self, name: str) -> float:
        self._parameter(name)
        return float(self._document["settings"][name])

    def set_raw(self, name: str, value: float) -> None:
        parameter = self._parameter(name)
        message = parameter.validate_raw(value)
        if message:
            raise ValueError(f"{name} {message}")
        self._document["settings"][name] = float(value)

    def get_normalized(self, name: str) -> float:
        return self._parameter(name).normalized_from_raw(self.get_raw(name))

    def set_normalized(self, name: str, value: float) -> None:
        self.set_raw(name, self._parameter(name).raw_from_normalized(value))

    def connect_modulation(self, slot: int, source: str, destination: str, *, amount: float = 1.0,
                           line_mapping: dict[str, Any] | None = None) -> None:
        if not 1 <= slot <= 64:
            raise ValueError("modulation slot must be between 1 and 64")
        if source not in self.schema.modulation_sources:
            raise ValueError(f"unknown modulation source: {source}")
        if destination not in self.schema.modulation_destinations:
            raise ValueError(f"unknown modulation destination: {destination}")
        candidate = deepcopy(self._document)
        connection: dict[str, Any] = {"source": source, "destination": destination}
        if line_mapping is not None:
            connection["line_mapping"] = deepcopy(line_mapping)
        candidate["settings"]["modulations"][slot - 1] = connection
        parameter = self._parameter(f"modulation_{slot}_amount")
        message = parameter.validate_raw(amount)
        if message:
            raise ValueError(f"modulation_{slot}_amount {message}")
        candidate["settings"][f"modulation_{slot}_amount"] = float(amount)
        self._replace_document(candidate)

    def disconnect_modulation(self, slot: int) -> None:
        if not 1 <= slot <= 64:
            raise ValueError("modulation slot must be between 1 and 64")
        candidate = deepcopy(self._document)
        self._clear_route(candidate, slot - 1)
        self._replace_document(candidate)

    def reset_component(self, component: ComponentRef) -> None:
        candidate = deepcopy(self._document)
        self._reset_document_component(candidate, component)
        report = validate_document(candidate, self.schema)
        report.require_valid()
        self._document = candidate

    def apply_profile(self, profile: ComponentProfile) -> None:
        candidate = VitalPreset(self._document, self.schema)
        profile.apply(candidate)
        report = candidate.validate()
        report.require_valid()
        self._document = candidate._document

    def validate(self, *, runtime: bool = False) -> ValidationReport:
        report = validate_document(self._document, self.schema)
        if runtime and report.valid:
            report = report.merge(validate_runtime(self.to_json(canonical=True)))
        return report

    def to_dict(self) -> dict[str, Any]:
        return deepcopy(self._document)

    def to_json(self, *, canonical: bool = False) -> str:
        if canonical:
            return canonical_json(self._document)
        return json.dumps(self._document, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True) + "\n"

    def save(self, path: Path | str) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        self.validate().require_valid()
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="\n", dir=destination.parent,
                                             prefix=f".{destination.name}.", suffix=".tmp", delete=False) as stream:
                stream.write(self.to_json())
                stream.flush()
                os.fsync(stream.fileno())
                temporary = Path(stream.name)
            VitalPreset.load(temporary, schema=self.schema).validate().require_valid()
            os.replace(temporary, destination)
            temporary = None
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    def _parameter(self, name: str):
        try:
            return self.schema.parameters[name]
        except KeyError as error:
            raise KeyError(f"unknown Vital parameter: {name}") from error

    def _reset_document_component(self, document: dict[str, Any], component: ComponentRef) -> None:
        registry = component_registry(self.schema.schema_id)
        if component not in registry:
            raise KeyError(f"unknown component: {component}")
        definition = registry[component]
        init = self.schema.load_init_document()
        settings = document["settings"]
        init_settings = init["settings"]
        for name in definition.owned_parameters:
            settings[name] = deepcopy(init_settings[name])
        for pointer in definition.owned_json_pointers:
            _set_pointer(document, pointer, deepcopy(_get_pointer(init, pointer)))

        destinations = set(definition.owned_parameters)
        sources = set(definition.modulation_source_names)
        for index, connection in enumerate(settings["modulations"]):
            if connection.get("source", "") in sources or connection.get("destination", "") in destinations:
                self._clear_route(document, index, init)

    def _clear_route(self, document: dict[str, Any], index: int, init: dict[str, Any] | None = None) -> None:
        init_document = init or self.schema.load_init_document()
        settings = document["settings"]
        init_settings = init_document["settings"]
        settings["modulations"][index] = deepcopy(init_settings["modulations"][index])
        prefix = f"modulation_{index + 1}_"
        for name in self.schema.parameters:
            if name.startswith(prefix):
                settings[name] = deepcopy(init_settings[name])

    def _replace_document(self, document: dict[str, Any]) -> None:
        report = validate_document(document, self.schema)
        if not report.valid:
            raise ValidationError(report)
        self._document = document

    def _profile_failure(self, code: str, message: str, **context: Any) -> ValidationError:
        return ValidationError(ValidationReport((Diagnostic(code, Severity.ERROR, message, context=context),)))
