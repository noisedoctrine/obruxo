from __future__ import annotations

from dataclasses import asdict, dataclass
from decimal import Decimal
import hashlib
import json
import re


IDENTITY_RULE = "patch_identity_v1"
DISPLAY_FIELDS = frozenset({"author", "comments", "preset_name"})


def require_digest(value: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError("expected a lowercase SHA-256 digest")
    return value


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def reject_constant(value):
    raise ValueError("nonfinite JSON constant")


def parse_document(raw: bytes) -> dict:
    document = json.loads(raw.decode("utf-8-sig"), parse_float=Decimal,
                          object_pairs_hook=unique_object, parse_constant=reject_constant)
    if not isinstance(document, dict):
        raise ValueError("preset root must be an object")
    return document


def typed_tree(value):
    """Preserve numeric types, Decimal precision and array order from the reviewed audit."""
    if isinstance(value, dict):
        return ["object", [[key, typed_tree(value[key])] for key in sorted(value)]]
    if isinstance(value, list):
        return ["array", [typed_tree(item) for item in value]]
    if isinstance(value, Decimal):
        return ["decimal", str(value)]
    if value is None:
        return ["null"]
    if isinstance(value, bool):
        return ["bool", value]
    if isinstance(value, int):
        return ["integer", str(value)]
    if isinstance(value, str):
        return ["string", value]
    raise TypeError(type(value).__name__)


def document_digest(document: dict, *, ignore_display: bool = False) -> str:
    if ignore_display:
        document = {key: value for key, value in document.items() if key not in DISPLAY_FIELDS}
    payload = json.dumps(typed_tree(document), ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(f"{IDENTITY_RULE}:ignore_display={ignore_display}:{payload}".encode()).hexdigest()


@dataclass(frozen=True)
class PresetIdentity:
    raw: str
    canonical: str
    state: str

    @classmethod
    def from_bytes(cls, raw: bytes) -> PresetIdentity:
        document = parse_document(raw)
        return cls(hashlib.sha256(raw).hexdigest(), document_digest(document), document_digest(document, ignore_display=True))

    def to_dict(self) -> dict:
        return asdict(self)
