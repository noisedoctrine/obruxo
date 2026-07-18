#!/usr/bin/env python3
"""Build a compact, shareable audit of a recursive Vital preset corpus.

The script scans every ``*.vital`` file below a corpus root, parses presets one at
a time, and writes one ZIP artifact containing aggregate program-space evidence.
It deliberately excludes raw embedded sample/audio payloads and wavetable buffers.

Standard-library only. Tested with Python 3.10+.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import platform
import re
import sys
import time
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator

SCRIPT_VERSION = "1.0.0"
DEFAULT_RELATIVE_ROOT = Path("datasets") / "presetshare" / "raw" / "presetshare_files" / "data"
DEFAULT_OUTPUT = "vital_preset_corpus_audit.zip"
MAX_UNIQUE_VALUES = 256
MAX_STRING_EXAMPLES = 16
MAX_EXAMPLE_LENGTH = 160
PAYLOAD_FIELD_NAMES = {
    "audio_file",
    "samples",
    "samples_stereo",
    "wave_data",
    "line",
}
LIKELY_BASE64_RE = re.compile(r"^[A-Za-z0-9+/=\r\n]+$")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def json_type(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int) and not isinstance(value, bool):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return type(value).__name__


def is_finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def safe_text_example(value: str) -> str:
    value = value.replace("\r", "\\r").replace("\n", "\\n")
    return value if len(value) <= MAX_EXAMPLE_LENGTH else value[: MAX_EXAMPLE_LENGTH - 1] + "…"


def looks_like_payload(field_name: str | None, value: Any) -> bool:
    if field_name in PAYLOAD_FIELD_NAMES:
        return True
    if not isinstance(value, str) or len(value) < 4096:
        return False
    sample = value[:4096]
    return bool(LIKELY_BASE64_RE.match(sample))


def normalized_child_path(parent: str, key: str) -> str:
    return f"{parent}.{key}" if parent != "$" else f"$.{key}"


@dataclass
class NumericStats:
    count: int = 0
    minimum: float | None = None
    maximum: float | None = None
    total: float = 0.0
    total_squares: float = 0.0

    def add(self, value: float) -> None:
        value = float(value)
        self.count += 1
        self.minimum = value if self.minimum is None else min(self.minimum, value)
        self.maximum = value if self.maximum is None else max(self.maximum, value)
        self.total += value
        self.total_squares += value * value

    def as_dict(self) -> dict[str, Any]:
        if not self.count:
            return {"count": 0, "min": None, "max": None, "mean": None, "stddev": None}
        mean = self.total / self.count
        variance = max(0.0, self.total_squares / self.count - mean * mean)
        return {
            "count": self.count,
            "min": self.minimum,
            "max": self.maximum,
            "mean": mean,
            "stddev": math.sqrt(variance),
        }


@dataclass
class CappedValues:
    limit: int = MAX_UNIQUE_VALUES
    values: set[Any] = field(default_factory=set)
    truncated: bool = False

    def add(self, value: Any) -> None:
        if self.truncated:
            return
        try:
            self.values.add(value)
        except TypeError:
            value = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
            self.values.add(value)
        if len(self.values) > self.limit:
            self.values.clear()
            self.truncated = True

    def sorted_values(self) -> list[Any]:
        if self.truncated:
            return []
        try:
            return sorted(self.values)
        except TypeError:
            return sorted(self.values, key=lambda item: repr(item))


@dataclass
class PathStats:
    occurrences: int = 0
    file_occurrences: int = 0
    types: Counter[str] = field(default_factory=Counter)
    numeric: NumericStats = field(default_factory=NumericStats)
    scalar_values: CappedValues = field(default_factory=CappedValues)
    string_examples: CappedValues = field(default_factory=lambda: CappedValues(MAX_STRING_EXAMPLES))
    string_lengths: NumericStats = field(default_factory=NumericStats)
    array_lengths: NumericStats = field(default_factory=NumericStats)
    object_field_sets: Counter[tuple[str, ...]] = field(default_factory=Counter)
    payload_occurrences: int = 0
    payload_lengths: NumericStats = field(default_factory=NumericStats)
    _last_file_id: int = -1

    def touch_file(self, file_id: int) -> None:
        if self._last_file_id != file_id:
            self.file_occurrences += 1
            self._last_file_id = file_id


@dataclass
class CorpusAudit:
    root: Path
    started_at: str = field(default_factory=utc_now)
    discovered_files: int = 0
    parsed_files: int = 0
    failed_files: int = 0
    total_bytes: int = 0
    paths: dict[str, PathStats] = field(default_factory=lambda: defaultdict(PathStats))
    root_field_sets: Counter[tuple[str, ...]] = field(default_factory=Counter)
    settings_field_sets: Counter[tuple[str, ...]] = field(default_factory=Counter)
    synth_versions: Counter[str] = field(default_factory=Counter)
    wavetable_component_types: Counter[str] = field(default_factory=Counter)
    wavetable_component_fields: dict[str, Counter[tuple[str, ...]]] = field(default_factory=lambda: defaultdict(Counter))
    wavetable_keyframe_fields: dict[str, Counter[tuple[str, ...]]] = field(default_factory=lambda: defaultdict(Counter))
    wavetable_counts: Counter[int] = field(default_factory=Counter)
    wavetable_group_counts: Counter[int] = field(default_factory=Counter)
    wavetable_groups_per_oscillator: Counter[tuple[int, ...]] = field(default_factory=Counter)
    lfo_counts: Counter[int] = field(default_factory=Counter)
    modulation_counts: Counter[int] = field(default_factory=Counter)
    modulation_sources: Counter[str] = field(default_factory=Counter)
    modulation_destinations: Counter[str] = field(default_factory=Counter)
    modulation_pairs: Counter[tuple[str, str]] = field(default_factory=Counter)
    line_mapping_field_sets: Counter[tuple[str, ...]] = field(default_factory=Counter)
    sample_field_sets: Counter[tuple[str, ...]] = field(default_factory=Counter)
    file_rows: list[dict[str, Any]] = field(default_factory=list)
    errors: list[dict[str, Any]] = field(default_factory=list)

    def record_value(self, path: str, value: Any, file_id: int, field_name: str | None = None) -> None:
        stats = self.paths[path]
        stats.occurrences += 1
        stats.touch_file(file_id)
        kind = json_type(value)
        stats.types[kind] += 1

        if looks_like_payload(field_name, value):
            stats.payload_occurrences += 1
            if isinstance(value, (str, bytes, bytearray, list)):
                stats.payload_lengths.add(len(value))
            return

        if is_finite_number(value):
            stats.numeric.add(float(value))
            stats.scalar_values.add(value)
        elif isinstance(value, bool) or value is None:
            stats.scalar_values.add(value)
        elif isinstance(value, str):
            stats.string_lengths.add(len(value))
            stats.string_examples.add(safe_text_example(value))
        elif isinstance(value, list):
            stats.array_lengths.add(len(value))
        elif isinstance(value, dict):
            stats.object_field_sets[tuple(sorted(map(str, value.keys())))] += 1

    def walk(self, value: Any, path: str, file_id: int, field_name: str | None = None) -> None:
        self.record_value(path, value, file_id, field_name)
        if looks_like_payload(field_name, value):
            return
        if isinstance(value, dict):
            for key, child in value.items():
                self.walk(child, normalized_child_path(path, str(key)), file_id, str(key))
        elif isinstance(value, list):
            item_path = f"{path}[]"
            for child in value:
                self.walk(child, item_path, file_id, field_name)


def iter_vital_files(root: Path) -> Iterator[Path]:
    for directory, _, filenames in os.walk(root):
        base = Path(directory)
        for filename in filenames:
            if filename.lower().endswith(".vital"):
                yield base / filename


def load_json_bytes(path: Path) -> tuple[bytes, dict[str, Any]]:
    raw = path.read_bytes()
    text = raw.decode("utf-8-sig")
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError(f"root JSON value is {json_type(value)}, expected object")
    return raw, value


def inspect_wavetables(audit: CorpusAudit, settings: dict[str, Any]) -> dict[str, Any]:
    wavetables = settings.get("wavetables")
    summary = {
        "wavetable_count": 0,
        "wavetable_group_counts": "",
        "wavetable_component_types": "",
        "unsupported_import_source_count": 0,
    }
    if not isinstance(wavetables, list):
        return summary

    audit.wavetable_counts[len(wavetables)] += 1
    summary["wavetable_count"] = len(wavetables)
    group_vector: list[int] = []
    component_counts: Counter[str] = Counter()

    for wavetable in wavetables:
        if not isinstance(wavetable, dict):
            group_vector.append(0)
            continue
        groups = wavetable.get("groups")
        group_count = len(groups) if isinstance(groups, list) else 0
        group_vector.append(group_count)
        audit.wavetable_group_counts[group_count] += 1
        if not isinstance(groups, list):
            continue
        for group in groups:
            if not isinstance(group, dict):
                continue
            components = group.get("components")
            if not isinstance(components, list):
                continue
            for component in components:
                if not isinstance(component, dict):
                    continue
                component_type = str(component.get("type", "<missing>"))
                component_counts[component_type] += 1
                audit.wavetable_component_types[component_type] += 1
                audit.wavetable_component_fields[component_type][tuple(sorted(component.keys()))] += 1
                keyframes = component.get("keyframes")
                if isinstance(keyframes, list):
                    for keyframe in keyframes:
                        if isinstance(keyframe, dict):
                            audit.wavetable_keyframe_fields[component_type][tuple(sorted(keyframe.keys()))] += 1

    audit.wavetable_groups_per_oscillator[tuple(group_vector)] += 1
    summary["wavetable_group_counts"] = json.dumps(group_vector, separators=(",", ":"))
    summary["wavetable_component_types"] = json.dumps(dict(sorted(component_counts.items())), separators=(",", ":"))
    summary["unsupported_import_source_count"] = component_counts.get("Audio File Source", 0)
    return summary


def inspect_relationships(audit: CorpusAudit, settings: dict[str, Any]) -> dict[str, Any]:
    modulations = settings.get("modulations")
    if isinstance(modulations, list):
        audit.modulation_counts[len(modulations)] += 1
        for item in modulations:
            if not isinstance(item, dict):
                continue
            source = str(item.get("source", "<missing>"))
            destination = str(item.get("destination", "<missing>"))
            audit.modulation_sources[source] += 1
            audit.modulation_destinations[destination] += 1
            audit.modulation_pairs[(source, destination)] += 1
            mapping = item.get("line_mapping")
            if isinstance(mapping, dict):
                audit.line_mapping_field_sets[tuple(sorted(mapping.keys()))] += 1
    else:
        audit.modulation_counts[-1] += 1

    lfos = settings.get("lfos")
    audit.lfo_counts[len(lfos) if isinstance(lfos, list) else -1] += 1

    sample = settings.get("sample")
    if isinstance(sample, dict):
        audit.sample_field_sets[tuple(sorted(sample.keys()))] += 1

    return {
        "modulation_count": len(modulations) if isinstance(modulations, list) else "",
        "lfo_shape_count": len(lfos) if isinstance(lfos, list) else "",
        "sample_field_count": len(sample) if isinstance(sample, dict) else "",
    }


def process_file(audit: CorpusAudit, path: Path, file_id: int) -> None:
    relative = path.relative_to(audit.root).as_posix()
    size = path.stat().st_size
    audit.total_bytes += size
    row: dict[str, Any] = {
        "relative_path": relative,
        "size_bytes": size,
        "sha256": "",
        "synth_version": "",
        "preset_name": "",
        "author": "",
        "root_field_count": "",
        "settings_key_count": "",
        "numeric_settings_key_count": "",
        "modulation_count": "",
        "lfo_shape_count": "",
        "wavetable_count": "",
        "wavetable_group_counts": "",
        "wavetable_component_types": "",
        "unsupported_import_source_count": "",
        "sample_field_count": "",
        "parse_status": "ok",
        "error": "",
    }
    try:
        raw, preset = load_json_bytes(path)
        row["sha256"] = hashlib.sha256(raw).hexdigest()
        audit.parsed_files += 1
        audit.root_field_sets[tuple(sorted(preset.keys()))] += 1
        row["root_field_count"] = len(preset)
        row["synth_version"] = str(preset.get("synth_version", ""))
        row["preset_name"] = safe_text_example(str(preset.get("preset_name", "")))
        row["author"] = safe_text_example(str(preset.get("author", "")))
        audit.synth_versions[row["synth_version"] or "<missing>"] += 1

        settings = preset.get("settings")
        if isinstance(settings, dict):
            audit.settings_field_sets[tuple(sorted(settings.keys()))] += 1
            row["settings_key_count"] = len(settings)
            row["numeric_settings_key_count"] = sum(is_finite_number(v) for v in settings.values())
            row.update(inspect_relationships(audit, settings))
            row.update(inspect_wavetables(audit, settings))
        else:
            row["parse_status"] = "invalid_settings"
            row["error"] = f"settings is {json_type(settings)}, expected object"

        audit.walk(preset, "$", file_id)
    except Exception as exc:
        audit.failed_files += 1
        row["parse_status"] = "error"
        row["error"] = f"{type(exc).__name__}: {exc}"
        audit.errors.append({"relative_path": relative, "size_bytes": size, "error": row["error"]})
    audit.file_rows.append(row)


def counter_rows(counter: Counter[Any], key_name: str = "value") -> list[dict[str, Any]]:
    return [{key_name: key, "count": count} for key, count in counter.most_common()]


def tuple_counter_rows(counter: Counter[tuple[str, ...]], prefix: str) -> list[dict[str, Any]]:
    return [
        {
            "count": count,
            "field_count": len(fields),
            "fields": json.dumps(list(fields), separators=(",", ":"), ensure_ascii=False),
            "signature": hashlib.sha256("\0".join(fields).encode("utf-8")).hexdigest()[:16],
            "kind": prefix,
        }
        for fields, count in counter.most_common()
    ]


def path_rows(audit: CorpusAudit) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path, stats in sorted(audit.paths.items()):
        numeric = stats.numeric.as_dict()
        string_lengths = stats.string_lengths.as_dict()
        array_lengths = stats.array_lengths.as_dict()
        payload_lengths = stats.payload_lengths.as_dict()
        unique_values = stats.scalar_values.sorted_values()
        rows.append(
            {
                "path": path,
                "occurrences": stats.occurrences,
                "file_occurrences": stats.file_occurrences,
                "file_fraction": stats.file_occurrences / audit.parsed_files if audit.parsed_files else 0.0,
                "types": json.dumps(dict(sorted(stats.types.items())), separators=(",", ":")),
                "numeric_count": numeric["count"],
                "numeric_min": numeric["min"],
                "numeric_max": numeric["max"],
                "numeric_mean": numeric["mean"],
                "numeric_stddev": numeric["stddev"],
                "unique_values_truncated": stats.scalar_values.truncated,
                "unique_values": json.dumps(unique_values, ensure_ascii=False, separators=(",", ":")),
                "string_min_length": string_lengths["min"],
                "string_max_length": string_lengths["max"],
                "string_examples": json.dumps(stats.string_examples.sorted_values(), ensure_ascii=False, separators=(",", ":")),
                "array_min_length": array_lengths["min"],
                "array_max_length": array_lengths["max"],
                "object_shape_count": len(stats.object_field_sets),
                "payload_occurrences": stats.payload_occurrences,
                "payload_min_length": payload_lengths["min"],
                "payload_max_length": payload_lengths["max"],
            }
        )
    return rows


def scalar_setting_rows(audit: CorpusAudit) -> list[dict[str, Any]]:
    prefix = "$.settings."
    rows = []
    for row in path_rows(audit):
        path = row["path"]
        if not path.startswith(prefix):
            continue
        suffix = path[len(prefix) :]
        if "." in suffix or "[]" in suffix:
            continue
        types = json.loads(row["types"])
        if not set(types).issubset({"integer", "number", "boolean", "null"}):
            continue
        row = dict(row)
        row["setting_key"] = suffix
        unique = json.loads(row["unique_values"])
        numeric_integerish = all(isinstance(v, int) or (isinstance(v, float) and v.is_integer()) for v in unique if isinstance(v, (int, float)))
        row["candidate_class"] = (
            "boolean-like"
            if not row["unique_values_truncated"] and set(unique).issubset({0, 1, 0.0, 1.0, False, True})
            else "low-cardinality indexed candidate"
            if not row["unique_values_truncated"] and numeric_integerish and len(unique) <= 64
            else "continuous or high-cardinality"
        )
        rows.append(row)
    return sorted(rows, key=lambda item: item["setting_key"])


def wavetable_component_rows(audit: CorpusAudit) -> list[dict[str, Any]]:
    rows = []
    component_types = set(audit.wavetable_component_fields) | set(audit.wavetable_keyframe_fields)
    for component_type in sorted(component_types):
        component_shapes = audit.wavetable_component_fields[component_type]
        keyframe_shapes = audit.wavetable_keyframe_fields[component_type]
        rows.append(
            {
                "component_type": component_type,
                "occurrences": audit.wavetable_component_types[component_type],
                "component_shape_count": len(component_shapes),
                "component_shapes": json.dumps(
                    [{"count": count, "fields": list(fields)} for fields, count in component_shapes.most_common()],
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
                "keyframe_shape_count": len(keyframe_shapes),
                "keyframe_shapes": json.dumps(
                    [{"count": count, "fields": list(fields)} for fields, count in keyframe_shapes.most_common()],
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
                "is_import_source": component_type == "Audio File Source",
            }
        )
    return rows


def write_csv_to_zip(zf: zipfile.ZipFile, name: str, rows: Iterable[dict[str, Any]], columns: list[str] | None = None) -> int:
    rows = list(rows)
    if columns is None:
        columns = []
        seen = set()
        for row in rows:
            for key in row:
                if key not in seen:
                    seen.add(key)
                    columns.append(key)
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=columns, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    zf.writestr(name, buffer.getvalue().encode("utf-8-sig"), compress_type=zipfile.ZIP_DEFLATED)
    return len(rows)


def write_json_to_zip(zf: zipfile.ZipFile, name: str, value: Any) -> None:
    zf.writestr(
        name,
        (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8"),
        compress_type=zipfile.ZIP_DEFLATED,
    )


def build_artifact(audit: CorpusAudit, output: Path, elapsed_seconds: float) -> None:
    paths = path_rows(audit)
    scalars = scalar_setting_rows(audit)
    component_rows = wavetable_component_rows(audit)
    structure_rows = []
    for kind, counter in (
        ("root_object", audit.root_field_sets),
        ("settings_object", audit.settings_field_sets),
        ("sample_object", audit.sample_field_sets),
        ("line_mapping_object", audit.line_mapping_field_sets),
    ):
        structure_rows.extend(tuple_counter_rows(counter, kind))

    manifest = {
        "artifact_schema": "obruxo_vital_corpus_audit_v1",
        "script_version": SCRIPT_VERSION,
        "created_at_utc": utc_now(),
        "scan_started_at_utc": audit.started_at,
        "corpus_root": str(audit.root.resolve()),
        "platform": platform.platform(),
        "python": sys.version,
        "discovered_vital_files": audit.discovered_files,
        "parsed_files": audit.parsed_files,
        "failed_files": audit.failed_files,
        "total_source_bytes": audit.total_bytes,
        "elapsed_seconds": elapsed_seconds,
        "payload_policy": "Raw embedded sample/audio/wavetable/line payload contents are excluded. Only lengths and structural metadata are recorded.",
        "unique_value_policy": f"Exact unique scalar values are retained only while cardinality <= {MAX_UNIQUE_VALUES}; otherwise the field is marked truncated.",
        "reports": {
            "files.csv": len(audit.file_rows),
            "json_paths.csv": len(paths),
            "scalar_settings.csv": len(scalars),
            "object_shapes.csv": len(structure_rows),
            "wavetable_components.csv": len(component_rows),
            "modulation_sources.csv": len(audit.modulation_sources),
            "modulation_destinations.csv": len(audit.modulation_destinations),
            "modulation_pairs.csv": len(audit.modulation_pairs),
            "versions.csv": len(audit.synth_versions),
            "errors.csv": len(audit.errors),
        },
    }

    readme = f"""# Vital preset corpus audit\n\nThis ZIP was generated by `build_vital_corpus_audit.py` version {SCRIPT_VERSION}.\n\n## Scope\n\n- Recursively scanned `{audit.root}` for `.vital` files.\n- Parsed presets one at a time to keep memory bounded.\n- Did **not** include raw preset files or embedded sample/audio/wavetable buffers.\n- Recorded structural paths, object shapes, scalar aggregate evidence, modulation identities, wavetable component grammar, versions, and parse errors.\n\n## Files\n\n- `manifest.json`: scan provenance and report counts.\n- `files.csv`: one row per preset, with hashes and structural counts.\n- `json_paths.csv`: normalized JSON paths and aggregate type/shape statistics.\n- `scalar_settings.csv`: top-level scalar setting keys and candidate value classes.\n- `object_shapes.csv`: observed field-set signatures for root/settings/sample/remap objects.\n- `wavetable_components.csv`: component and keyframe field grammars by component type.\n- `modulation_sources.csv`, `modulation_destinations.csv`, `modulation_pairs.csv`: relationship vocabulary.\n- `versions.csv`: `synth_version` counts.\n- `array_cardinalities.json`: wavetable/LFO/modulation/group count distributions.\n- `errors.csv`: files that could not be parsed.\n\nObserved corpus values are validation evidence, not authoritative Vital program limits or defaults.\n"""

    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_suffix(output.suffix + ".tmp")
    with zipfile.ZipFile(temp, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9, allowZip64=True) as zf:
        zf.writestr("README.md", readme.encode("utf-8"))
        write_json_to_zip(zf, "manifest.json", manifest)
        write_csv_to_zip(zf, "files.csv", audit.file_rows)
        write_csv_to_zip(zf, "json_paths.csv", paths)
        write_csv_to_zip(zf, "scalar_settings.csv", scalars)
        write_csv_to_zip(zf, "object_shapes.csv", structure_rows)
        write_csv_to_zip(zf, "wavetable_components.csv", component_rows)
        write_csv_to_zip(zf, "modulation_sources.csv", counter_rows(audit.modulation_sources, "source"))
        write_csv_to_zip(zf, "modulation_destinations.csv", counter_rows(audit.modulation_destinations, "destination"))
        write_csv_to_zip(
            zf,
            "modulation_pairs.csv",
            [
                {"source": source, "destination": destination, "count": count}
                for (source, destination), count in audit.modulation_pairs.most_common()
            ],
        )
        write_csv_to_zip(zf, "versions.csv", counter_rows(audit.synth_versions, "synth_version"))
        write_csv_to_zip(zf, "errors.csv", audit.errors, ["relative_path", "size_bytes", "error"])
        write_json_to_zip(
            zf,
            "array_cardinalities.json",
            {
                "wavetable_count_per_preset": dict(sorted(audit.wavetable_counts.items())),
                "group_count_per_wavetable": dict(sorted(audit.wavetable_group_counts.items())),
                "group_count_vector_per_preset": {
                    json.dumps(list(vector), separators=(",", ":")): count
                    for vector, count in audit.wavetable_groups_per_oscillator.most_common()
                },
                "lfo_shape_count_per_preset": dict(sorted(audit.lfo_counts.items())),
                "modulation_count_per_preset": dict(sorted(audit.modulation_counts.items())),
            },
        )
    temp.replace(output)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recursively audit a Vital preset corpus and write one shareable ZIP artifact.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "root",
        nargs="?",
        type=Path,
        default=DEFAULT_RELATIVE_ROOT,
        help="Corpus root containing one or more nested .vital files.",
    )
    parser.add_argument("-o", "--output", type=Path, default=Path(DEFAULT_OUTPUT), help="Output ZIP path.")
    parser.add_argument("--progress-every", type=int, default=250, help="Print progress after this many files; 0 disables progress.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = args.root.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if not root.is_dir():
        print(f"error: corpus root does not exist or is not a directory: {root}", file=sys.stderr)
        return 2
    if output == root or root in output.parents:
        print("error: output ZIP must be outside the corpus root", file=sys.stderr)
        return 2

    files = sorted(iter_vital_files(root), key=lambda path: path.as_posix().lower())
    audit = CorpusAudit(root=root, discovered_files=len(files))
    if not files:
        print(f"error: no .vital files found below {root}", file=sys.stderr)
        return 2

    print(f"Scanning {len(files):,} .vital files below {root}")
    started = time.monotonic()
    for file_id, path in enumerate(files, start=1):
        process_file(audit, path, file_id)
        if args.progress_every and (file_id % args.progress_every == 0 or file_id == len(files)):
            elapsed = time.monotonic() - started
            rate = file_id / elapsed if elapsed else 0.0
            print(
                f"[{file_id:,}/{len(files):,}] parsed={audit.parsed_files:,} failed={audit.failed_files:,} "
                f"rate={rate:.1f} files/s",
                flush=True,
            )

    elapsed = time.monotonic() - started
    build_artifact(audit, output, elapsed)
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    print(f"Wrote {output}")
    print(f"ZIP size: {output.stat().st_size:,} bytes")
    print(f"SHA-256: {digest}")
    return 0 if audit.parsed_files else 1


if __name__ == "__main__":
    raise SystemExit(main())
