"""Evidence-based, read-only PresetShare pairing and private manifest creation."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import sys
import tempfile
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .labels import add_source_metadata, performance_labels

EXCLUSION_CODES = (
    "pair.ambiguous",
    "pair.missing_audio",
    "pair.missing_midi",
    "pair.unreadable_audio",
    "pair.invalid_midi",
    "pair.empty_reference",
    "pair.duplicate_identity",
    "pair.invalid_render_sidecar",
    "pair.invalid_preset",
    "pair.derived_render_unavailable",
    "pair.derived_render_failed",
)
PAIRING_METHOD = "same_directory_exact_one_midi_one_audio"
DERIVED_PAIRING_METHOD = "same_directory_exact_one_midi_derived_render"
_RENDERER_UNAVAILABLE = object()


class CorpusInputError(ValueError):
    """A corpus or output contract failure with no private detail in the message."""


class DerivedRenderUnavailable(RuntimeError):
    """The approved local renderer cannot be used in the current runtime."""


@dataclass(frozen=True)
class EvaluationPair:
    pair_id: str
    audio_path: Path
    midi_path: Path
    preset_path: Path | None
    preset_id: str | None
    pairing_method: str
    provenance_status: str
    render_result_path: Path | None
    qa_warning_codes: tuple[str, ...]
    labels: dict[str, Any]

    def manifest_record(self) -> dict[str, Any]:
        return {
            "pair_id": self.pair_id,
            "audio_path": str(self.audio_path),
            "midi_path": str(self.midi_path),
            "preset_path": None if self.preset_path is None else str(self.preset_path),
            "preset_id": self.preset_id,
            "pairing_method": self.pairing_method,
            "provenance_status": self.provenance_status,
            "render_result_path": None if self.render_result_path is None else str(self.render_result_path),
            "qa_warning_codes": list(self.qa_warning_codes),
            "labels": self.labels,
        }


def _approved_output_root() -> Path:
    return (Path(__file__).resolve().parents[2] / "outputs").resolve()


def _safe_output(path: Path | str, *, output_root: Path | None = None) -> Path:
    root = output_root or _approved_output_root()
    resolved = Path(path).resolve(strict=False)
    if resolved == root or not resolved.is_relative_to(root):
        raise CorpusInputError("output must be inside the approved ignored Basic Pitch output area")
    parent = resolved.parent.resolve(strict=False)
    if not parent.is_relative_to(root):
        raise CorpusInputError("output parent is outside the approved ignored Basic Pitch output area")
    return resolved


def _atomic_write(path: Path, text: str) -> None:
    temporary: Path | None = None
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="\n", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _resolve_metadata_path(base: Path, value: str) -> Path:
    candidate = Path(value)
    if candidate.is_absolute():
        return candidate
    for root in (base, base.parent, base.parent.parent):
        resolved = (root / candidate).resolve(strict=False)
        if resolved.exists():
            return resolved
    return (base / candidate).resolve(strict=False)


def _load_metadata(corpus_root: Path) -> dict[Path, dict[str, str]]:
    raw_root = corpus_root.parent.parent
    metadata_path = raw_root / "presetshare_vital_metadata.csv"
    analytics_path = raw_root.parent / "analyze" / "preset_analytics.csv"
    result: dict[Path, dict[str, str]] = {}
    if metadata_path.is_file():
        with metadata_path.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                value = row.get("preset_file", "")
                if value:
                    result[_resolve_metadata_path(raw_root, value)] = dict(row)
    if analytics_path.is_file():
        with analytics_path.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                value = row.get("preset_file", "")
                if not value:
                    continue
                path = _resolve_metadata_path(analytics_path.parent, value)
                if path in result:
                    result[path].update({key: value for key, value in row.items() if value not in (None, "")})
    return result


def _metadata_paths(corpus_root: Path) -> tuple[Path, ...]:
    raw_root = corpus_root.parent.parent
    paths = [raw_root / "presetshare_vital_metadata.csv", raw_root.parent / "analyze" / "preset_analytics.csv"]
    return tuple(path.resolve() for path in paths if path.is_file())


def source_snapshot(root: Path | str, *, extra_paths: tuple[Path, ...] = ()) -> list[dict[str, Any]]:
    """Capture private source stat records without reading or changing source bytes."""
    source_root = Path(root).resolve(strict=True)
    paths = {path.resolve() for path in source_root.rglob("*") if path.is_file()}
    paths.update(path.resolve() for path in extra_paths if path.is_file())
    records = []
    for path in sorted(paths, key=lambda item: str(item).casefold()):
        stat = path.stat()
        records.append({"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns})
    return records


def compare_source_records(before: list[Mapping[str, Any]], after: list[Mapping[str, Any]]) -> dict[str, Any]:
    """Compare two private source-stat captures without exposing them in public reports."""
    before_by_path = {str(record["path"]): dict(record) for record in before}
    after_by_path = {str(record["path"]): dict(record) for record in after}
    mismatches = []
    for path_text in sorted(set(before_by_path) | set(after_by_path), key=str.casefold):
        record = before_by_path.get(path_text)
        current = after_by_path.get(path_text)
        if record is None or current is None or dict(record) != dict(current):
            mismatches.append({"before": record, "after": current})
    return {"source_stat_records": len(before), "source_stat_mismatches": mismatches}


def compare_source_snapshot(records: list[Mapping[str, Any]]) -> dict[str, Any]:
    """Re-stat private sources and return sanitized counts plus private mismatch rows."""
    current_records = []
    for record in records:
        path = Path(str(record["path"]))
        try:
            stat = path.stat()
            current_records.append({"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns})
        except OSError:
            current_records.append({"path": str(path), "missing": True})
    return compare_source_records(records, current_records)


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _pair_id(midi_path: Path, audio_path: Path | None, preset_path: Path | None) -> str:
    digest = hashlib.sha256()
    digest.update(b"obruxo-presetshare-pair-v1\0")
    for label, path in ((b"midi", midi_path), (b"audio", audio_path), (b"preset", preset_path)):
        digest.update(label + b"\0")
        digest.update(b"missing\0" if path is None else _file_digest(path).encode() + b"\0")
    return f"pair-{digest.hexdigest()}"


def _read_audio(path: Path) -> None:
    try:
        import numpy as np
        from scipy.io import wavfile

        _, samples = wavfile.read(path)
        values = np.asarray(samples)
        if values.size == 0 or not np.isfinite(values.astype(np.float64, copy=False)).all():
            raise ValueError("audio is empty or non-finite")
    except (ImportError, OSError, OverflowError, TypeError, ValueError) as exc:
        raise CorpusInputError("audio could not be decoded") from exc


def _sidecar_details(audio_path: Path) -> tuple[Path | None, str, tuple[str, ...]]:
    sidecar = audio_path.with_suffix(".json")
    if not sidecar.is_file():
        return None, "legacy_unknown", ()
    try:
        value = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CorpusInputError("render sidecar is invalid") from exc
    codes: set[str] = set()
    for item in value.get("diagnostics", []) if isinstance(value, Mapping) else []:
        if isinstance(item, Mapping) and item.get("severity") in {"warning", "error"}:
            code = str(item.get("code", "diagnostic")).strip()
            if code:
                codes.add(code)
    qa = value.get("qa", {}) if isinstance(value, Mapping) else {}
    if isinstance(qa, Mapping):
        for key in ("silence_warning", "clipping_count", "tail_warning", "dc_offset_warning"):
            if qa.get(key):
                codes.add(f"qa.{key}")
    return sidecar.resolve(), "available", tuple(sorted(codes))


def _renderer() -> Any:
    data_generation_root = Path(__file__).resolve().parents[5] / "research" / "data_generation"
    if not data_generation_root.is_dir():
        raise DerivedRenderUnavailable("data-generation renderer path unavailable")
    root_text = str(data_generation_root)
    inserted = root_text not in sys.path
    if inserted:
        sys.path.insert(0, root_text)
    try:
        from importlib.util import find_spec

        if find_spec("dawdreamer") is None:
            raise DerivedRenderUnavailable("DawDreamer unavailable")
        from obruxo_data.render import VitalRenderer

        return VitalRenderer.from_config(data_generation_root / "configs" / "renderer.yaml")
    finally:
        if inserted:
            sys.path.remove(root_text)


def _derive_audio(renderer: Any, preset_path: Path, midi_path: Path, destination: Path, output_root: Path) -> Path:
    data_generation_root = Path(__file__).resolve().parents[5] / "research" / "data_generation"
    root_text = str(data_generation_root)
    inserted = root_text not in sys.path
    if inserted:
        sys.path.insert(0, root_text)
    try:
        from obruxo_data.midi import Performance
        from obruxo_data.render import RenderRequest
        from obruxo_data.vital import VitalPreset

        preset = VitalPreset.load(preset_path)
        performance = Performance.from_midi(midi_path)
        request = RenderRequest(
            preset=preset,
            performance=performance,
            sample_rate=44_100,
            end_tick=performance.end_tick,
            tail_seconds=2.0,
            renderer_id=renderer.renderer_id,
        )
        previous_tempdir = tempfile.tempdir
        tempfile.tempdir = str(output_root)
        try:
            result = renderer.render(request)
        finally:
            tempfile.tempdir = previous_tempdir
        if any(getattr(item.severity, "value", None) == "error" for item in result.diagnostics):
            raise RuntimeError("derived render diagnostics contain errors")
        destination.parent.mkdir(parents=True, exist_ok=True)
        result.write_wav(destination)
        sidecar = destination.with_suffix(".json")
        result.write_json(sidecar)
        metadata = json.loads(sidecar.read_text(encoding="utf-8"))
        metadata["audio_source"] = "derived_render"
        metadata["derived_render_provenance"] = {
            "renderer": "obruxo_data.render.VitalRenderer",
            "renderer_id": str(getattr(renderer, "renderer_id", "unknown")),
            "config_path": str(data_generation_root / "configs" / "renderer.yaml"),
            "python": sys.version.split()[0],
            "source_preset_path": str(preset_path),
            "source_midi_path": str(midi_path),
        }
        _atomic_write(sidecar, json.dumps(metadata, indent=2, sort_keys=True) + "\n")
        return destination
    finally:
        if inserted:
            sys.path.remove(root_text)


def _derived_destination(output: Path, pair_id: str, preset_path: Path, midi_path: Path) -> Path:
    root = _approved_output_root()
    destination = (output.parent / "derived_renders" / f"{pair_id}.wav").resolve(strict=False)
    if destination == root or not destination.is_relative_to(root):
        raise CorpusInputError("derived render destination is outside approved outputs")
    for source in (preset_path, midi_path):
        source_root = source.parent.resolve()
        if destination == source_root or destination.is_relative_to(source_root):
            raise CorpusInputError("derived render destination overlaps a source directory")
    if destination.exists():
        raise CorpusInputError("refusing to overwrite an existing derived render")
    return destination


def _candidate_record(directory: Path, status: str, reason: str | None = None) -> dict[str, Any]:
    value = {"directory": str(directory), "status": status}
    if reason:
        value["reason"] = reason
    return value


def _spot_check(audit_rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    """Check a deterministic private sample of each observed pairing method."""
    candidates = [row for row in audit_rows if row.get("status") == "eligible"]
    candidates += [
        row
        for row in audit_rows
        if row.get("reason") in {"pair.derived_render_unavailable", "pair.derived_render_failed"}
    ]
    checked = 0
    discrepancies = 0
    methods: set[str] = set()
    for row in candidates[:8]:
        directory = Path(str(row["directory"]))
        midi_count = len(list(directory.glob("*.mid")))
        audio_count = len(list(directory.glob("*.wav")))
        vital_count = len(list(directory.glob("*.vital")))
        if row.get("reason", "").startswith("pair.derived_render"):
            methods.add(DERIVED_PAIRING_METHOD)
            valid = midi_count == 1 and vital_count == 1 and audio_count == 0
        else:
            methods.add(PAIRING_METHOD)
            valid = midi_count == 1 and audio_count == 1
        checked += 1
        discrepancies += not valid
    return {"count": checked, "discrepancies": discrepancies, "status": "passed" if discrepancies == 0 else "failed", "methods": sorted(methods)}


def _evaluate_candidate(
    directory: Path,
    *,
    metadata: Mapping[Path, Mapping[str, str]],
    output: Path,
    allow_derived_render: bool,
    renderer: Any,
) -> tuple[EvaluationPair | None, dict[str, Any], Any]:
    vital_paths = sorted(directory.glob("*.vital"), key=lambda path: path.name)
    midi_paths = sorted(directory.glob("*.mid"), key=lambda path: path.name)
    audio_paths = sorted(directory.glob("*.wav"), key=lambda path: path.name)
    if len(midi_paths) == 0:
        return None, _candidate_record(directory, "excluded", "pair.missing_midi"), renderer
    if len(midi_paths) != 1 or len(audio_paths) > 1:
        return None, _candidate_record(directory, "pairing_ambiguous", "pair.ambiguous"), renderer
    if not audio_paths and len(vital_paths) > 1:
        return None, _candidate_record(directory, "pairing_ambiguous", "pair.ambiguous"), renderer
    midi_path = midi_paths[0].resolve()
    preset_path = vital_paths[0].resolve() if len(vital_paths) == 1 else None
    try:
        _, midi_labels = performance_labels(midi_path)
    except ValueError as exc:
        reason = "pair.empty_reference" if "no note spans" in str(exc) else "pair.invalid_midi"
        return None, _candidate_record(directory, "excluded", reason), renderer
    metadata_row = metadata.get(preset_path, {}) if preset_path is not None else {}
    labels = add_source_metadata(midi_labels, metadata_row)
    audio_path: Path
    render_result_path: Path | None
    provenance_status: str
    qa_codes: tuple[str, ...]
    pairing_method = PAIRING_METHOD
    if audio_paths:
        audio_path = audio_paths[0].resolve()
        try:
            _read_audio(audio_path)
            render_result_path, provenance_status, qa_codes = _sidecar_details(audio_path)
        except CorpusInputError as exc:
            reason = "pair.invalid_render_sidecar" if "sidecar" in str(exc) else "pair.unreadable_audio"
            return None, _candidate_record(directory, "excluded", reason), renderer
    else:
        if not allow_derived_render:
            return None, _candidate_record(directory, "excluded", "pair.missing_audio"), renderer
        if preset_path is None:
            return None, _candidate_record(directory, "excluded", "pair.missing_audio"), renderer
        pair_id = _pair_id(midi_path, None, preset_path)
        try:
            audio_path = _derived_destination(output, pair_id, preset_path, midi_path)
            if renderer is _RENDERER_UNAVAILABLE:
                return None, _candidate_record(directory, "excluded", "pair.derived_render_unavailable"), renderer
            if renderer is None:
                renderer = _renderer()
            _derive_audio(renderer, preset_path, midi_path, audio_path, _approved_output_root())
            _read_audio(audio_path)
        except CorpusInputError:
            return None, _candidate_record(directory, "excluded", "pair.derived_render_failed"), renderer
        except DerivedRenderUnavailable:
            return None, _candidate_record(directory, "excluded", "pair.derived_render_unavailable"), _RENDERER_UNAVAILABLE
        except (ImportError, OSError, RuntimeError, TypeError, ValueError):
            return None, _candidate_record(directory, "excluded", "pair.derived_render_failed"), renderer
        render_result_path, provenance_status, qa_codes = _sidecar_details(audio_path)
        pairing_method = DERIVED_PAIRING_METHOD
    pair_id = _pair_id(midi_path, audio_path if pairing_method == PAIRING_METHOD else None, preset_path)
    pair = EvaluationPair(
        pair_id=pair_id,
        audio_path=audio_path,
        midi_path=midi_path,
        preset_path=preset_path,
        preset_id=metadata_row.get("preset_id") or None,
        pairing_method=pairing_method,
        provenance_status=provenance_status,
        render_result_path=render_result_path,
        qa_warning_codes=qa_codes,
        labels=labels,
    )
    return pair, _candidate_record(directory, "eligible"), renderer


def build_evaluation_manifest(
    corpus_root: Path | str,
    *,
    output: Path | str,
    audit: Path | str,
    allow_derived_render: bool = False,
    force: bool = False,
) -> dict[str, Any]:
    """Scan only the observed direct-child PresetShare layout and write private artifacts."""
    root = Path(corpus_root).resolve(strict=True)
    output_path = _safe_output(output)
    audit_path = _safe_output(audit)
    if not root.is_dir() or output_path == audit_path or output_path == root or output_path.is_relative_to(root) or audit_path.is_relative_to(root):
        raise CorpusInputError("invalid corpus or output path")
    if not force and (output_path.exists() or audit_path.exists()):
        raise FileExistsError("refusing to overwrite evaluation artifacts without force=True")
    metadata = _load_metadata(root)
    source_records = source_snapshot(root, extra_paths=_metadata_paths(root))
    candidates = sorted((item for item in root.iterdir() if item.is_dir()), key=lambda path: path.relative_to(root).as_posix())
    pairs: list[EvaluationPair] = []
    audit_rows: list[dict[str, Any]] = []
    renderer = None
    seen_ids: set[str] = set()
    for directory in candidates:
        pair, record, renderer = _evaluate_candidate(
            directory,
            metadata=metadata,
            output=output_path,
            allow_derived_render=allow_derived_render,
            renderer=renderer,
        )
        if pair is not None and pair.pair_id in seen_ids:
            pair = None
            record = _candidate_record(directory, "excluded", "pair.duplicate_identity")
        if pair is not None:
            seen_ids.add(pair.pair_id)
            pairs.append(pair)
        audit_rows.append(record)
    source_after = source_snapshot(root, extra_paths=_metadata_paths(root))
    source_check = compare_source_records(source_records, source_after)
    if source_check["source_stat_mismatches"]:
        raise CorpusInputError("a source artifact changed during pairing")
    if not pairs:
        manifest_text = ""
    else:
        manifest_text = "\n".join(json.dumps(pair.manifest_record(), sort_keys=True) for pair in pairs) + "\n"
    counts = Counter(str(row["status"]) for row in audit_rows)
    reasons = Counter(str(row["reason"]) for row in audit_rows if row.get("reason"))
    spot_check = _spot_check(audit_rows)
    methods = sorted({pair.pairing_method for pair in pairs} | set(spot_check["methods"]))
    audit_value = {
        "format_version": 1,
        "pairing_contract_version": 1,
        "corpus_layout": "direct child directory with exact one-to-one extensions",
        "candidate_count": len(candidates),
        "eligible_count": len(pairs),
        "excluded_count": sum(count for key, count in reasons.items() if key != "pair.ambiguous"),
        "ambiguous_count": reasons.get("pair.ambiguous", 0),
        "status_counts": dict(sorted(counts.items())),
        "excluded_by_reason": dict(sorted(reasons.items())),
        "selected_pairing_methods": methods,
        "derived_render_opt_in": allow_derived_render,
        "spot_check": {
            "count": spot_check["count"],
            "discrepancies": spot_check["discrepancies"],
            "status": spot_check["status"],
        },
        "source_snapshot": {
            "source_stat_records": source_check["source_stat_records"],
            "source_stat_mismatches": 0,
        },
        "candidates": audit_rows,
    }
    _atomic_write(output_path, manifest_text)
    _atomic_write(audit_path, json.dumps(audit_value, indent=2, sort_keys=True) + "\n")
    _atomic_write(
        output_path.with_name("source_snapshot.json"),
        json.dumps({"before": source_records, "after": source_after, "comparison": source_check}, indent=2, sort_keys=True) + "\n",
    )
    return {
        "candidate_count": len(candidates),
        "eligible_count": len(pairs),
        "excluded_by_reason": dict(sorted(reasons.items())),
        "ambiguous_count": reasons.get("pair.ambiguous", 0),
        "selected_pairing_methods": methods,
        "derived_render_opt_in": allow_derived_render,
    }


def load_evaluation_manifest(path: Path | str) -> tuple[EvaluationPair, ...]:
    """Load the private resolved JSONL manifest without rediscovery."""
    manifest_path = Path(path).resolve(strict=True)
    rows = []
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    pairs = []
    ids: set[str] = set()
    for row in rows:
        if not isinstance(row, Mapping):
            raise CorpusInputError("evaluation manifest row is not an object")
        pair_id = str(row.get("pair_id", ""))
        if not pair_id or pair_id in ids:
            raise CorpusInputError("evaluation manifest has duplicate identity")
        audio = Path(str(row["audio_path"])).resolve(strict=True)
        midi = Path(str(row["midi_path"])).resolve(strict=True)
        preset_value = row.get("preset_path")
        preset = None if preset_value in (None, "") else Path(str(preset_value)).resolve(strict=True)
        if not audio.is_file() or not midi.is_file() or (preset is not None and not preset.is_file()):
            raise CorpusInputError("evaluation manifest references missing source")
        ids.add(pair_id)
        pairs.append(
            EvaluationPair(
                pair_id=pair_id,
                audio_path=audio,
                midi_path=midi,
                preset_path=preset,
                preset_id=None if row.get("preset_id") in (None, "") else str(row["preset_id"]),
                pairing_method=str(row["pairing_method"]),
                provenance_status=str(row["provenance_status"]),
                render_result_path=None if row.get("render_result_path") in (None, "") else Path(str(row["render_result_path"])).resolve(strict=True),
                qa_warning_codes=tuple(str(value) for value in row.get("qa_warning_codes", [])),
                labels=dict(row.get("labels", {})),
            )
        )
    return tuple(pairs)
