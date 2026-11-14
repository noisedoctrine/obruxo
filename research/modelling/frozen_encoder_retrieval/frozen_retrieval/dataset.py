from __future__ import annotations

from collections import Counter
from fractions import Fraction
import hashlib
import json
from pathlib import Path, PurePosixPath, PureWindowsPath

from .core import POLICY_ID, canonical_json, check_render_budget, digest


SNAPSHOT_ARTIFACTS = ("contract.json", "assets.json", "performances.json", "splits.json", "members.json", "summary.json")
SOURCE_KIND = "newly_rendered_original_preset"
NONPERFORMANCE_META_TYPES = frozenset({"copyright", "cue_point", "device_name", "instrument_name", "key_signature",
                                       "lyrics", "marker", "sequence_number", "text", "track_name"})


def _load_snapshot(root: Path) -> tuple[dict, dict, dict]:
    from obruxo_data.hashing import canonical_json as obruxo_canonical_json, canonical_sha256
    from obruxo_training.splits import SplitManifest, build_splits

    root = root.resolve(strict=True)
    status = json.loads((root / "status.json").read_text(encoding="utf-8"))
    if status.get("complete") is not True or status.get("stage") != "complete":
        raise ValueError("only a complete frozen source snapshot can be used")
    if set(status.get("artifact_ids", {})) != set(SNAPSHOT_ARTIFACTS):
        raise ValueError("frozen snapshot artifact inventory differs from the audited contract")
    docs = {name: json.loads((root / name).read_text(encoding="utf-8")) for name in SNAPSHOT_ARTIFACTS}
    for name, value in docs.items():
        if canonical_sha256(value) != status["artifact_ids"][name]:
            raise ValueError(f"source snapshot changed: {name}")
    manifest = SplitManifest.from_dict(docs["splits.json"])
    if docs["summary.json"].get("split_manifest_id") != manifest.manifest_id:
        raise ValueError("frozen source summary and split manifest disagree")
    members = docs["members.json"]
    if members.get("version") != 1 or not isinstance(members.get("parents"), dict):
        raise ValueError("unsupported frozen membership artifact")
    identities = [row["identity"]["state"] for row in members.get("records", []) + members.get("exclusions", [])
                  if row.get("identity")]
    rebuilt = build_splits(identities, parents=members["parents"], seed=manifest.seed)
    if rebuilt.manifest_id != manifest.manifest_id:
        raise ValueError("frozen source ancestry and split map disagree")
    for row in members.get("records", []):
        identity = row.get("identity", {})
        state = identity.get("state")
        assignment = manifest.by_state.get(state)
        if assignment is None or (row.get("lineage_id"), row.get("split")) != (assignment.lineage_id, assignment.split):
            raise ValueError("source member and frozen split assignment disagree")
    return docs, status, manifest


def _safe_source(root: Path, relative: str) -> Path:
    if not isinstance(relative, str) or "\\" in relative or PureWindowsPath(relative).drive:
        raise ValueError("frozen source path is not a portable relative path")
    path = PurePosixPath(relative)
    if path.is_absolute() or ".." in path.parts or not path.parts or str(path) != relative:
        raise ValueError("frozen source path escapes its declared data root")
    current = root
    for part in path.parts:
        current = current / part
        if current.is_symlink() or current.is_junction():
            raise ValueError("source path traverses a link")
    resolved = current.resolve(strict=True)
    if not resolved.is_relative_to(root):
        raise ValueError("source path escapes its declared data root")
    return resolved


def _paired_source_midi(root: Path, directory: Path, selected_preset: Path) -> tuple[Path | None, str | None]:
    """Associate one direct-child MIDI with its exact source preset; source WAVs are irrelevant."""
    midi_paths = sorted(directory.glob("*.mid"), key=lambda path: path.name)
    vital_paths = sorted(directory.glob("*.vital"), key=lambda path: path.name)
    if not midi_paths:
        return None, "pair.missing_midi"
    if len(midi_paths) != 1 or len(vital_paths) != 1:
        return None, "pair.ambiguous"
    if vital_paths[0].resolve(strict=True) != selected_preset:
        return None, "pair.selected_source_mismatch"
    relative = midi_paths[0].relative_to(root).as_posix()
    return _safe_source(root, relative), None


def _remove_nonperformance_meta_events(performance) -> dict[str, int]:
    """Drop only known descriptive MIDI metadata and return auditable per-type counts."""
    from obruxo_data.midi import EventKind

    removed = Counter()
    kept = []
    for event in performance.events:
        payload = dict(event.opaque)
        try:
            message = json.loads(payload.get("message", "{}"))
        except (TypeError, json.JSONDecodeError):
            message = {}
        message_type = message.get("type")
        if event.kind == EventKind.OPAQUE and message.get("is_meta") and message_type in NONPERFORMANCE_META_TYPES:
            removed[message_type] += 1
            continue
        kept.append(event)
    performance.replace_events(kept)
    return dict(sorted(removed.items()))


def _midi_duration(performance, tempo_map, start_tick: int, end_tick: int) -> float:
    return float(tempo_map.tick_to_seconds(end_tick) - tempo_map.tick_to_seconds(start_tick))


def bounded_excerpt(performance, capabilities, *, minimum_seconds: float = 2.0, maximum_seconds: float = 8.0):
    """Select a leading, contiguous, whole-note excerpt and preserve supported controller state."""
    from obruxo_data.midi import EventKind, MidiEvent, Performance, TempoMap

    performance.validate(capabilities).require_valid()
    notes = performance.note_spans()
    if not notes:
        raise ValueError("no_complete_notes")
    tempo_map = TempoMap.from_performance(performance)
    start_tick = min(note.start_tick for note in notes)
    target_seconds = tempo_map.tick_to_seconds(start_tick) + Fraction(str(maximum_seconds))
    low, high = start_tick, max(performance.end_tick, max(note.end_tick for note in notes))
    while low < high:
        middle = (low + high + 1) // 2
        if tempo_map.tick_to_seconds(middle) <= target_seconds:
            low = middle
        else:
            high = middle - 1
    limit_tick = low
    selected_notes = [note for note in notes if note.start_tick >= start_tick and note.end_tick <= limit_tick]
    if not selected_notes:
        raise ValueError("no_whole_note_fits_excerpt")
    end_tick = max(note.end_tick for note in selected_notes)
    duration = _midi_duration(performance, tempo_map, start_tick, end_tick)
    if duration < minimum_seconds:
        raise ValueError("excerpt_under_two_seconds")
    selected_events = {note.on_event for note in selected_notes} | {note.off_event for note in selected_notes}
    stateful = {EventKind.PITCH_BEND, EventKind.CHANNEL_PRESSURE, EventKind.CONTROL_CHANGE}
    for event in performance.canonical_events():
        if event.tick < start_tick and event.kind in stateful:
            if event.kind != EventKind.CONTROL_CHANGE or event.data[0] in capabilities.control_changes:
                selected_events.add(event)
        elif start_tick <= event.tick <= end_tick and event.kind not in (EventKind.NOTE_ON, EventKind.NOTE_OFF, EventKind.TEMPO):
            selected_events.add(event)
    first_tempo = max((segment for segment in tempo_map.segments if segment.tick <= start_tick), key=lambda item: item.tick)
    excerpt_events = [MidiEvent(0, -1, EventKind.TEMPO, None, (first_tempo.microseconds_per_beat,))]
    latest_state = {}
    for event in sorted(selected_events, key=lambda item: item.sort_key()):
        if event.tick < start_tick and event.kind in stateful:
            key = (event.kind, event.channel, event.data[0] if event.kind == EventKind.CONTROL_CHANGE else None)
            latest_state[key] = event
            continue
        excerpt_events.append(MidiEvent(event.tick - start_tick, event.order, event.kind, event.channel, event.data, event.opaque))
    for event in latest_state.values():
        excerpt_events.append(MidiEvent(0, event.order, event.kind, event.channel, event.data, event.opaque))
    excerpt = Performance(performance.ticks_per_beat, bpm=None, events=excerpt_events, end_tick=end_tick - start_tick)
    excerpt.validate(capabilities).require_valid()
    return excerpt, {"excerpt_duration_seconds": round(duration, 6), "excerpt_start_tick": start_tick,
                     "excerpt_end_tick": end_tick, "excerpt_policy": "leading_whole_notes_2_to_8s_plus_2s_release_v1"}


def midi_features(performance) -> dict:
    from obruxo_data.midi import EventKind, TempoMap

    notes = performance.note_spans()
    tempo_map = TempoMap.from_performance(performance)
    duration = float(tempo_map.tick_to_seconds(performance.end_tick))
    pitches = sorted(note.pitch for note in notes)
    velocities = [note.velocity for note in notes]
    note_durations = [float(tempo_map.tick_to_seconds(note.end_tick) - tempo_map.tick_to_seconds(note.start_tick)) for note in notes]
    onset_seconds = sorted(float(tempo_map.tick_to_seconds(note.start_tick)) for note in notes)
    onset_intervals = [right - left for left, right in zip(onset_seconds, onset_seconds[1:])]
    changes = Counter()
    for event in performance.events:
        if event.kind == EventKind.CONTROL_CHANGE:
            changes[str(event.data[0])] += 1
    active = 0
    maximum_polyphony = 0
    note_events = sorted([(note.start_tick, 1) for note in notes] + [(note.end_tick, -1) for note in notes],
                         key=lambda item: (item[0], item[1]))
    for _, delta in note_events:
        active += delta
        maximum_polyphony = max(maximum_polyphony, active)
    tempos = [event for event in performance.events if event.kind == EventKind.TEMPO]
    median_note_duration = sorted(note_durations)[len(note_durations) // 2]
    median_onset_interval = sorted(onset_intervals)[len(onset_intervals) // 2] if onset_intervals else median_note_duration
    return {"note_count": len(notes), "max_polyphony": maximum_polyphony,
            "pitch_min": min(pitches), "pitch_max": max(pitches), "pitch_median": pitches[len(pitches) // 2],
            "velocity_min": min(velocities), "velocity_max": max(velocities),
            "velocity_mean": round(sum(velocities) / len(velocities), 4),
            "note_duration_median_seconds": round(median_note_duration, 6),
            "onset_interval_median_seconds": round(median_onset_interval, 6),
            "articulation": "held" if median_note_duration / max(median_onset_interval, 1e-9) >= 0.9 else "phrased",
            "duration_seconds": round(duration, 6),
            "notes_per_second": round(len(notes) / duration, 6) if duration else None,
            "tempo_events": len(tempos), "tempo_bpm": [round(60_000_000 / event.data[0], 4) for event in tempos],
            "control_change_counts": dict(sorted(changes.items())),
            "sustain_events": changes.get("64", 0), "expression_events": sum(changes.get(str(code), 0) for code in (1, 74)),
            "pitch_bend_events": sum(event.kind == EventKind.PITCH_BEND for event in performance.events),
            "channel_pressure_events": sum(event.kind == EventKind.CHANNEL_PRESSURE for event in performance.events)}


def prepare_dataset(snapshot: Path, data_root: Path, renderer_config: Path, output: Path, *, split: str = "validation",
                    gallery_cap: int = 4, max_render_requests: int = 50_000) -> dict:
    """Freeze all eligible source presets crossed with the split's compatible real MIDI pool."""
    import yaml
    from obruxo_data.hashing import canonical_json as obruxo_canonical_json, canonical_sha256
    from obruxo_data.midi import Performance
    from obruxo_data.render.capabilities import RendererCapabilities
    from obruxo_data.render.base import FULL_CORPUS_RENDERER_ID, RenderRequest
    from obruxo_training.identity import PresetIdentity

    if split not in {"train", "validation"}:
        raise ValueError("benchmark split must be train or validation; reserved test is never admitted")
    root = data_root.resolve(strict=True)
    if not root.is_dir():
        raise ValueError("data_root must be a directory")
    output = output.resolve()
    if output.is_relative_to(root):
        raise ValueError("private benchmark outputs must be outside the source corpus")
    docs, status, manifest = _load_snapshot(snapshot)
    config = yaml.safe_load(renderer_config.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or config.get("backend") != "dawdreamer" or config.get("renderer_id") != FULL_CORPUS_RENDERER_ID:
        raise ValueError("benchmark requires the corpus-qualified Vital profile with curve input and native readback")
    capabilities = RendererCapabilities.from_dict(config.get("capabilities", {}))
    if not capabilities.notes:
        raise ValueError("configured renderer does not support notes")
    counters = Counter()
    performances, pairing_audit = [], []
    selected = [row for row in docs["members.json"]["records"] if row.get("split") == split and row.get("static_eligible")]
    selected_by_state = {}
    for row in selected:
        counters["eligible_preset_records"] += 1
        relative = row.get("relative_path")
        state_id = row["identity"]["state"]
        family_id = row["lineage_id"]
        try:
            source = _safe_source(root, relative)
            raw = source.read_bytes()
            identity = PresetIdentity.from_bytes(raw)
            if (identity.raw, identity.state) != (row["identity"]["raw"], state_id):
                raise ValueError("source_identity_drift")
            if source.parent.parent != root / "data":
                raise ValueError("preset_not_in_direct_child_presetshare_layout")
            preset = {"state_id": state_id, "family_id": family_id, "split": split,
                      "relative_path": relative, "raw_sha256": identity.raw}
            previous = selected_by_state.get(state_id)
            if previous and previous["relative_path"] != relative:
                if previous["raw_sha256"] != identity.raw:
                    raise ValueError("one_source_state_maps_to_different_preset_bytes")
                counters["duplicate_state_aliases_collapsed"] += 1
                if relative < previous["relative_path"]:
                    selected_by_state[state_id] = preset
            else:
                selected_by_state[state_id] = preset
        except Exception as error:
            counters[f"preset_excluded:{str(error)[:80]}"] += 1
            continue

    presets = sorted(selected_by_state.values(), key=lambda row: row["state_id"])
    by_directory = {}
    for preset in selected_by_state.values():
        source = _safe_source(root, preset["relative_path"])
        by_directory.setdefault(source.parent, []).append(preset)
    for directory, source_presets in sorted(by_directory.items(), key=lambda item: item[0].relative_to(root).as_posix()):
        relative_directory = directory.relative_to(root).as_posix()
        selected_states = sorted(preset["state_id"] for preset in source_presets)
        midi_path, reason = (None, "pair.ambiguous") if len(source_presets) != 1 else _paired_source_midi(
            root, directory, _safe_source(root, source_presets[0]["relative_path"]))
        if midi_path is None:
            counters[f"pair_excluded:{reason or 'pair.unknown'}"] += 1
            pairing_audit.append({"relative_directory": relative_directory, "selected_state_ids": selected_states,
                                  "status": "excluded", "reason": reason or "pair.unknown"})
            continue

        preset = source_presets[0]
        pairing_audit.append({"relative_directory": relative_directory, "selected_state_ids": selected_states,
                              "status": "eligible", "pairing_contract": "same_directory_exact_one_midi_one_vital",
                              "source_audio_used": False, "midi_relative_path": midi_path.relative_to(root).as_posix()})
        counters["midi_pair_candidates"] += 1
        try:
            midi_raw = midi_path.read_bytes()
            source_sha256 = hashlib.sha256(midi_raw).hexdigest()
            performance = Performance.from_midi(midi_path)
            ignored_metadata = _remove_nonperformance_meta_events(performance)
            validation = performance.validate(capabilities)
            if not validation.valid:
                codes = sorted({diagnostic.code for diagnostic in validation.diagnostics if diagnostic.severity.value == "error"})
                for code in codes:
                    counters[f"midi_excluded:{code}"] += 1
                pairing_audit[-1].update({"status": "excluded", "reason": "renderer_incompatible_midi",
                                          "diagnostic_codes": codes, "ignored_nonperformance_meta": ignored_metadata})
                continue
            stats = midi_features(performance)
            excerpt, excerpt_info = bounded_excerpt(performance, capabilities)
            midi_id = digest({"normalized_performance": excerpt.to_dict(), "policy": "normalized_midi_events_v1"})
            excerpt_id = digest({"midi_id": midi_id, **excerpt_info, "policy": POLICY_ID})
            performances.append({"midi_id": midi_id, "source_state_id": preset["state_id"], "source_family_id": preset["family_id"],
                                 "split": split, "source_relative_path": midi_path.relative_to(root).as_posix(),
                                 "source_sha256": source_sha256, "excerpt_id": excerpt_id, "source_stats": stats,
                                 "stats": midi_features(excerpt), "ignored_nonperformance_meta": ignored_metadata,
                                 "excerpt": excerpt.to_dict(), **excerpt_info})
            counters["midi_parsed"] += 1
            pairing_audit[-1].update({"status": "midi_eligible", "ignored_nonperformance_meta": ignored_metadata,
                                      "midi_id": midi_id, "midi_stats": stats})
            for meta_type, count in ignored_metadata.items():
                counters[f"midi_descriptive_meta_ignored:{meta_type}"] += count
        except Exception as error:
            reason = f"{type(error).__name__}:{str(error).split(':', 1)[0].strip()[:60]}"
            counters[f"midi_excluded:{reason}"] += 1
            pairing_audit[-1].update({"status": "excluded", "reason": reason})
    unique_perf = {}
    for row in sorted(performances, key=lambda item: (item["midi_id"], item["source_state_id"])):
        existing = unique_perf.get(row["midi_id"])
        if existing is None:
            row["source_state_ids"] = [row["source_state_id"]]
            row["source_family_ids"] = [row["source_family_id"]]
            row["source_relative_paths"] = [row["source_relative_path"]]
            row["source_sha256s"] = [row["source_sha256"]]
            unique_perf[row["midi_id"]] = row
        else:
            for field, value in (("source_state_ids", row["source_state_id"]), ("source_family_ids", row["source_family_id"]),
                                 ("source_relative_paths", row["source_relative_path"]), ("source_sha256s", row["source_sha256"])):
                if value not in existing[field]:
                    existing[field].append(value)
            counters["duplicate_midi_performance_aliases_collapsed"] += 1
    performances = list(unique_perf.values())
    if len(presets) < 2 or len(performances) < 2:
        raise ValueError(f"insufficient renderer-compatible corpus for retrieval: presets={len(presets)}, "
                         f"unique_midi_excerpts={len(performances)}; need at least two of each")
    estimated_render_requests = check_render_budget(len(presets), len(performances), max_render_requests)
    candidate_sets, episodes = __import__("frozen_retrieval.core", fromlist=["build_episodes"]).build_episodes(
        presets, performances, gallery_cap=gallery_cap)
    preset_by_state = {row["state_id"]: row for row in presets}
    performance_by_id = {row["midi_id"]: row for row in performances}
    request_specs = {}
    for candidate_set in candidate_sets.values():
        for state_id in candidate_set["state_ids"]:
            for midi_id in performance_by_id:
                request_specs.setdefault((state_id, midi_id), set()).add("gallery")
    for episode in episodes:
        request_specs.setdefault((episode["target_state_id"], episode["query_midi_id"]), set()).add("query")
    output.mkdir(parents=True, exist_ok=False)
    plan_path = output / "render_plan.private.jsonl"
    plan_count = 0
    canonical_presets = {}
    with plan_path.open("w", encoding="utf-8", newline="\n") as plan_stream:
        for (state_id, midi_id), roles in sorted(request_specs.items()):
            preset, midi = preset_by_state[state_id], performance_by_id[midi_id]
            if state_id not in canonical_presets:
                raw = _safe_source(root, preset["relative_path"]).read_bytes()
                if hashlib.sha256(raw).hexdigest() != preset["raw_sha256"]:
                    raise ValueError("preset changed while freezing render plan")
                canonical_presets[state_id] = obruxo_canonical_json(json.loads(raw.decode("utf-8-sig")))
            request = RenderRequest(preset_json=canonical_presets[state_id], performance=Performance.from_dict(midi["excerpt"]),
                                     sample_rate=44_100, channels=2, end_tick=midi["excerpt"]["end_tick"],
                                     tail_seconds=2.0, renderer_id=config["renderer_id"])
            request_id = request.request_id
            plan_row = {"request_id": request_id, "state_id": state_id, "family_id": preset["family_id"], "split": split,
                        "preset_relative_path": preset["relative_path"], "preset_sha256": preset["raw_sha256"],
                        "midi_id": midi_id, "midi_relative_path": midi["source_relative_path"],
                        "midi_sha256": midi["source_sha256"], "excerpt_id": midi["excerpt_id"], "roles": sorted(roles),
                        "source_kind": SOURCE_KIND, "renderer_id": config["renderer_id"],
                        "excerpt_policy": midi["excerpt_policy"],
                        "excerpt_metadata": {"excerpt_duration_seconds": midi["excerpt_duration_seconds"],
                                             "excerpt_start_tick": midi["excerpt_start_tick"],
                                             "excerpt_end_tick": midi["excerpt_end_tick"]},
                        "sample_rate": 44_100, "channels": 2, "tail_seconds": 2.0}
            plan_stream.write(canonical_json(plan_row) + "\n")
            plan_count += 1
    _write_jsonl(output / "presets.private.jsonl", presets)
    _write_jsonl(output / "performances.private.jsonl", performances)
    _write_jsonl(output / "episodes.private.jsonl", episodes)
    _write_json(output / "candidate_sets.private.json", candidate_sets)
    _write_jsonl(output / "pairing_audit.private.jsonl", pairing_audit)
    manifest_files = ("presets.private.jsonl", "performances.private.jsonl", "episodes.private.jsonl",
                      "render_plan.private.jsonl", "candidate_sets.private.json", "pairing_audit.private.jsonl")
    summary = {"version": 1, "policy_id": POLICY_ID, "source_snapshot_id": canonical_sha256(status["artifact_ids"]),
               "split_manifest_id": docs["summary.json"]["split_manifest_id"],
               "source_split": split, "reserved_split_access": "none", "source_kind": SOURCE_KIND,
               "renderer_id": config["renderer_id"], "renderer_capabilities": capabilities.to_dict(),
               "source_root": str(root), "manifest_sha256": {name: _file_sha256(output / name) for name in manifest_files},
               "presets_selected": len(selected_by_state), "presets_verified": len(presets),
               "source_midi_pair_candidates": counters["midi_pair_candidates"],
               "renderer_compatible_midi_pairs": counters["midi_parsed"],
               "unique_midi_excerpts": len(performances), "episodes": len(episodes),
               "candidate_sets": len(candidate_sets), "render_requests": plan_count,
               "render_budget_maximum": max_render_requests, "render_budget_estimate": estimated_render_requests,
               "gallery_size": len(next(iter(candidate_sets.values()))["state_ids"]) if candidate_sets else 0,
               "gallery_midi_pool_size": len(next(iter(candidate_sets.values()))["gallery_midi_ids"]) if candidate_sets else 0,
               "exclusions": dict(sorted(counters.items())),
               "source_audio": "not used; every clip is a new Vital render from an unchanged original preset",
               "midi_eligibility": "only renderer-supported MIDI behavior is rendered; documented descriptive metadata is omitted and counted"}
    _write_json(output / "dataset_summary.private.json", summary)
    return summary


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(canonical_json(row) + "\n")


def _file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def validate_dataset_manifests(dataset_dir: Path, summary: dict) -> None:
    manifest_sha = summary.get("manifest_sha256")
    if not isinstance(manifest_sha, dict) or not manifest_sha:
        raise ValueError("dataset summary lacks frozen manifest digests")
    for name, expected in manifest_sha.items():
        if Path(name).name != name or not isinstance(expected, str):
            raise ValueError("invalid frozen manifest digest record")
        path = dataset_dir / name
        if not path.is_file() or _file_sha256(path) != expected:
            raise ValueError(f"frozen dataset manifest drift: {name}")
