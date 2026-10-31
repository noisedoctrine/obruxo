"""The same small note/duration grid for each of three existing training patches."""

from itertools import product
from pathlib import Path
from time import time

import numpy as np
import torch
from obruxo_data.hashing import canonical_sha256, sha256_bytes
from obruxo_data.midi import Performance, TempoMap
from obruxo_data.render.qa import audio_float32_sha256

from .audit import write_json
from .basic_direct import restore_direct
from .benchmark import MemoryMonitor
from .checkpoints import atomic_torch_save
from .codec import PilotTarget
from .conditioned_bridge import read
from .features import tensor_digest, validate_features
from .native_fit import append_durable, digest
from .paired_native_evidence import native_ledger
from .paired_native_fit import PairedSession
from .performances import CAPABILITIES, initial_bank
from .reproducibility import FULL_PHASE_CONTROL
from .shared_auxiliary_run import source_identity
from .unison_fit_data import require_request
from .unison_sound_fit import lines

SPEC = {"format": "balanced_three_patch_performances_v1", "training_indices": [2, 8, 11], "pitches": [36, 60, 84],
        "duration_ticks": [384, 1152], "duration_seconds": [.4, 1.2], "ticks_per_beat": 480, "bpm": 120,
        "velocity": 96, "end_tick": 3840, "tail_seconds": .25, "source_recordings": 18,
        "repeat_grid_indices": [0, 5], "repeat_recordings": 6, "experimental_updates": 0, "dense_outputs": 14}
LIMITS = {"wall_seconds": 900, "process_tree_bytes": 4 * 1024**3, "archive_bytes": 128 * 1024**2,
          "replay_bytes": 128 * 1024**2, "training_attempts": 30, "qualification_attempts": 0, "native_executions": 2}
SOURCES = ("obruxo_training/balanced_basic_data.py", "reports/scripts/prepare_balanced_basic_data.py", "reports/BALANCED_BASIC_DATA_CARD.md")


def performance_ids(value):
    if isinstance(value, list):
        return set().union(*(performance_ids(item) for item in value))
    if not isinstance(value, dict):
        return set()
    found = {canonical_sha256(value)} if "events" in value and "ticks_per_beat" in value else set()
    found |= {item for key, item in value.items() if key == "performance_id" and isinstance(item, str)}
    return found | set().union(*(performance_ids(item) for item in value.values()))


def performances(forbidden=()):
    rows = []
    for pitch, duration in product(SPEC["pitches"], SPEC["duration_ticks"]):
        midi = Performance(ticks_per_beat=SPEC["ticks_per_beat"], bpm=SPEC["bpm"])
        midi.add_note(pitch=pitch, velocity=SPEC["velocity"], start_tick=0, duration_ticks=duration)
        midi.end_tick = SPEC["end_tick"]
        midi.validate(CAPABILITIES).require_valid()
        identity = canonical_sha256(midi.to_dict())
        if identity in forbidden:
            raise ValueError("balanced training performance overlaps a reserved performance")
        rows.append({"grid_index": len(rows), "pitch": pitch, "duration_ticks": duration, "velocity": SPEC["velocity"],
                     "performance_id": identity, "performance": midi.to_dict()})
    return rows


def grid_rows(patches, forbidden=()):
    if (len(patches) != 3 or len({row["state_id"] for row in patches}) != 3
            or [row["training_index"] for row in patches] != SPEC["training_indices"]):
        raise ValueError("balanced data requires the same three distinct training patches")
    return [{**patch, **performance, "index": state * 6 + performance["grid_index"]}
            for state, patch in enumerate(patches) for performance in performances(forbidden)]


def validate_rows(rows):
    states = sorted({row["state_id"] for row in rows})
    if len(rows) != SPEC["source_recordings"] or len(states) != 3 or [row["index"] for row in rows] != list(range(18)):
        raise ValueError("balanced source grid is incomplete or duplicated")
    expected = performances()
    for state in states:
        selected = [row for row in rows if row["state_id"] == state]
        if len({canonical_sha256(row["target"]) for row in selected}) != 1:
            raise ValueError("source controls changed with the musical performance")
        if [{key: row[key] for key in expected[0]} for row in selected] != expected:
            raise ValueError("each patch must receive the same correctly labelled six performances")


def code_identity(prior):
    root = Path(__file__).parents[1]
    return {**source_identity(prior), **{name: sha256_bytes((root / name).read_bytes().replace(b"\r\n", b"\n")) for name in SOURCES}}


def make_plan(fit, bridge, pool, reserved):
    prior, proof = read(fit / "plan.json"), read(fit / "verification.json")
    saved = torch.load(bridge / "model.pt", map_location="cpu", weights_only=True)
    model, _frontend, codec = restore_direct(saved)
    if (not proof["verified"] or not proof["complete"] or not proof["models_adam_calibration_rng_replay_exact"]
            or proof["plan_id"] != canonical_sha256(prior) or proof["qualified_for_release"] or proof["test_split_used"]
            or digest(fit / "plan.json") != proof["evidence_sha256"]["plan.json"]
            or digest(bridge / "model.pt") != prior["initial_file_sha256"]
            or tensor_digest(model.state_dict()) != prior["initial_model_sha256"]
            or pool.pool_id != prior["pool_id"] or codec.to_dict() != pool.codec.to_dict()
            or prior["render_phase_controls"] != FULL_PHASE_CONTROL or prior["render_profile"]["preroll_frames"] != 32768):
        raise ValueError("balanced data requires the audited comparison, original initializer and qualified rendering profile")
    reserved_ids = performance_ids(read(reserved)["performances"])
    if len(reserved_ids) != 24:
        raise ValueError("expected the existing 24-performance confirmation reservation")
    forbidden = reserved_ids | {entry.performance_id for entry in initial_bank()}
    entries = {entry.state_id: entry for entry in pool.entries}
    patches = []
    for index in SPEC["training_indices"]:
        recipe = prior["training"][index]
        entry = entries[recipe["state_id"]]
        if pool.manifest.by_state[entry.state_id].split != "train" or entry.target.to_dict() != recipe["target"]:
            raise ValueError("balanced sources must retain the exact three existing training patches")
        patches.append({"training_index": index, "state_id": entry.state_id, "target": entry.target.to_dict()})
    rows = grid_rows(patches, forbidden)
    validate_rows(rows)
    return {"format": SPEC["format"], "spec": SPEC, "observations": rows, "pool_id": pool.pool_id, "codec": codec.to_dict(),
            "initial_model_sha256": prior["initial_model_sha256"], "initial_file_sha256": prior["initial_file_sha256"],
            "fit_plan_file": str(fit / "plan.json"), "source": code_identity(prior),
            "inputs_sha256": {str(path): digest(path) for path in (fit / "plan.json", fit / "verification.json", bridge / "model.pt", reserved)},
            "parent_fit_plan_id": proof["plan_id"], "limits": LIMITS, "planned_native_requests": 24,
            "render_profile": prior["render_profile"], "render_phase_controls": prior["render_phase_controls"],
            "expected_renderer_identity": prior["expected_renderer_identity"], "reserved_performance_count": len(reserved_ids),
            "confirmation_audio_rendered": False, "test_split_used": False, "experimental_updates": 0, "qualified_for_release": False}


def validate_plan(plan, pool):
    if (plan["spec"] != SPEC or plan["limits"] != LIMITS or plan["source"] != code_identity(plan)
            or any(digest(Path(path)) != value for path, value in plan["inputs_sha256"].items())
            or pool.pool_id != plan["pool_id"] or pool.codec.to_dict() != plan["codec"]):
        raise ValueError("frozen balanced data code, inputs or allocation changed")
    validate_rows(plan["observations"])
    if plan["fit_plan_file"] not in plan["inputs_sha256"]:
        raise ValueError("balanced source identities are not bound to their preceding fit")
    prior = read(Path(plan["fit_plan_file"]))
    entries = {entry.state_id: entry for entry in pool.entries}
    expected = {prior["training"][index]["state_id"]: index for index in SPEC["training_indices"]}
    if (canonical_sha256(prior) != plan["parent_fit_plan_id"] or prior["initial_file_sha256"] != plan["initial_file_sha256"]
            or any(row["state_id"] not in expected or row["training_index"] != expected[row["state_id"]]
                   or row["target"] != entries[row["state_id"]].target.to_dict()
                   or pool.manifest.by_state[row["state_id"]].split != "train" for row in plan["observations"])):
        raise ValueError("balanced data changed the selected source controls or training patches")


def initializer(plan, bridge):
    if digest(bridge / "model.pt") != plan["initial_file_sha256"]:
        raise ValueError("balanced data initializer file changed")
    model, frontend, codec = restore_direct(torch.load(bridge / "model.pt", map_location="cpu", weights_only=True))
    if tensor_digest(model.state_dict()) != plan["initial_model_sha256"] or codec.to_dict() != plan["codec"]:
        raise ValueError("balanced data model, codec or frozen frontend changed")
    return model, frontend, codec


def collect(plan, pool, bridge, output, *, resume=False):
    import os

    import psutil
    executions = lines(output / "executions.jsonl")
    if (len(executions) >= LIMITS["native_executions"] or bool(executions) != resume
            or (output / "run.json").exists() != resume):
        raise ValueError("balanced source execution or recovery allocation changed")
    if resume:
        old, events = read(output / "summary.json"), lines(output / "renders.jsonl")
        last = executions[-1]
        if psutil.pid_exists(last["pid"]) and abs(psutil.Process(last["pid"]).create_time() - last["created_unix"]) < .01:
            raise ValueError("preceding balanced source process is still live")
        if (old["complete"] or not old["within_resource_limits"] or not events or events[-1].get("error_type") != "RenderTimeout"):
            raise ValueError("balanced source recovery requires a terminal known timeout")
        write_json(output / "interruption-summary.json", old)
    else:
        write_json(output / "run.json", {"plan_id": canonical_sha256(plan), "started_unix": time()})
    append_durable(output / "executions.jsonl", {"pid": os.getpid(), "created_unix": psutil.Process().create_time(), "resume": resume})
    model, frontend, codec = initializer(plan, bridge)
    before = tensor_digest(model.state_dict())
    monitor = MemoryMonitor()
    monitor.start()
    session = PairedSession(output, plan, codec, monitor)
    records, failure = [], None
    (output / "records").mkdir(exist_ok=True)
    try:
        for row in plan["observations"]:
            index, performance, target = row["index"], Performance.from_dict(row["performance"]), PilotTarget.from_dict(row["target"])
            path = output / "records" / f"{index:02d}.pt"
            record_path = path.with_suffix(".json")
            if record_path.exists():
                record = read(record_path)
                if digest(path) != record["file_sha256"] or record["observation"] != row:
                    raise ValueError("committed balanced source recording changed")
                saved = torch.load(path, map_location="cpu", weights_only=True)
            else:
                audio, receipt = session.render(target, performance, "training", f"source:{index:02d}")
                if audio is None:
                    raise RuntimeError(f"balanced source render failed: {receipt.get('error_type', receipt['status'])}")
                require_request(codec, plan, target, performance, receipt)
                features = frontend(audio)
                validate_features(features)
                saved = {"audio": torch.from_numpy(audio.copy()), "features": features}
                atomic_torch_save(path, saved)
                record = {"observation": row, "receipt": receipt, "file_sha256": digest(path), "features_sha256": tensor_digest(features)}
                write_json(record_path, record)
            if row["grid_index"] in SPEC["repeat_grid_indices"]:
                repeat_path = path.with_name(path.stem + "-repeat.pt")
                repeat_record = repeat_path.with_suffix(".json")
                if not repeat_record.exists():
                    audio, receipt = session.render(target, performance, "training", f"repeat:{index:02d}")
                    if audio is None or not np.array_equal(audio, saved["audio"].numpy()):
                        raise ValueError("fixed-phase balanced source repeat is not exact")
                    require_request(codec, plan, target, performance, receipt)
                    atomic_torch_save(repeat_path, {"audio": torch.from_numpy(audio.copy())})
                    write_json(repeat_record, {"receipt": receipt, "file_sha256": digest(repeat_path), "exact": True})
            records.append({**row, "features": saved["features"]})
            write_json(output / "progress.json", {"source_recordings": len(records), "planned": 18})
            session.check_limits()
            if sum(path.stat().st_size for path in output.rglob("*") if path.is_file()) > LIMITS["archive_bytes"]:
                raise ValueError("balanced source archive limit exceeded")
        frontend.require_frozen()
        if tensor_digest(model.state_dict()) != before:
            raise ValueError("source preparation changed the inverse initializer")
        atomic_torch_save(output / "features.pt", {"rows": records})
    except BaseException as error:
        failure = f"{type(error).__name__}: {error}"
        raise
    finally:
        monitor.close()
        elapsed = time() - session.started
        requests = sum(row["event"] == "attempt" for row in session.events)
        archive = sum(path.stat().st_size for path in output.rglob("*") if path.is_file())
        within = bool(monitor.peak_bytes and monitor.peak_bytes <= LIMITS["process_tree_bytes"] and not monitor.errors
                      and elapsed <= LIMITS["wall_seconds"] and archive <= LIMITS["archive_bytes"] and requests <= 30)
        write_json(output / "summary.json", {"plan_id": canonical_sha256(plan), "complete": failure is None and len(records) == 18,
            "failure": failure, "native_attempts": requests, "within_resource_limits": within, "source_recordings": len(records),
            "elapsed_seconds": elapsed, "peak_process_tree_rss_bytes": monitor.peak_bytes, "rss_sampling_errors": monitor.errors,
            "archive_bytes_before_summary": archive, "experimental_updates": 0, "qualified_for_release": False, "test_split_used": False})
    return read(output / "summary.json")


def verify(plan, pool, bridge, output):
    summary = read(output / "summary.json")
    if not summary["complete"] or summary["failure"] or not summary["within_resource_limits"]:
        raise ValueError("balanced source collection is incomplete or out of budget")
    evidence = {path.relative_to(output).as_posix(): digest(path) for path in output.rglob("*") if path.is_file() and path.name != "verification.json"}
    started, monitor = time(), MemoryMonitor()
    monitor.start()
    try:
        model, frontend, codec = initializer(plan, bridge)
        original = tensor_digest(model.state_dict())
        features = torch.load(output / "features.pt", map_location="cpu", weights_only=True)["rows"]
        attempts, outcomes, recovered = native_ledger(lines(output / "renders.jsonl"), plan["expected_renderer_identity"])
        if len(attempts) != summary["native_attempts"] or len(features) != 18:
            raise ValueError("balanced source accounting or cache count changed")
        repeats = 0
        for row, cached in zip(plan["observations"], features, strict=True):
            index, midi, target = row["index"], Performance.from_dict(row["performance"]), PilotTarget.from_dict(row["target"])
            path = output / "records" / f"{index:02d}.pt"
            saved, record = torch.load(path, map_location="cpu", weights_only=True), read(path.with_suffix(".json"))
            receipt, audio = record["receipt"], saved["audio"].numpy()
            frames = TempoMap.from_performance(midi).render_frame_count(midi.end_tick, codec.contract.tail_seconds, 44100)
            if (record["observation"] != row or digest(path) != record["file_sha256"] or audio.dtype != np.float32
                    or audio.shape != (frames, 2) or not np.isfinite(audio).all()
                    or audio_float32_sha256(audio) != receipt["qa"]["audio_float32_sha256"]
                    or receipt != outcomes[receipt["phase"], receipt["attempt"]]
                    or {key: value for key, value in cached.items() if key != "features"} != row):
                raise ValueError("balanced source waveform, MIDI label or receipt changed")
            require_request(codec, plan, target, midi, receipt)
            rebuilt = frontend(audio)
            if not tensor_digest(rebuilt) == tensor_digest(saved["features"]) == tensor_digest(cached["features"]) == record["features_sha256"]:
                raise ValueError("balanced source features cannot be reconstructed exactly")
            if row["grid_index"] in SPEC["repeat_grid_indices"]:
                repeat_path = path.with_name(path.stem + "-repeat.pt")
                repeated, proof = torch.load(repeat_path, map_location="cpu", weights_only=True), read(repeat_path.with_suffix(".json"))
                require_request(codec, plan, target, midi, proof["receipt"])
                if (not proof["exact"] or digest(repeat_path) != proof["file_sha256"] or not torch.equal(repeated["audio"], saved["audio"])
                        or proof["receipt"] != outcomes[proof["receipt"]["phase"], proof["receipt"]["attempt"]]):
                    raise ValueError("retained balanced repeat is not exact")
                repeats += 1
        frontend.require_frozen()
        if tensor_digest(model.state_dict()) != original:
            raise ValueError("data audit changed the inverse model")
    finally:
        monitor.close()
    elapsed = time() - started
    if elapsed > 600 or not monitor.peak_bytes or monitor.peak_bytes > LIMITS["process_tree_bytes"] or monitor.errors:
        raise ValueError("balanced data audit exceeded ten minutes or 4 GiB")
    report = {"plan_id": canonical_sha256(plan), "verified": True, "source_recordings": 18, "exact_repeat_pairs": repeats,
              "features_reconstructed_exactly": 18, "evidence_sha256": evidence, "recovered_timeouts": recovered,
              "elapsed_seconds": elapsed, "peak_process_tree_rss_bytes": monitor.peak_bytes, "rss_sampling_errors": monitor.errors,
              "new_native_requests": 0, "experimental_updates": 0, "test_split_used": False, "qualified_for_release": False}
    write_json(output / "verification.json", report)
    return report
