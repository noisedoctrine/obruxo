"""Reconstruct the production-choice native check from recorded queries and gradients."""

import json
from time import perf_counter

import torch

from obruxo_data.hashing import canonical_sha256, sha256_bytes
from obruxo_data.midi import Performance
from obruxo_data.render.base import RenderRequest
from obruxo_data.render.qa import audio_float32_sha256

from . import category_scale_refresh as refresh
from .audit import write_json
from .benchmark import MemoryMonitor
from .codec import PilotTarget
from .evaluate import audio_distance
from .features import tensor_digest
from .production_controls import ProductionControlAdapter
from .production_direction import direction_summary, guided_prediction
from .production_native_screen import LIMITS, make_plan
from .reproducibility import fixed_generator_phases
from .spectral import SpectralTargets, spectral_loss


def verify_screen(source, pool, *inputs):
    plan, model, features = make_plan(pool, *inputs)
    summary = json.loads((source / "summary.json").read_text())
    cases = json.loads((source / "cases.json").read_text())
    if (json.loads((source / "plan.json").read_text()) != plan or summary["plan"] != plan
            or summary["plan_id"] != canonical_sha256(plan) or summary["cases"] != cases or summary["failure"] is not None
            or len(cases) != len(plan["cases"])):
        raise ValueError("incomplete or changed native production-choice evidence")
    events = [json.loads(line) for line in (source / "renders.jsonl").read_text().splitlines()]
    attempts = [row for row in events if row["event"] == "attempt"]
    outcomes = [row for row in events if row["event"] == "outcome"]
    flattened = [row for case in cases for row in case["renders"]]
    if len(attempts) != len(outcomes) or len(attempts) != len(flattened) or len(attempts) != summary["native_attempts"]:
        raise ValueError("native production-choice attempt/outcome accounting mismatch")
    adapter = ProductionControlAdapter(pool.codec, scales=plan["adapter"]["fixed_scales"])
    cursor, source_digests, candidate_digests = 0, {}, {}
    measure = SpectralTargets()
    for planned, case in zip(plan["cases"], cases, strict=True):
        if any(case[key] != planned[key] for key in ("index", "role", "state_id")):
            raise ValueError("native production-choice case identity changed")
        rows = case["renders"]
        for row in rows:
            attempt, receipt = attempts[cursor], outcomes[cursor]
            candidate = PilotTarget.from_dict(row["candidate"])
            if candidate != pool.codec.canonical_target(candidate):
                raise ValueError("native production-choice candidate is not canonical")
            preset = pool.codec.decode(candidate)
            request = RenderRequest(preset_json=fixed_generator_phases(preset).decode(), performance=Performance.from_dict(planned["performance"]),
                                    renderer_id=pool.codec.contract.renderer_id, tail_seconds=pool.codec.contract.tail_seconds,
                                    preroll_frames=plan["render_profile"]["preroll_frames"])
            identity = {"phase": "qualification", "label": f"{planned['index']}:{planned['role']}:{row['label']}",
                        "attempt": cursor, "request_id": request.request_id, "source_sha256": sha256_bytes(preset),
                        "controlled_sha256": sha256_bytes(request.preset_json.encode())}
            if (any(attempt.get(key) != value or receipt.get(key) != value for key, value in identity.items())
                    or receipt["status"] != "accepted" or receipt["renderer_identity"] != plan["expected_renderer_identity"]
                    or row["request_id"] != request.request_id or row["audio_sha256"] != receipt["qa"]["audio_float32_sha256"]):
                raise ValueError("native production-choice request/receipt changed")
            cursor += 1
        if (len(rows) < 5 or rows[0]["label"] != "source" or rows[1]["label"] != "repeat"
                or any(row["candidate"] != planned["source"] or row["audio_sha256"] != planned["target_audio_sha256"] for row in rows[:2])
                or not case["repeat_exact"]):
            raise ValueError("native production-choice source repeat changed")
        expected_file = f"sources/{planned['index']:02d}-{planned['role']}.pt"
        if case["source_tensor"] != expected_file or refresh.digest(source / expected_file) != case["source_tensor_sha256"]:
            raise ValueError("native production-choice source tensor changed")
        saved = torch.load(source / expected_file, map_location="cpu", weights_only=True)
        audio = saved["audio"]
        if (set(saved) != {"audio"} or audio.dtype != torch.float32 or audio.shape != (planned["source_samples"], 2)
                or audio_float32_sha256(audio.numpy()) != planned["target_audio_sha256"]):
            raise ValueError("native production-choice source audio hash changed")
        source_digests[expected_file] = case["source_tensor_sha256"]
        target = measure(audio.T[None])
        for row in (rows[2], *rows[-2:]):
            expected_audio = f"audits/{planned['index']:02d}-{planned['role']}-{row['label']}.pt"
            if row["audio_tensor"] != expected_audio or refresh.digest(source / expected_audio) != row["audio_tensor_sha256"]:
                raise ValueError("native production-choice candidate waveform changed")
            saved_candidate = torch.load(source / expected_audio, map_location="cpu", weights_only=True)
            candidate_audio = saved_candidate["audio"]
            if (set(saved_candidate) != {"audio"} or candidate_audio.dtype != torch.float32 or candidate_audio.shape != audio.shape
                    or audio_float32_sha256(candidate_audio.numpy()) != row["audio_sha256"]):
                raise ValueError("native production-choice candidate audio hash changed")
            rescored = {"spectral": float(spectral_loss(measure(candidate_audio.T[None]), target)["loss"]),
                        "mrstft": audio_distance(audio.numpy(), candidate_audio.numpy())["mrstft"]}
            if rescored != row["scores"]:
                raise ValueError("native production-choice sound score cannot be reconstructed")
            candidate_digests[expected_audio] = row["audio_tensor_sha256"]
        measured = {PilotTarget.from_dict(row["candidate"]): row["scores"]["spectral"] for row in rows[2:-2]}
        if len(measured) != len(rows) - 4:
            raise ValueError("duplicate native production-choice probes")

        def evaluate(index, candidate):
            if index != 0:
                raise ValueError("unexpected native production-choice batch index")
            return measured[candidate]

        direction = guided_prediction(model, features[planned["index"], planned["role"]], adapter, evaluate)
        if (direction != case["direction"] or direction["raw_prediction"] != planned["raw_prediction"]
                or direction["production_patch"] != planned["production_patch"] or len(rows) != direction["estimate"]["queries"] + 4
                or rows[2]["scores"] != case["scores"]["start"] or rows[2]["candidate"] != planned["production_patch"]):
            raise ValueError("native production-choice gradients/forward cannot be reconstructed")
        for row, method in zip(rows[-2:], ("continuous", "joint"), strict=True):
            if row["label"] != method or row["candidate"] != direction["candidates"][method] or row["scores"] != case["scores"][method]:
                raise ValueError("native production-choice local step changed")
        changed = direction["candidates"]["joint"]["categorical"] != direction["production_patch"]["categorical"]
        if changed != case["category_changed"]:
            raise ValueError("native production-choice category diagnostic changed")
    result = direction_summary(cases, plan)
    archive = sum(path.stat().st_size for path in source.rglob("*") if path.is_file())
    resources = (summary["rss_sampling_errors"] == 0 and 0 < summary["peak_process_tree_rss_bytes"] <= LIMITS["process_tree_bytes"]
                 and summary["elapsed_seconds"] <= LIMITS["wall_seconds"] and archive + 1024**2 <= LIMITS["archive_bytes"]
                 and len(attempts) <= LIMITS["qualification_attempts"])
    unchanged = tensor_digest(model.state_dict()) == plan["inference"]["model_sha256"]
    qualified = result["direction_gates_passed"] and resources and unchanged
    if (summary["result"] != result or summary["qualified_production_direction"] != qualified or summary["within_resource_limits"] != resources
            or summary["renderer_identity"] != plan["expected_renderer_identity"] or not summary["all_native_accepted"]
            or summary["weights_unchanged"] != unchanged or summary["optimizer_updates"] != 0 or summary["test_split_used"]
            or summary["qualified_for_release"]):
        raise ValueError("native production-choice qualification decision changed")
    names = ("plan.json", "run.json", "cases.json", "renders.jsonl", "summary.json")
    return {**summary, "all_model_gradients_and_local_steps_reconstructed": True, "all_requests_reconstructed": True,
            "source_audio_hashes_verified": len(source_digests), "candidate_audio_rescored": len(candidate_digests), "gradient_probe_audio_rescored": False,
            "retained_archive_bytes": archive, "evidence_sha256": {name: refresh.digest(source / name) for name in names},
            "source_tensor_sha256": source_digests, "candidate_tensor_sha256": candidate_digests,
            "native_requests_added": 0, "optimizer_updates_added": 0}


def run_verification(source, pool, *inputs):
    output = source / "verification.json"
    if output.exists():
        raise ValueError("native production-choice verification already exists; preserve evidence")
    started, monitor = perf_counter(), MemoryMonitor()
    monitor.start()
    try:
        with refresh.cpu_settings():
            result = verify_screen(source, pool, *inputs)
    finally:
        monitor.close()
    elapsed = perf_counter() - started
    bound = result["retained_archive_bytes"] + 2 * len(json.dumps(result, indent=2).encode()) + 1024**2
    resources = (0 < elapsed < 600 and monitor.peak_bytes is not None and 0 < monitor.peak_bytes < 4 * 1024**3
                 and monitor.errors == 0 and bound <= LIMITS["archive_bytes"])
    result["verification_resources"] = {"elapsed_seconds": elapsed, "peak_process_tree_rss_bytes": monitor.peak_bytes,
                                        "rss_sampling_errors": monitor.errors, "archive_bytes_upper_bound": bound,
                                        "within_resource_limits": resources, "wall_limit_seconds": 600, "memory_limit_bytes": 4 * 1024**3}
    result["verified"] = resources
    write_json(output, result)
    if not resources:
        raise ValueError("native production-choice verification exceeded its resource limits")
    return {key: result[key] for key in ("verified", "qualified_production_direction", "native_attempts", "verification_resources")}
