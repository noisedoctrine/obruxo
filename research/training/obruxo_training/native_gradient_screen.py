"""Bounded development qualification of measured native sound-loss gradients."""

import json
from pathlib import Path
import platform
from time import time

import numpy as np
import torch

from obruxo_data.hashing import canonical_sha256, sha256_bytes
from obruxo_data.midi import Performance

from .audit import write_json
from .benchmark import MemoryMonitor
from .codec import PilotTarget
from .evaluate import audio_distance
from .native_gradient import guided_candidates, measure_native_gradient, native_gradient_spec
from .performances import initial_bank
from .render_profile import BASIC_PREROLL_FRAMES, basic_render_profile
from .spectral import SpectralTargets, spectral_loss
from .spectral_controls import BasicControlAdapter
from .temporal_proxy import perturbed_targets
from .temporal_proxy_data import NativeSession, category_balanced_entries


def make_plan(pool, *, preroll_frames=BASIC_PREROLL_FRAMES):
    profile = basic_render_profile(preroll_frames)
    entries = category_balanced_entries(pool, "validation", 12)
    cases = []
    for index, entry in enumerate(entries):
        start = perturbed_targets(pool.codec, entry.target, index)["start"]
        choices = list(start.categorical)
        if index % 3:
            head = index % 3 - 1
            choices[head] = (choices[head] + 1) % (3 if head == 0 else 2)
        # Preserve raw values: inactive controls may become active in a categorical probe.
        start = PilotTarget(tuple(torch.tensor(start.continuous, dtype=torch.float32).tolist()), tuple(choices))
        for role, offset in (("reference", 0), ("transfer", 4)):
            performance = initial_bank()[offset + index % 4].performance()
            cases.append({"index": index, "role": role, "state_id": entry.state_id, "source": entry.target.to_dict(),
                          "start": start.to_dict(), "performance": performance.to_dict()})
    names = ("native_gradient.py", "native_gradient_screen.py", "codec.py", "spectral.py", "spectral_controls.py",
             "temporal_proxy_data.py", "temporal_proxy.py", "evaluate.py", "reproducibility.py", "stream.py", "render_profile.py")
    paths = [Path(__file__).with_name(name) for name in names]
    paths += [Path(__file__).parents[2] / "data_generation/obruxo_data/render" / name for name in ("vita.py", "base.py", "midi_transport.py", "qa.py")]
    paths += [Path(__file__).parents[1] / "reports/scripts/run_native_gradient_screen.py"]
    return {"format": "basic_native_gradient_qualification_v1", "pool_id": pool.pool_id,
            "manifest_id": pool.manifest.manifest_id, "codec": pool.codec.to_dict(),
            "contract": BasicControlAdapter(pool.codec).contract(), "estimator": native_gradient_spec(), "cases": cases, "render_profile": profile,
            "source": {path.name: sha256_bytes(path.read_bytes().replace(b"\r\n", b"\n")) for path in paths},
            "runtime": {"python": platform.python_version(), "torch": str(torch.__version__), "numpy": np.__version__,
                        "device": "cpu", "cpu_threads": 2, "precision": "float32"},
            "limits": {"qualification_attempts": 624, "wall_seconds": 2700 if preroll_frames else 1800, "process_tree_bytes": 8 * 1024**3,
                       "archive_bytes": 32 * 1024**2},
            "update": {"continuous_radius": .05, "continuous_scaling": "divide_by_gradient_max_absolute",
                       "category_initial_logit_margin": .1, "category_step": 1., "category_temperature": 1.,
                       "category_scaling": "each_head_divide_by_gradient_max_absolute",
                       "qualification_choices": "deterministic_hard_argmax_with_softmax_backward",
                       "training_compatible_choices": "adapter_hard_Gumbel_softmax",
                       "steps": 1, "backtracking": False, "checkpoint_selection": False},
            "gates": {"roles": ["reference", "transfer"], "cases_per_role": 12, "methods": ["continuous", "joint"],
                      "minimum_improved_fraction": .6, "mean_mrstft_delta_below": 0., "mean_spectral_delta_below": 0.,
                      "repeat_exact": True, "all_native_accepted": True},
            "test_split_used": False, "inverse_optimizer_updates": 0, "learned_helper_used": False}


def qualification_summary(cases, plan):
    roles, complete = {}, True
    expected = [(row["index"], row["role"]) for row in plan["cases"]]
    actual = [(row["index"], row["role"]) for row in cases]
    complete = sorted(actual) == sorted(expected) and len(set(actual)) == len(actual)
    for role in plan["gates"]["roles"]:
        selected = [row for row in cases if row["role"] == role]
        methods = {}
        for method in plan["gates"]["methods"]:
            deltas = [row["scores"][method]["mrstft"] - row["scores"]["start"]["mrstft"] for row in selected]
            spectral = [row["scores"][method]["spectral"] - row["scores"]["start"]["spectral"] for row in selected]
            improved = sum(value < -1e-7 for value in deltas)
            mean_delta, mean_spectral = (float(np.mean(deltas)), float(np.mean(spectral))) if selected else (None, None)
            passed = (len(selected) == plan["gates"]["cases_per_role"]
                      and improved / plan["gates"]["cases_per_role"] >= plan["gates"]["minimum_improved_fraction"]
                      and mean_delta < 0 and mean_spectral < 0)
            methods[method] = {"improved": improved, "planned": plan["gates"]["cases_per_role"],
                               "mean_mrstft_delta": mean_delta, "mean_spectral_delta": mean_spectral, "pass": bool(passed)}
        roles[role] = methods
    repeats = complete and all(row["repeat_exact"] for row in cases)
    passed = repeats and all(method["pass"] for role in roles.values() for method in role.values())
    return {"complete": complete, "repeat_exact": repeats, "roles": roles, "direction_gates_passed": passed,
            "claim": "local_direction_feasibility_not_trained_inverse_quality"}


def qualify_case(case, session):
    codec, measure = session.codec, SpectralTargets()
    source, start = PilotTarget.from_dict(case["source"]), PilotTarget.from_dict(case["start"])
    performance, rows = Performance.from_dict(case["performance"]), []

    def render(candidate, label):
        audio, receipt = session.render(candidate, performance, "qualification", f"{case['index']}:{case['role']}:{label}")
        if audio is None:
            raise RuntimeError("native gradient qualification render rejected")
        rows.append({"label": label, "candidate": candidate.to_dict(), "request_id": receipt["request_id"],
                     "audio_sha256": receipt["qa"]["audio_float32_sha256"]})
        return audio

    reference = render(source, "source")
    repeated = render(source, "repeat")
    if not np.array_equal(reference, repeated):
        raise RuntimeError("fixed-phase source repeat changed")
    target, scores = measure(torch.from_numpy(reference.copy()).T[None]), {}

    def score(candidate, label):
        audio = render(candidate, label)
        components = spectral_loss(measure(torch.from_numpy(audio.copy()).T[None]), target)
        value = {"spectral": float(components["loss"].item()), "mrstft": audio_distance(reference, audio)["mrstft"]}
        rows[-1]["scores"] = value
        return value

    def evaluate(candidate):
        label = "start" if len(rows) == 2 else f"probe-{len(rows) - 3:02d}"
        result = score(candidate, label)
        if label == "start":
            scores["start"] = result
        return result["spectral"]

    estimate = measure_native_gradient(codec, start, evaluate)
    candidates, gradients = guided_candidates(codec, estimate)
    for method, candidate in candidates.items():
        scores[method] = score(candidate, method)
    return {"index": case["index"], "role": case["role"], "state_id": case["state_id"], "scores": scores,
            "repeat_exact": True, "estimate": estimate.to_dict(), "gradients": gradients,
            "category_changed": candidates["joint"].categorical != start.categorical, "renders": rows}


def run_screen(pool, output, *, execute=False, preroll_frames=BASIC_PREROLL_FRAMES):
    torch.set_num_threads(2)
    plan = make_plan(pool, preroll_frames=preroll_frames)
    plan_id = canonical_sha256(plan)
    if not execute:
        output.mkdir(parents=True, exist_ok=False)
        write_json(output / "plan.json", plan)
        return {"plan_id": plan_id, "cases": len(plan["cases"]), "maximum_native_requests": 624}
    if json.loads((output / "plan.json").read_text()) != plan or (output / "run.json").exists():
        raise ValueError("plan changed or this qualification was already attempted")
    write_json(output / "run.json", {"started_unix": time(), "plan_id": plan_id})
    monitor = MemoryMonitor()
    monitor.start()
    session, cases, failure = NativeSession(output, plan, pool.codec, monitor), [], None
    try:
        for case in plan["cases"]:
            cases.append(qualify_case(case, session))
            write_json(output / "cases.json", cases)
            session.check_limits()
            if sum(path.stat().st_size for path in output.rglob("*") if path.is_file()) + 1024**2 > plan["limits"]["archive_bytes"]:
                raise RuntimeError("native gradient archive budget exhausted")
            print(json.dumps({"completed": len(cases), "planned": 24,
                              "spectral_delta": {method: cases[-1]["scores"][method]["spectral"] - cases[-1]["scores"]["start"]["spectral"]
                                                 for method in ("continuous", "joint")}}), flush=True)
    except Exception as error:
        failure = f"{type(error).__name__}: {error}"
        raise
    finally:
        monitor.close()
        elapsed = time() - session.started
        archive = sum(path.stat().st_size for path in output.rglob("*") if path.is_file())
        attempts = sum(row["event"] == "attempt" for row in session.events)
        outcomes = [row for row in session.events if row["event"] == "outcome"]
        accepted = len(outcomes) == attempts and all(row["status"] == "accepted" for row in outcomes)
        resources = (monitor.errors == 0 and monitor.peak_bytes is not None
                     and monitor.peak_bytes <= plan["limits"]["process_tree_bytes"] and elapsed <= plan["limits"]["wall_seconds"]
                     and archive + 1024**2 <= plan["limits"]["archive_bytes"] and attempts <= plan["limits"]["qualification_attempts"])
        result = qualification_summary(cases, plan)
        summary = {"plan_id": plan_id, "plan": plan, "result": result, "cases": cases, "failure": failure,
                   "qualified_native_gradient_path": result["direction_gates_passed"] and accepted and resources and failure is None,
                   "elapsed_seconds": elapsed, "peak_process_tree_rss_bytes": monitor.peak_bytes,
                   "rss_sampling_errors": monitor.errors, "native_attempts": attempts, "all_native_accepted": accepted,
                   "within_resource_limits": resources, "archive_bytes_before_summary": archive,
                   "renderer_identity": session.renderer_identity, "test_split_used": False, "inverse_optimizer_updates": 0}
        write_json(output / "summary.json", summary)
    return {"qualified_native_gradient_path": summary["qualified_native_gradient_path"], "native_attempts": attempts}
