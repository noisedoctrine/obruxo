"""Fit wider training observations, then compare fixed old/new input panels."""

from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path
from time import time

import torch

from obruxo_data.hashing import canonical_sha256, sha256_bytes

from .audit import write_json
from .basic_bootstrap import append, rows
from .benchmark import MemoryMonitor
from .checkpoints import load_training_checkpoint
from .evaluate import _evaluate_models, load_evaluation_checkpoint
from .features import tensor_digest
from .native_fit import require_qualification
from .profiled_sampler import profile_metadata, training_sampler
from .render_audit import AuditedFactory
from .stream import NativeRendererFactory, StreamConfig
from .trainer import TrainConfig, train_pilot


def verified_parent(root, audit_path):
    plan, summary = [json.loads((root / name).read_text()) for name in ("plan.json", "summary.json")]
    audit = json.loads(audit_path.read_text())
    if (not summary["complete"] or summary["plan_id"] != canonical_sha256(plan) or audit["plan_id"] != summary["plan_id"]
            or not {"plan.json", "summary.json"} <= audit["evidence_sha256"].keys()):
        raise ValueError("parent plan or audit is incomplete")
    for name, digest in audit["evidence_sha256"].items():
        if sha256_bytes((root / name).read_bytes()) != digest:
            raise ValueError("audited parent evidence changed")
    return plan, summary, audit


def make_plan(pool, qualification_path, initial_root, initial_audit, input_root, input_audit):
    qualification = json.loads(qualification_path.read_text())
    require_qualification(qualification, preroll_frames=32768)
    old_plan, old, old_audit = verified_parent(initial_root, initial_audit)
    input_plan, inputs, inputs_audit = verified_parent(input_root, input_audit)
    model_path = initial_root / "fit-000/model.pt"
    if (old_plan["pool_id"] != pool.pool_id or input_plan["pool_id"] != pool.pool_id or not old["preflight_passed"]
            or not old_audit["model_optimizer_batch_and_global_rng_exact"] or old_audit["replayed_updates"] != 16000
            or old_audit["native_renders_added"] != 0 or "fit-000/model.pt" not in old_audit["evidence_sha256"]
            or not inputs["passed"] or inputs["accepted"] != 76 or inputs_audit["exact_repeat_pairs"] != 36
            or inputs_audit["historical_hashes_matched"] != 4 or not inputs_audit["all_varied_source_requests_reconstructed"]
            or inputs_audit["native_renders_added"] != 0 or inputs_audit["optimizer_updates_added"] != 0
            or not old_plan["renderer_identity"] == input_plan["renderer_identity"] == qualification["renderer_identity"]):
        raise ValueError("varied initialization requires compatible verified weights and native input check")
    raw = torch.load(model_path, map_location="cpu", weights_only=True)
    if tensor_digest(raw["model"]) != old["model_state_sha256"]:
        raise ValueError("initial inference weights differ from the audited model")
    config = TrainConfig(steps=8000, replay_capacity=808, replay_mib=32, batch_size=32, learning_rate=.001,
                         seed=0, checkpoint_every=250, render_budget=808, cpu_threads=2, prediction_scope="basic",
                         preroll_frames=32768, sampling_policy="paired_views", input_performances="varied")
    sampler, old_sampler = training_sampler(pool, config), training_sampler(pool, TrainConfig(**old_plan["config"]))
    recipes = [sampler.recipe_at(index) for index in range(sampler.epoch_size)]
    if len(recipes) != 808 or len({recipe.state_id for recipe in recipes}) != 404:
        raise ValueError("varied initialization requires the original 404-state population with both views")
    old_recipes = [old_sampler.recipe_at(index) for index in range(8)]
    held_by_state = {recipe.state_id: recipe for recipe in recipes if recipe.input_view == 0}
    panels = {"original": old_recipes, "wide": [held_by_state[recipe.state_id] for recipe in old_recipes]}
    names = ("varied_bootstrap.py", "render_audit.py", "input_performances.py", "profiled_sampler.py", "trainer.py",
             "checkpoints.py", "model.py", "features.py", "extended_features.py", "evaluate.py", "paired.py", "stream.py")
    return {"experiment": "varied_basic_initialization_v1", "pool_id": pool.pool_id, "manifest_id": pool.manifest.manifest_id,
            "config": asdict(config), "original_sampler_config": old_plan["config"],
            "initial_model": {"file_sha256": sha256_bytes(model_path.read_bytes()), "model_state_sha256": tensor_digest(raw["model"])},
            "parents": {"initial_plan_id": old["plan_id"], "input_plan_id": inputs["plan_id"],
                        "initial_audit_sha256": sha256_bytes(initial_audit.read_bytes()), "input_audit_sha256": sha256_bytes(input_audit.read_bytes())},
            "qualification_sha256": sha256_bytes(qualification_path.read_bytes()), "renderer_identity": qualification["renderer_identity"],
            **profile_metadata(32768), "observation_contract": sampler.observation_contract,
            "sources": [{"sequence_id": recipe.sequence_id, "state_id": recipe.state_id, "recipe_id": recipe.recipe_id,
                         "request_id": recipe.render_request().request_id, "performance_id": recipe.performance_id} for recipe in recipes],
            "panels": {name: [{"sequence_id": recipe.sequence_id, "state_id": recipe.state_id, "recipe_id": recipe.recipe_id,
                              "performance_id": recipe.performance_id} for recipe in panel] for name, panel in panels.items()},
            "source": {name: sha256_bytes(Path(__file__).with_name(name).read_bytes().replace(b"\r\n", b"\n")) for name in names},
            "limits": {"wall_seconds": 7200, "process_tree_bytes": 4 * 1024**3, "archive_bytes": 512 * 1024**2,
                       "training_attempts": 810, "evaluation_attempts": 202, "optimizer_updates": 8000},
            "planned_requests": {"training": 808, "evaluation": 192, "total": 1000},
            "gates": {"wide_candidate_baseline_ratio_below_both_roles": .9, "original_candidate_baseline_ratio_at_most_both_roles": 1.1,
                      "candidate_template_ratio_below_all_panels_roles": .9, "all_outcomes_accepted": True, "all_source_repeats_exact": True},
            "recovery": "explicit_resume_original_deadline_two_source_and_ten_evaluation_extra_attempts",
            "selection": "final_update_only_all_acoustic_gates_no_validation_or_parameter_accuracy_admission",
            "quality_claim": "training_input_initialization_comparison_not_A2_or_release", "test_split_used": False}


def acoustic_gates(plan, audits):
    ratios, templates = {}, {}
    for panel, audit in audits.items():
        ratios[panel], templates[panel] = {}, {}
        count = len(plan["panels"][panel])
        for role in ("reference", "transfer"):
            scores = {name: audit["aggregates"][name][role] for name in ("baseline", "varied", "fixed_template", "source_repeat")}
            if any(score["scored"] != count or score["planned"] != count or score["mean_mrstft"] is None for score in scores.values()):
                raise ValueError("acoustic comparison contains incomplete cases")
            ratios[panel][role] = (scores["varied"]["mean_mrstft"] + 1e-12) / (scores["baseline"]["mean_mrstft"] + 1e-12)
            templates[panel][role] = (scores["varied"]["mean_mrstft"] + 1e-12) / (scores["fixed_template"]["mean_mrstft"] + 1e-12)
    gates = plan["gates"]
    passed = (all(value < gates["wide_candidate_baseline_ratio_below_both_roles"] for value in ratios["wide"].values())
              and all(value <= gates["original_candidate_baseline_ratio_at_most_both_roles"] for value in ratios["original"].values())
              and all(value < gates["candidate_template_ratio_below_all_panels_roles"] for panel in templates.values() for value in panel.values())
              and all(audit["aggregates"]["source_repeat"][role]["mean_mrstft"] == 0 for audit in audits.values() for role in ("reference", "transfer")))
    return {"candidate_baseline_ratios": ratios, "candidate_template_ratios": templates, "acoustic_gates_passed": passed}


def run(pool, plan, initial_root, output, *, execute=False, resume=False, renderer_factory=None):
    output = Path(output).resolve()
    if output.exists():
        if json.loads((output / "plan.json").read_text()) != plan:
            raise ValueError("frozen varied initialization plan changed")
    else:
        output.mkdir(parents=True)
        write_json(output / "plan.json", plan)
    if not execute:
        return {"plan_id": canonical_sha256(plan), "executed": False}
    if (output / "run.json").exists() != resume:
        raise ValueError("existing run needs explicit resume; a new run cannot resume")
    model_path = initial_root / "fit-000/model.pt"
    if sha256_bytes(model_path.read_bytes()) != plan["initial_model"]["file_sha256"]:
        raise ValueError("initial model file changed")
    if not resume:
        write_json(output / "run.json", {"started_unix": time(), "plan_id": canonical_sha256(plan)})
    started = json.loads((output / "run.json").read_text())["started_unix"]
    prior_status = json.loads((output / "status.json").read_text()) if (output / "status.json").exists() else {}
    monitor = MemoryMonitor()
    monitor.start()

    def peak():
        return max(monitor.peak_bytes or 0, prior_status.get("peak_process_tree_rss_bytes", 0))

    def check(*_args):
        if time() - started >= plan["limits"]["wall_seconds"]:
            raise RuntimeError("varied initialization original wall-time budget exhausted")
        if peak() > plan["limits"]["process_tree_bytes"]:
            raise RuntimeError("varied initialization memory budget exhausted")
        if sum(path.stat().st_size for path in output.rglob("*") if path.is_file()) > plan["limits"]["archive_bytes"]:
            raise RuntimeError("varied initialization archive budget exhausted")

    def progress(event):
        check()
        append(output / "progress.jsonl", {"elapsed_seconds": time() - started, **event})
        print(json.dumps(event), flush=True)

    base = renderer_factory or NativeRendererFactory(renderer_id=pool.codec.contract.renderer_id)
    try:
        check()
        segments = sorted(output.glob("fit-*"))
        previous = segments[-1] if segments else None
        if previous and (previous / "summary.json").exists() and json.loads((previous / "summary.json").read_text())["complete"]:
            fit_root, fit = previous, json.loads((previous / "summary.json").read_text())
        else:
            fit_root = output / f"fit-{len(segments):03d}"
            fit = train_pilot(pool, fit_root, config=TrainConfig(**plan["config"]), stream_config=StreamConfig(prefetch_clips=1),
                              renderer_factory=AuditedFactory(str(output), "training", base),
                              initial_model=model_path if previous is None else None,
                              resume=previous / "checkpoint.pt" if previous else None, after_step=check, progress=progress)
        saved = load_training_checkpoint(fit_root / "checkpoint.pt")
        if (fit["renderer_identity"] != plan["renderer_identity"] or fit["initial_model"]["file_sha256"] != plan["initial_model"]["file_sha256"]
                or [item["recipe_id"] for item in saved["replay"]] != [spec["recipe_id"] for spec in plan["sources"]]
                or fit["totals"]["accepted"] != len(plan["sources"]) or saved["step"] != plan["config"]["steps"]):
            raise ValueError("varied initialization failed frozen source/weight/update coverage")
        old_saved, old_model, frontend, old_sampler = load_evaluation_checkpoint(initial_root / "fit-000/checkpoint.pt", pool)
        new_saved, new_model, new_frontend, new_sampler = load_evaluation_checkpoint(fit_root / "checkpoint.pt", pool)
        if (frontend.describe() != new_frontend.describe() or old_saved["renderer_identity"] != plan["renderer_identity"]
                or tensor_digest(old_saved["model"]) != plan["initial_model"]["model_state_sha256"]):
            raise ValueError("acoustic comparison frontend or renderer changed")
        models = {"baseline": (old_saved, old_model), "varied": (new_saved, new_model)}
        artifacts = {name: {"file_sha256": sha256_bytes(path.read_bytes())}
                     for name, path in (("baseline", initial_root / "fit-000/checkpoint.pt"), ("varied", fit_root / "checkpoint.pt"))}
        audits = {}
        for panel, sampler in (("original", old_sampler), ("wide", new_sampler)):
            audit_root = output / f"audit-{panel}"
            audits[panel] = _evaluate_models(pool, models, frontend, sampler, audit_root, count=len(plan["panels"][panel]), split="train", curve=True, comparison=True,
                                            artifacts=artifacts, retrieval={"replay": saved["replay"], "source": {"checkpoint_sha256": artifacts["varied"]["file_sha256"]}},
                                            training_sequence_ids=[row["sequence_id"] for row in plan["panels"][panel]],
                                            renderer_factory=AuditedFactory(str(output), "evaluation", base), progress=progress, resume=audit_root.exists())
        receipts = [{"phase": "training", **row} for segment in sorted(output.glob("fit-*")) for row in rows(segment / "renders.jsonl")]
        receipts += [{"phase": "evaluation", **row} for panel in audits for row in rows(output / f"audit-{panel}/renders.jsonl")]
        attempts = rows(output / "attempts.jsonl")
        matched = Counter((row["phase"], row["request_id"]) for row in attempts) == Counter(
            (row["phase"], row.get("request_id", row.get("provenance", {}).get("request_id"))) for row in receipts)
        check()
        summary = {"plan_id": canonical_sha256(plan), "complete": True, "training": fit, "audits": audits, **acoustic_gates(plan, audits),
                   "attempts": len(attempts), "receipts": len(receipts), "attempt_outcome_counts_match": matched,
                   "all_outcomes_accepted": matched and all(row["status"] == "accepted" for row in receipts),
                   "attempts_by_phase": dict(Counter(row["phase"] for row in attempts)), "elapsed_seconds": time() - started,
                   "peak_process_tree_rss_bytes": peak(), "rss_sampling_errors": monitor.errors + prior_status.get("rss_sampling_errors", 0),
                   "archive_bytes": sum(path.stat().st_size for path in output.rglob("*") if path.is_file()),
                   "model_state_sha256": tensor_digest(saved["model"]), "qualified_for_release": False, "test_split_used": False}
        summary["passed"] = summary["acoustic_gates_passed"] and summary["all_outcomes_accepted"] and monitor.peak_bytes is not None and summary["rss_sampling_errors"] == 0
        write_json(output / "summary.json", summary)
        write_json(output / "status.json", {"complete": True, "passed": summary["passed"], "peak_process_tree_rss_bytes": peak()})
        return summary
    except BaseException as error:
        write_json(output / "status.json", {"complete": False, "error_type": type(error).__name__, "elapsed_seconds": time() - started,
                                            "peak_process_tree_rss_bytes": peak(), "rss_sampling_errors": monitor.errors})
        raise
    finally:
        monitor.close()
