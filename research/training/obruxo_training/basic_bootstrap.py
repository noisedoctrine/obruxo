"""Costed eight-source check before refreshing the basic model's initialization."""

from collections import Counter
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import platform
from time import time

import numpy as np
import torch

from obruxo_data.hashing import canonical_sha256, sha256_bytes

from .audit import write_json
from .benchmark import MemoryMonitor
from .checkpoints import load_training_checkpoint
from .evaluate import evaluate_pilot
from .features import tensor_digest
from .native_fit import require_qualification
from .profiled_sampler import profile_metadata, training_sampler
from .stream import NativeRendererFactory, StreamConfig
from .trainer import TrainConfig, train_pilot


def append(path, row):
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(row, allow_nan=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def make_plan(pool, qualification_path):
    qualification = json.loads(qualification_path.read_text())
    require_qualification(qualification, preroll_frames=32768)
    config = TrainConfig(steps=800, replay_capacity=8, replay_mib=8, batch_size=8, learning_rate=.001,
                         seed=0, checkpoint_every=50, render_budget=8, cpu_threads=2,
                         sampling_policy="state_round_robin", prediction_scope="basic", preroll_frames=32768)
    sampler = training_sampler(pool, config)
    recipes = [sampler.recipe_at(index) for index in range(8)]
    if len({recipe.state_id for recipe in recipes}) != 8:
        raise ValueError("bootstrap needs eight distinct training states")
    names = ("basic_bootstrap.py", "profiled_sampler.py", "trainer.py", "checkpoints.py", "model.py", "features.py", "evaluate.py")
    return {"experiment": "settled_basic_bootstrap_preflight_v1", "pool_id": pool.pool_id,
            "manifest_id": pool.manifest.manifest_id, "config": asdict(config),
            "qualification_sha256": sha256_bytes(qualification_path.read_bytes()),
            "renderer_identity": qualification["renderer_identity"], **profile_metadata(32768),
            "sources": [{"sequence_id": recipe.sequence_id, "state_id": recipe.state_id, "recipe_id": recipe.recipe_id,
                         "request_id": recipe.render_request().request_id, "performance_id": recipe.performance_id} for recipe in recipes],
            "source": {name: sha256_bytes(Path(__file__).with_name(name).read_bytes().replace(b"\r\n", b"\n")) for name in names},
            "limits": {"wall_seconds": 1800, "process_tree_bytes": 4 * 1024**3, "archive_bytes": 256 * 1024**2,
                       "training_attempts": 10, "evaluation_attempts": 90, "optimizer_updates": 800},
            "planned_requests": {"training": 8, "evaluation": 80, "total": 88},
            "recovery": "explicit_resume_original_deadline_up_to_two_source_and_ten_evaluation_extra_attempts",
            "selection": "final_only_no_held_out_states", "test_split_used": False,
            "runtime": {"python": platform.python_version(), "torch": str(torch.__version__), "numpy": np.__version__,
                        "device": "cpu", "cpu_threads": 2, "native_workers": 1, "precision": "float32"},
            "gates": {"trainer_tiny_overfit": True, "model_vs_template_mrstft_ratio_both_roles": .9,
                      "all_outcomes_accepted": True, "all_source_repeats_exact": True},
            "quality_claim": "training_fit_preflight_only_not_a_released_or_sound_trained_model"}


@dataclass(frozen=True)
class AuditedFactory:
    output: str
    phase: str
    base: object

    def __call__(self):
        return AuditedRenderer(self)


class AuditedRenderer:
    def __init__(self, factory):
        self.factory = factory

    def render(self, request):
        root = Path(self.factory.output)
        plan = json.loads((root / "plan.json").read_text())
        started = json.loads((root / "run.json").read_text())["started_unix"]
        attempts = rows(root / "attempts.jsonl")
        phase = self.factory.phase
        if time() - started >= plan["limits"]["wall_seconds"]:
            raise RuntimeError("bootstrap original wall-time budget exhausted")
        if sum(row["phase"] == phase for row in attempts) >= plan["limits"][phase + "_attempts"]:
            raise RuntimeError("bootstrap native attempt budget exhausted")
        append(root / "attempts.jsonl", {"phase": phase, "request_id": request.request_id,
                                        "controlled_sha256": sha256_bytes(request.preset_json.encode()), "started_unix": time()})
        return self.factory.base().render(request)


def run_preflight(pool, qualification_path, output, *, execute=False, resume=False, renderer_factory=None):
    plan = make_plan(pool, qualification_path)
    return run_frozen_plan(pool, plan, output, execute=execute, resume=resume, renderer_factory=renderer_factory)


def run_frozen_plan(pool, plan, output, *, execute=False, resume=False, renderer_factory=None):
    output = Path(output).resolve()
    if output.exists():
        if json.loads((output / "plan.json").read_text()) != plan:
            raise ValueError("frozen bootstrap plan changed")
    else:
        output.mkdir(parents=True)
        write_json(output / "plan.json", plan)
    if not execute:
        return {"plan_id": canonical_sha256(plan), "executed": False}
    if (output / "run.json").exists() != resume:
        raise ValueError("an existing run needs explicit resume; a new run cannot resume")
    if not resume:
        write_json(output / "run.json", {"started_unix": time(), "plan_id": canonical_sha256(plan)})
    started = json.loads((output / "run.json").read_text())["started_unix"]
    monitor = MemoryMonitor()
    monitor.start()

    def check(*_args):
        if time() - started >= plan["limits"]["wall_seconds"]:
            raise RuntimeError("bootstrap original wall-time budget exhausted")
        if (monitor.peak_bytes or 0) > plan["limits"]["process_tree_bytes"]:
            raise RuntimeError("bootstrap memory budget exhausted")

    def progress(event):
        check()
        if sum(path.stat().st_size for path in output.rglob("*") if path.is_file()) > plan["limits"]["archive_bytes"]:
            raise RuntimeError("bootstrap archive budget exhausted")
        append(output / "progress.jsonl", {"elapsed_seconds": time() - started, **event})
        print(json.dumps(event), flush=True)

    base = renderer_factory or NativeRendererFactory(renderer_id=pool.codec.contract.renderer_id)
    try:
        segments = sorted(output.glob("fit-*"))
        previous = segments[-1] if segments else None
        if previous and (previous / "summary.json").exists() and json.loads((previous / "summary.json").read_text())["complete"]:
            fit_root = previous
            fit = json.loads((fit_root / "summary.json").read_text())
        else:
            fit_root = output / f"fit-{len(segments):03d}"
            fit = train_pilot(pool, fit_root, config=TrainConfig(**plan["config"]),
                              stream_config=StreamConfig(prefetch_clips=1),
                              renderer_factory=AuditedFactory(str(output), "training", base),
                              resume=previous / "checkpoint.pt" if previous else None, after_step=check, progress=progress)
        saved = load_training_checkpoint(fit_root / "checkpoint.pt")
        if fit["renderer_identity"] != plan["renderer_identity"]:
            raise ValueError("bootstrap renderer differs from qualified native renderer")
        if ([item["recipe_id"] for item in saved["replay"]] != [row["recipe_id"] for row in plan["sources"]]
                or fit["totals"]["accepted"] != len(plan["sources"])):
            raise ValueError("bootstrap failed its frozen source coverage")
        evaluation_root = output / "training-audit"
        audit = evaluate_pilot(pool, fit_root / "checkpoint.pt", evaluation_root, count=8, split="train",
                               renderer_factory=AuditedFactory(str(output), "evaluation", base), progress=progress,
                               resume=evaluation_root.exists())
        receipts = [{"phase": "training", **row} for segment in sorted(output.glob("fit-*")) for row in rows(segment / "renders.jsonl")]
        receipts.extend({"phase": "evaluation", **row} for row in rows(evaluation_root / "renders.jsonl"))
        attempts = rows(output / "attempts.jsonl")
        receipt_ids = [(row["phase"], row.get("request_id", row.get("provenance", {}).get("request_id"))) for row in receipts]
        counts_match = Counter((row["phase"], row["request_id"]) for row in attempts) == Counter(receipt_ids)
        source_repeats = all(audit["aggregates"]["source_repeat"][role]["mean_mrstft"] == 0 for role in ("reference", "transfer"))
        ratios = {role: audit["aggregates"]["model"][role]["mean_mrstft"] /
                       max(audit["aggregates"]["fixed_template"][role]["mean_mrstft"], 1e-12) for role in ("reference", "transfer")}
        check()
        summary = {"plan_id": canonical_sha256(plan), "complete": True, "training": fit, "training_audit": audit,
                   "attempts": len(attempts), "receipts": len(receipts), "attempt_outcome_counts_match": counts_match,
                   "attempts_by_phase": dict(Counter(row["phase"] for row in attempts)),
                   "all_outcomes_accepted": counts_match and all(row["status"] == "accepted" for row in receipts),
                   "source_repeats_exact": source_repeats, "model_vs_template_ratio": ratios,
                   "model_state_sha256": tensor_digest(saved["model"]), "elapsed_seconds": time() - started,
                   "peak_process_tree_rss_bytes": monitor.peak_bytes, "rss_sampling_errors": monitor.errors,
                   "archive_bytes": sum(path.stat().st_size for path in output.rglob("*") if path.is_file()),
                   "qualified_for_release": False, "test_split_used": False}
        summary["within_resource_limits"] = (monitor.peak_bytes is not None and monitor.errors == 0
                                              and summary["archive_bytes"] < plan["limits"]["archive_bytes"])
        parameter_gate = plan["gates"].get("parameter_fit")
        parameter_fit = (fit["final_metrics"]["continuous_mae"] < parameter_gate["continuous_mae_below"]
                         and fit["final_metrics"]["categorical_accuracy"] == parameter_gate["categorical_accuracy"]
                         and fit["final_metrics"]["loss"] < fit["initial_metrics"]["loss"] * parameter_gate["final_initial_loss_ratio_below"]
                         if parameter_gate else fit["tiny_overfit_gate"])
        summary["parameter_fit_passed"] = parameter_fit
        summary["preflight_passed"] = (parameter_fit and summary["all_outcomes_accepted"] and source_repeats
                                       and summary["within_resource_limits"]
                                       and all(ratio < .9 for ratio in ratios.values()))
        write_json(output / "summary.json", summary)
        write_json(output / "status.json", {"complete": True, "preflight_passed": summary["preflight_passed"]})
        return summary
    except BaseException as error:
        write_json(output / "status.json", {"complete": False, "error_type": type(error).__name__, "elapsed_seconds": time() - started})
        raise
    finally:
        monitor.close()
