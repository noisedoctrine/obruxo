"""Bind direct sound fitting to the qualified initializer and original recordings."""

from math import prod
from pathlib import Path

import torch

from obruxo_data.hashing import canonical_sha256, sha256_bytes
from obruxo_data.midi import Performance, TempoMap

from .basic_direct import COUNTS, REHEARSAL, SPEC, rehearsal_indices, restore_direct
from .basic_direct_bridge import reference_manifest
from .conditioned_bridge import read
from .conditioned_short_audit import cached_features, feature_bank
from .features import PITCH_SIZE, TIMBRE_SHAPE, tensor_digest
from .model import batch_features
from .native_fit import digest
from .native_fit_storage import target_tensor_bytes
from .paired_native_step import reciprocal_updates
from .production_controls import ProductionControlAdapter
from .unison_fit_data import Inputs as PriorInputs

LIMITS = {"wall_seconds": 14400, "process_tree_bytes": 4 * 1024**3, "archive_bytes": 2 * 1024**3,
          "replay_bytes": 512 * 1024**2, "optimizer_attempts": 49, "native_executions": 2,
          "training_attempts": 6480, "qualification_attempts": 240}


def storage_bound(cells, tail_seconds):
    frames, sources = [], {}
    for cell in cells:
        recipe, view = cell["recipe"], cell["playback_view"]
        performance = Performance.from_dict(recipe["performances"][view])
        count = TempoMap.from_performance(performance).render_frame_count(performance.end_tick, tail_seconds, 44100)
        frames.append(count)
        sources[recipe["source_audio_sha256"][view]] = count
    # Targets and source audio are LRU-bounded to four recordings, never the whole panel.
    largest = sorted(sources.values(), reverse=True)[:4]
    features = len(sources) * 4 * (prod(TIMBRE_SHAPE) + PITCH_SIZE)
    replay = features + sum(target_tensor_bytes(n) + 8 * n for n in largest)
    values = {"unique_source_audio_bytes": 8 * sum(sources.values()),
              "initial_and_two_final_comparison_audio_bytes": 3 * 8 * sum(frames),
              "waveform_serialization_allowance_bytes": (len(sources) + 3 * len(frames)) * 16384,
              "resident_feature_and_target_allowance_bytes": replay,
              "checkpoint_export_prefix_interruption_reserve_bytes": 96 * 1024**2,
              "native_probe_ledger_and_publication_reserve_bytes": 192 * 1024**2,
              "transient_spectral_and_audio_allowance_bytes": 4 * (target_tensor_bytes(max(frames)) + 8 * max(frames)),
              "archive_guard_bytes": 32 * 1024**2}
    total = sum(values.values())
    if replay > LIMITS["replay_bytes"] or total > LIMITS["archive_bytes"]:
        raise ValueError("direct fit source replay or evidence exceeds its declared allocation")
    return {**values, "estimated_archive_upper_bound_bytes": total, "resident_replay_upper_bound_bytes": replay,
            "unique_sources": len(sources), "retained_candidate_waveforms_maximum": 3 * len(frames),
            "training_probe_audio_retained": False, "target_cache_records": 4}


def make_plan(parent, bridge, recent, cache, pool):
    proof, qualified, bridge_plan = read(parent / "verification.json"), read(bridge / "summary.json"), read(bridge / "plan.json")
    prior = proof["plan"]
    if (not proof["verified"] or not proof["offline_training_replay_exact"] or not proof["complete"] or not proof["all_native_accepted"]
            or proof["committed_updates_per_arm"] != 48 or proof["comparison_waveforms_rescored"] != 312
            or not qualified["initialization_gradient_reference_recovery_passed"] or not qualified["parent_artifacts_unchanged"]
            or qualified["new_native_requests"] != 0 or qualified["experimental_training_updates"] != 0 or qualified["qualified_for_release"]
            or qualified["plan_id"] != canonical_sha256(bridge_plan) or qualified["model_budget"] != COUNTS
            or digest(bridge / "model.pt") != qualified["model_file_sha256"] or digest(cache) != prior["input_cache_sha256"]
            or any(bridge_plan[key] != prior[key] for key in ("expected_renderer_identity", "render_profile", "render_phase_controls"))
            or pool.pool_id != prior["pool_id"] or prior["validation"] or prior["test_split_used"]):
        raise ValueError("direct fit requires the qualified direct bridge and unchanged verified source recordings")
    files = {f"parent/{name}": digest(parent / name) for name in proof["evidence_sha256"]}
    if {name: files[f"parent/{name}"] for name in proof["evidence_sha256"]} != proof["evidence_sha256"]:
        raise ValueError("direct fit parent evidence changed")
    references, reference_files = reference_manifest(recent, parent)
    if references != bridge_plan["reference_manifest"] or reference_files != bridge_plan["reference_files"]:
        raise ValueError("qualified reference rehearsal manifest changed")
    files.update({f"bridge/{name}": digest(bridge / name) for name in ("plan.json", "summary.json", "model.pt")})
    files.update({f"reference/{name}": value for name, value in reference_files.items()})
    cases = read(parent / "fit_cases.json")
    if len(cases) != 104 or len(prior["training"]) != 12:
        raise ValueError("direct fit requires the fixed twelve-patch / 104-case panel")
    for case in cases:
        for name in (f"audits/{case['label']}.pt", case["rows"]["initial"]["receipt"]["retained_audio"]["file"]):
            files[f"parent/{name}"] = digest(parent / name)
    files["parent/fit_cases.json"] = digest(parent / "fit_cases.json")
    files["parent/verification.json"] = digest(parent / "verification.json")
    root = Path(__file__).parents[1]
    for name, value in bridge_plan["source"].items():
        path = root / name if "/" in name else Path(__file__).with_name(name)
        if sha256_bytes(path.read_bytes().replace(b"\r\n", b"\n")) != value:
            raise ValueError("qualified direct implementation changed")
    names = ("basic_direct_fit_data.py", "basic_direct_fit_step.py", "basic_direct_fit.py", "basic_direct_fit_audit.py",
             "unison_fit_data.py", "unison_sound_fit.py", "paired_native_fit.py", "paired_native_evidence.py",
             "conditioned_short_audit.py", "native_fit_storage.py", "local_execution.py", "benchmark.py", "checkpoints.py")
    code = {name: sha256_bytes(Path(__file__).with_name(name).read_bytes().replace(b"\r\n", b"\n")) for name in names}
    script = root / "reports/scripts/run_basic_direct_fit.py"
    code["reports/scripts/run_basic_direct_fit.py"] = sha256_bytes(script.read_bytes().replace(b"\r\n", b"\n"))
    train = {state for state, assignment in pool.manifest.by_state.items() if assignment.split == "train"}
    if not set(prior["regression_state_ids"]).issubset(train) or any(row["state_id"] not in train for row in prior["training"]):
        raise ValueError("direct fit cannot use held-out states")
    updates = reciprocal_updates(12, seed=0) + reciprocal_updates(12, seed=1)
    updates = [{**row, "references": list(rehearsal_indices(i))} for i, row in enumerate(updates)]
    return {"format": "basic_direct_sound_rehearsal_fit_v1", "input_files": files, "input_cache_sha256": digest(cache),
            "pool_id": pool.pool_id, "bridge_plan_id": qualified["plan_id"], "initial_file_sha256": qualified["model_file_sha256"],
            "initial_model_sha256": qualified["model_sha256"], "source": {**bridge_plan["source"], **code},
            "training": prior["training"], "fit_audit_cells": prior["fit_audit_cells"], "reference_manifest": references,
            "regression_state_ids": prior["regression_state_ids"], "training_view_roles": prior["training_view_roles"],
            "render_profile": prior["render_profile"], "render_phase_controls": prior["render_phase_controls"],
            "expected_renderer_identity": prior["expected_renderer_identity"], "model_budget": COUNTS, "model_spec": SPEC,
            "updates": updates, "epoch_shuffle_seeds": [0, 1], "rehearsal": REHEARSAL,
            "optimizer": {"name": "Adam", "learning_rate": .001, "seed": 0, "gradient_norm_cap": 5., "updates_per_arm": 48},
            "loss": "weighted_paired_input_transfer_and_reference_excess_sound_v1", "parameter_control": "weighted_masked_basic_labels_raw_corrected_scores",
            "temperature": {"start": 1., "end": .5}, "limits": LIMITS, "prefix_check_updates": 4,
            "planned_native_requests": {"paired_gradients": 4224, "reference_gradients": 2112, "qualification": 208, "total": 6544},
            "storage_bound": storage_bound(prior["fit_audit_cells"], pool.codec.contract.tail_seconds),
            "gates": {"six_groups_sound_ratio": .9, "six_groups_at_most_parameter": True, "six_groups_mrstft_strict_improvement": True,
                      "original_32_mean_nonregression": True, "original_32_mrstft_p90_ratio": 1.1},
            "runtime": {"threads": 2, "device": "cpu", "precision": "float32", "native_workers": 1},
            "development_scoring_allowed": False, "test_split_used": False, "model_admission_allowed": False}


class Inputs(PriorInputs):
    def __init__(self, parent, bridge, cache, plan):
        self.parent, self.plan, self.initial_path = parent, plan, bridge / "model.pt"
        self.model, self.frontend, self.codec = restore_direct(torch.load(self.initial_path, map_location="cpu", weights_only=True))
        if tensor_digest(self.model.state_dict()) != plan["initial_model_sha256"]:
            raise ValueError("direct fitting initializer tensors changed")
        self.adapter = ProductionControlAdapter(self.codec, scales=SPEC["adapter_scales"])
        self.cases = read(parent / "fit_cases.json")
        bank = feature_bank(cache)
        self.features, self.baselines = {}, {}
        for cell in plan["fit_audit_cells"]:
            key = self.key(cell)
            if key not in self.features:
                self.features[key] = cached_features(bank, cell["recipe"], cell["input_view"])
                with torch.no_grad():
                    self.baselines[key] = self.model(batch_features([{"features": self.features[key]}]))
        self.bytes = sum(value.numel() * value.element_size() for features in self.features.values() for value in features.values())
        if self.bytes > plan["limits"]["replay_bytes"]:
            raise ValueError("direct fit feature replay allocation exceeded")
        self.lookup = {(cell["recipe"]["state_id"], cell["input_view"], cell["playback_view"]): i for i, cell in enumerate(plan["fit_audit_cells"])}
        self.references = [self.lookup[(row["state_id"], 0, 0)] for row in plan["reference_manifest"]]
        if len(set(self.references)) != 32:
            raise ValueError("direct fit must rehearse the intact original panel")
        for reference, index in zip(plan["reference_manifest"], self.references, strict=True):
            row = self.cases[index]["rows"]["initial"]
            if reference["initial_patch"] != row["candidate"] or reference["initial_scores"] != row["scores"]:
                raise ValueError("reference baseline differs from the accepted initial prediction")

    def predict(self, model, cell):
        return model(batch_features([{"features": self.features[self.key(cell)]}]))

    def observations(self, update):
        rows = []
        for example in update["examples"]:
            state = self.plan["training"][example["state"]]["state_id"]
            rows.append([self.lookup[(state, example["input_view"], view)] for view in (example["input_view"], example["view"])])
        return rows + [[self.references[index]] for index in update["references"]]
