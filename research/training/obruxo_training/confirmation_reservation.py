"""Reserve independent basic confirmation sources without rendering or scoring."""

from collections import defaultdict
import hashlib
import itertools
import json
import math
from statistics import mean, stdev

from obruxo_data.hashing import canonical_sha256
from obruxo_data.midi import Performance, TempoMap

from .audit import write_json
from .codec import PilotCodec
from .performances import CAPABILITIES, initial_bank
from .pilot import PilotPool

RULE = "basic_confirmation_reservation_v1"
PER_STRATUM, BANK_SIZE, EXPANDED_COUNT = 32, 8, 4096
ROLES = ("reference", "transfer_held", "transfer_pattern")
LIMITS = {"native_requests": 3840, "planned_native_requests": 3648, "optimizer_updates": 0, "wall_seconds": 14400,
          "process_tree_bytes": 8 * 1024**3, "archive_bytes": 8 * 1024**3}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_bytes())


def development_precision(directory):
    plan, verified = read(directory / "plan.json"), read(directory / "verification.json")
    if not verified["verified"] or verified["plan_id"] != canonical_sha256(plan):
        raise ValueError("precision planning requires an independently verified development fit")
    for name in ("plan.json", "finals.jsonl"):
        if digest(directory / name) != verified["evidence_sha256"].get(name):
            raise ValueError("development evidence changed after verification")
    scores = {}
    for line in (directory / "finals.jsonl").read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        key, value = (row["index"], row["arm"]), row["scores"]["mrstft"]
        if key in scores or not math.isfinite(value) or value < 0:
            raise ValueError("development comparison has duplicate or invalid scores")
        scores[key] = value
    result = {}
    for role in ("reference", "transfer"):
        families = defaultdict(list)
        for index, cell in enumerate(plan["fit_audit_cells"]):
            if cell["input_view"] != 0 or (cell["playback_view"] == 0) != (role == "reference"):
                continue
            families[cell["recipe"]["state_id"]].append((scores[(index, "spectral")], scores[(index, "parameter")]))
        pairs = [(mean(row[0] for row in rows), mean(row[1] for row in rows)) for rows in families.values()]
        if len(pairs) < 2:
            raise ValueError("precision planning needs at least two independent paired families")
        baseline = mean(row[1] for row in pairs)
        if baseline <= 0:
            raise ValueError("precision planning requires positive development control error")
        relative_sd = stdev(row[0] - row[1] for row in pairs) / baseline
        result[role] = {"independent_families": len(pairs), "relative_paired_sd": relative_sd,
                        "estimated_families_for_10pct_ci_half_width": math.ceil((1.96 * relative_sd / .10)**2),
                        "estimated_95ci_half_width_at_48": 1.96 * relative_sd / math.sqrt(48)}
    return {"source_plan_id": verified["plan_id"], "verification_sha256": digest(directory / "verification.json"),
            "method": "normal_95_interval_half_width_divided_by_development_control_mean",
            "limitation": "rough_development_variability_estimate_not_guaranteed_fresh_panel_precision_or_power", "roles": result}


def confirmation_bank():
    bank = []
    for index, role in itertools.product(range(BANK_SIZE), ROLES):
        role_index = ROLES.index(role)
        bpm = 101 + index * 2 + role_index * 26
        performance = Performance(bpm=bpm)
        def ticks(seconds):
            return round(seconds * bpm * 8)
        if role == "reference":
            base, velocity = 50 + index * 2, (72, 88, 104, 112)[index % 4]
            offsets = (0,) if index % 2 == 0 else (0, 3, 7)
            notes = [(base + offset, velocity, 0, 2.05) for offset in offsets]
        elif role == "transfer_held":
            base = 36 + index if index % 2 == 0 else 77 + index
            notes = [(base, (64, 80, 96, 112)[index % 4], 0, 2.15)]
        else:
            notes = [(pitch, velocity, start, duration) for pitch, velocity, start, duration in
                     zip((48 + index, 67 - index, 55 + index, 72 - index), (72, 96, 112, 64), (0, .35, .8, 1.35), (.24, .42, .31, .48), strict=True)]
        for pitch, velocity, start, duration in notes:
            performance.add_note(pitch=pitch, velocity=velocity, start_tick=ticks(start), duration_ticks=ticks(duration))
        performance.end_tick += 17
        performance.validate(CAPABILITIES).require_valid()
        frames = TempoMap.from_performance(performance).render_frame_count(performance.end_tick, 2., 44100)
        if frames > 5 * 44100:
            raise ValueError("confirmation performance exceeds its five-second storage bound")
        bank.append({"bank_index": index, "role": role, "performance_id": canonical_sha256(performance.to_dict()),
                     "performance": performance.to_dict(), "render_frames": frames})
    identities = {row["performance_id"] for row in bank}
    if len(identities) != BANK_SIZE * len(ROLES) or identities & {row.performance_id for row in initial_bank()}:
        raise ValueError("confirmation performances overlap each other or the original bank")
    return bank


def select_families(parent, expanded):
    proof = expanded.expansion_from(parent)
    if type(parent.codec) is not PilotCodec:
        raise ValueError("confirmation reservation supports only the 14-output basic codec")
    previous = {entry.state_id for entry in parent.entries}
    groups = defaultdict(list)
    for entry in expanded.entries:
        if entry.state_id not in previous and expanded.manifest.by_state[entry.state_id].split == "validation":
            groups[entry.target.categorical].append(entry.state_id)
    cases = []
    for category in itertools.product(range(3), range(2)):
        states = sorted(groups[category], key=lambda state: canonical_sha256({"rule": RULE, "state_id": state}))
        if len(states) < PER_STRATUM:
            raise ValueError("the predeclared expansion has insufficient independent validation families")
        for index, state in enumerate(states[:PER_STRATUM]):
            assignment = expanded.manifest.by_state[state]
            cases.append({"state_id": state, "lineage_id": assignment.lineage_id, "split": assignment.split,
                          "category": list(category), "bank_index": index % BANK_SIZE})
    return sorted(cases, key=lambda row: canonical_sha256({"rule": RULE, "case": row})), proof


def reserve(parent_path, development, pending_fit, features, output, *, implementation_commit):
    if output.exists():
        raise FileExistsError(output)
    parent = PilotPool.load(parent_path)
    if len(parent.entries) != 512 or type(parent.codec) is not PilotCodec:
        raise ValueError("reservation requires the existing 512-state basic population")
    precision = development_precision(development)
    if max(row["estimated_families_for_10pct_ci_half_width"] for row in precision["roles"].values()) > 192:
        raise ValueError("the fixed panel is too small for the declared development precision estimate")
    fit_plan, old_plan = read(pending_fit / "plan.json"), read(development / "plan.json")
    if (not fit_plan.get("require_trainable_input_updates") or fit_plan["pool_id"] != parent.pool_id or old_plan["pool_id"] != parent.pool_id
            or digest(features) != old_plan["input_cache_sha256"] or digest(features) != fit_plan["input_cache_sha256"]):
        raise ValueError("pending fit or training-only retrieval feature identity differs")
    bank = confirmation_bank()
    used = {canonical_sha256(value) for cell in old_plan["fit_audit_cells"] for value in cell["recipe"]["performances"]}
    if used & {row["performance_id"] for row in bank}:
        raise ValueError("confirmation bank overlaps a development-fit performance")
    expanded = parent.expand(EXPANDED_COUNT)
    cases, proof = select_families(parent, expanded)
    audio_bound = LIMITS["planned_native_requests"] * 5 * 44100 * 2 * 4
    if audio_bound + 512 * 1024**2 > LIMITS["archive_bytes"]:
        raise ValueError("confirmation audio and overhead exceed the declared archive limit")
    previous = development / "diagnostic-spectral.pt"
    verified = read(development / "verification.json")
    if digest(previous) != verified["evidence_sha256"].get(previous.name):
        raise ValueError("the previous-model artifact differs from its independent audit")
    reservation = {"format": RULE, "implementation_commit": implementation_commit, "purpose": "reserve_only_no_audio_or_candidate_scoring",
                   "qualified_for_release": False, "source_pool_file_sha256": digest(parent_path), "expansion": proof,
                   "cases": cases, "performances": bank, "precision_planning": precision,
                   "pending_fit_plan_id": canonical_sha256(fit_plan), "pending_fit_plan_sha256": digest(pending_fit / "plan.json"),
                   "previous_model_file_sha256": digest(previous), "training_only_feature_cache_sha256": digest(features),
                   "inference_policy": "one_prediction_from_reference_audio_only_then_reuse_under_all_three_performances",
                   "methods": ["candidate", "matched_parameter_control", "previous_model", "fixed_template", "training_retrieval"],
                   "requests_per_family": {"three_sources": 3, "five_methods_three_performances": 15, "exact_reference_repeat": 1},
                   "rendering": {"sample_rate": 44100, "channels": 2, "dtype": "float32", "preroll_frames": 32768, "tail_seconds": 2.,
                                 "phase": "fixed_oscillator_1_2_3_and_sampler_random_phase"},
                   "limits": LIMITS, "audio_bytes_upper_bound": audio_bound, "archive_overhead_reserve_bytes": 512 * 1024**2,
                   "before_evaluation": ["independently_verified_A2_fit_and_unchanged_quality_preservation_gates",
                                         "freeze_candidate_control_weights_complete_runtime_and_scoring_plan",
                                         "qualify_training_only_retrieval_and_complete_no_render_recovery_preflight"],
                   "new_native_requests": 0, "optimizer_applications": 0, "test_states_evaluated": 0}
    reservation["reservation_id"] = canonical_sha256(reservation)
    output.mkdir(parents=True)
    expanded.save(output / "pool.json")
    write_json(output / "reservation.json", reservation)
    return {"reservation_id": reservation["reservation_id"], "families": len(cases), "performance_bank_entries": len(bank),
            "parent_assignments_preserved": len(parent.entries), "expanded_population": expanded.summary(),
            "new_native_requests": 0, "optimizer_applications": 0, "candidate_scored": False, "qualified_for_release": False}
