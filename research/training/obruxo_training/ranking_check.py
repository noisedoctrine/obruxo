"""Check the shipped audio-only ranker separately from evaluator-only best-of-four."""

import json
from pathlib import Path
import platform
from time import perf_counter

import numpy as np
import torch

from obruxo_data.hashing import canonical_sha256, sha256_bytes
from obruxo_data.render.base import RenderRequest

from .audit import write_json
from .audio_candidates import RANKING_SPEC, export_candidate_model, load_candidate_model, rank_audio
from .benchmark import MemoryMonitor
from .codec import PilotTarget
from .contracts import RenderRecipe
from .evaluate import _restore_evaluation, audio_distance, curve_statistics
from .performances import initial_bank
from .proxy_screen import balanced_entries, source_identity
from .splits import SplitLedger
from .stream import NativeRendererFactory, render_isolated


def check_audio_ranking(pool, checkpoint, output, *, count=10, renderer_factory=None, timeout=30., resume=False, progress=None):
    if type(count) is not int or count <= 0:
        raise ValueError("ranking validation needs a positive authored validation panel")
    torch.set_num_threads(4)
    output = Path(output)
    if not resume:
        output.mkdir(parents=True, exist_ok=False)
        identity = export_candidate_model(pool, checkpoint, output / "candidate-model.pt")
    else:
        artifact = torch.load(output / "candidate-model.pt", map_location="cpu", weights_only=True)
        identity = artifact["identity"]
    cold_started = perf_counter()
    model, frontend, codec, bank, renderer_identity = load_candidate_model(output / "candidate-model.pt", allow_unqualified=True)
    cold_load_seconds = perf_counter() - cold_started
    panel = balanced_entries(pool, "validation", count)
    ledger = SplitLedger(pool.manifest)
    recipes = [RenderRecipe.create(codec.decode(entry.target), initial_bank()[index % 4].performance(), schema=codec.schema,
                                  catalog=codec.catalog, contract=codec.contract, ledger=ledger, purpose="evaluate", sequence_id=index)
               for index, entry in enumerate(panel)]
    plan = {"evaluation": "audio_only_candidate_ranking_v1", "pool_id": pool.pool_id, "identity": identity,
            "runtime": {"python": platform.python_version(), "torch": str(torch.__version__), "numpy": np.__version__,
                        "device": "cpu", "precision": "float32", "cpu_threads": 4},
            "source_checkpoint_sha256": sha256_bytes(Path(checkpoint).read_bytes()), "source": source_identity(),
            "recipe_ids": [recipe.recipe_id for recipe in recipes], "validation_states": [entry.state_id for entry in panel],
            "maximum_render_attempts": count * 16, "maximum_attempts_with_recovery": count * 16 + 16,
            "qualification": {"mean_deployed_vs_model_ratio_max": .95, "ranked_coverage_min": .8,
                              "native_rejection_fraction_max": .01, "warm_ranking_p95_seconds_max": 20., "peak_rss_bytes_max": 8 * 2**30},
            "primary": "deployed_audio_only_choice_reference_transfer_mean", "oracle": "diagnostic_only_never_selects_shipped_patch",
            "timeout": timeout, "test_split_evaluated": False}
    rows, renders, completed, prior_elapsed = [], [], [], 0.
    if resume:
        state, rows, renders = _restore_evaluation(output, plan, renderer_identity)
        completed, prior_elapsed = state["completed_cases"], state["elapsed_seconds"]
    else:
        write_json(output / "plan.json", plan)
        (output / "renders.jsonl").touch()
    started, monitor = perf_counter(), MemoryMonitor()
    factory = renderer_factory or NativeRendererFactory()

    def save(row=None):
        if row is not None:
            path = output / f"case-{row['case']:03d}.json"
            write_json(path, row)
            completed.append({"case": row["case"], "case_sha256": sha256_bytes(path.read_bytes()), "predictions": {}})
        write_json(output / "progress.json", {"plan_sha256": canonical_sha256(plan), "completed_cases": completed,
                                              "committed_render_count": len(renders), "renders_sha256": canonical_sha256(renders),
                                              "elapsed_seconds": prior_elapsed + perf_counter() - started})

    def budget():
        if len(renders) >= plan["maximum_attempts_with_recovery"]:
            raise RuntimeError("frozen audio-ranking native-render budget exhausted")

    def record(event, case, role):
        row = {**event, "case": case, "role": role}
        renders.append(row)
        with (output / "renders.jsonl").open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(row, allow_nan=False) + "\n")
            handle.flush()
        if row["status"] != "error":
            identity = {key: value for key, value in row["provenance"].items() if key != "request_id"}
            if identity != renderer_identity:
                raise ValueError("ranking check renderer changed")

    def render(target, performance, index, method, role):
        budget()
        request = RenderRequest(preset_json=codec.decode(target).decode(), performance=performance, sample_rate=44100, channels=2,
                                tail_seconds=codec.contract.tail_seconds, renderer_id=codec.contract.renderer_id)
        audio, metadata = render_isolated(request, renderer_factory=factory, timeout=timeout)
        record({"method": method, "request_id": request.request_id, **metadata}, index, role)
        return audio

    save()
    monitor.start()
    try:
        for index in range(len(rows), len(panel)):
            entry = panel[index]
            performances = {"reference": initial_bank()[index % 4].performance(), "transfer": initial_bank()[4 + index % 4].performance()}
            sources = {role: render(entry.target, played, index, "source", role) for role, played in performances.items()}
            row = {"case": index, "state_id": entry.state_id, "methods": {}, "ranked": False}
            if sources["reference"] is not None:
                # Reserve the full inference allowance before entering the public audio-only interface.
                if len(renders) + RANKING_SPEC["maximum_candidates"] > plan["maximum_attempts_with_recovery"]:
                    raise RuntimeError("insufficient remaining audio-ranking render budget")
                _, result = rank_audio(model, frontend, codec, bank, sources["reference"], renderer_factory=factory, timeout=timeout,
                                       expected_renderer=renderer_identity, progress=lambda event: record(event, index, "inferred_performance"))
                row.update(ranked=result["ranked"], deployed=result["selected"], ranking=result)
                scores = {}
                for name, target in result["candidates"].items():
                    scores[name] = {}
                    for role, played in performances.items():
                        if sources[role] is None:
                            continue
                        audio = render(PilotTarget.from_dict(target), played, index, name, role)
                        if audio is not None:
                            scores[name][role] = audio_distance(sources[role], audio)
                row["candidate_scores_under_evaluator_notes"] = scores
                row["methods"].update(model=scores.get("model", {}), deployed=scores.get(result["selected"], {}),
                                       train_retrieval=scores.get("retrieval", scores.get("model", {})))
                eligible = [name for name, values in scores.items() if all(role in values for role in performances)]
                if eligible:
                    best = min(eligible, key=lambda name: np.mean([scores[name][role]["mrstft"] for role in performances]))
                    row["oracle_choice"] = best
                    row["methods"]["oracle_best_four"] = scores[best]
                repeats = {}
                for role, played in performances.items():
                    if sources[role] is None:
                        continue
                    repeated = render(entry.target, played, index, "source_repeat", role)
                    if repeated is not None:
                        repeats[role] = audio_distance(sources[role], repeated)
                row["methods"]["source_repeat"] = repeats
            else:
                row["failure"] = "source_reference_rejected"
            rows.append(row)
            save(row)
            if progress:
                progress({"stage": "audio_ranking_check", "completed": index + 1, "planned": count, "ranked": row["ranked"]})
        statistics = curve_statistics(rows, ["deployed"], baseline_name="model")
        means = {}
        for method in ("model", "deployed", "train_retrieval", "oracle_best_four", "source_repeat"):
            values = [np.mean([row["methods"][method][role]["mrstft"] for role in ("reference", "transfer")]) for row in rows
                      if all("mrstft" in row["methods"].get(method, {}).get(role, {}) for role in ("reference", "transfer"))]
            means[method] = {"planned": count, "scored": len(values), "mean_mrstft": float(np.mean(values)) if values else None}
        coverage = sum(row["ranked"] for row in rows) / count
        latencies = [row["ranking"]["elapsed_seconds"] for row in rows if "ranking" in row]
        accepted = sum(event["status"] == "accepted" for event in renders)
        gates = plan["qualification"]
        qualified = (means["model"]["scored"] == means["deployed"]["scored"] == count and means["model"]["mean_mrstft"] > 0
                     and means["deployed"]["mean_mrstft"] / means["model"]["mean_mrstft"] <= gates["mean_deployed_vs_model_ratio_max"]
                     and coverage >= gates["ranked_coverage_min"] and len(latencies) == count
                     and 1 - accepted / len(renders) <= gates["native_rejection_fraction_max"]
                     and float(np.quantile(latencies, .95)) <= gates["warm_ranking_p95_seconds_max"]
                     and monitor.peak_bytes is not None and monitor.peak_bytes <= gates["peak_rss_bytes_max"])
        qualification = {"qualified": bool(qualified), "identity": identity, "plan_id": canonical_sha256(plan),
                         "gates": gates, "means": means, "ranked_coverage": coverage,
                         "warm_p95_seconds": float(np.quantile(latencies, .95)) if latencies else None}
        if qualified:
            export_candidate_model(pool, checkpoint, output / "candidate-model.pt", qualification=qualification)
        summary = {"qualification": qualification, "paired_deployed_vs_model": statistics, "renders_attempted": len(renders),
                   "renders_accepted": accepted, "cold_bundle_load_seconds": cold_load_seconds,
                   "elapsed_seconds": prior_elapsed + perf_counter() - started, "peak_process_tree_rss_bytes": monitor.peak_bytes,
                   "audio_files_written": 0, "final_test_used": False, "release_model_promoted": False}
        write_json(output / "summary.json", summary)
        return summary
    finally:
        monitor.close()
