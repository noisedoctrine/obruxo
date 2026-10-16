"""Publish phase-controlled worker correctness and paired timing evidence."""

import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import mean
import sys

RESEARCH = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(RESEARCH / "data_generation")]
from obruxo_data.hashing import canonical_sha256  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--phase-probe", type=Path, required=True)
    parser.add_argument("--tests-passed", type=int, required=True)
    args = parser.parse_args()
    plan = json.loads((args.run / "plan.json").read_text())
    summary = json.loads((args.run / "summary.json").read_text())
    probe = json.loads((args.phase_probe / "summary.json").read_text())
    events = [json.loads(line) for line in (args.run / "renders.jsonl").read_text().splitlines()]
    if (summary["plan_sha256"] != canonical_sha256(plan) or plan["phase_control"] != {f"osc_{i}_random_phase": 0 for i in range(1, 4)}
            or plan["timing_design"] != "adjacent_fresh_reuse_pairs_with_alternating_order"
            or len(events) != 66 or any(row["status"] != "accepted" for row in events)
            or not probe["all_audio_identical"] or not probe["all_native_states_identical"]
            or len(summary["checks"]) != 33 or not all(row["audio_float32_identical"] and row["native_state_identical"]
                                                      and row["provenance_identical"] for row in summary["checks"])):
        raise ValueError("worker evidence is incomplete or not phase-controlled")
    for name, expected in plan["source"].items():
        if hashlib.sha256((RESEARCH / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"worker qualification source changed: {name}")
    fresh = [row for row in events if row["mode"] == "fresh"]
    reused = [row for row in events if row["mode"] == "reuse"]
    if (not math.isclose(mean(row["wall_seconds"] for row in fresh), summary["matched_fresh_mean_seconds"], abs_tol=1e-12)
            or not math.isclose(mean(row["wall_seconds"] for row in reused), summary["reuse_mean_seconds"], abs_tol=1e-12)
            or [row["worker_job"] for row in reused] != [*range(1, 33), 1]):
        raise ValueError("worker timings or process lifetime do not match native outcomes")
    summary.update(local_tests_passed=args.tests_passed, github_actions="disabled",
                   phase_probe=probe, raw_evidence_sha256=hashlib.sha256((args.run / "renders.jsonl").read_bytes()).hexdigest())
    reports = Path(__file__).resolve().parents[1]
    for name, value in (("render_worker_plan.json", plan), ("render_worker_summary.json", summary)):
        (reports / "data" / name).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    decision = "passed all local correctness, timing and memory gates" if summary["qualified"] else "passed correctness but did not pass every performance gate"
    rows = ["# Reusable render-worker qualification", "",
            f"The reusable worker **{decision}**. With oscillator phases fixed, all 33 reused-worker outputs matched fresh-process float32 audio, native state and provenance exactly, including the process restart after 32 jobs. All 66 native attempts passed QA.", "",
            f"Mean whole-call time was **{summary['matched_fresh_mean_seconds']:.3f}s fresh** versus **{summary['reuse_mean_seconds']:.3f}s reused**, a **{100 * summary['mean_wall_reduction']:+.1f}% reduction**. The paired check alternated fresh-first and reuse-first order for adjacent renders of the same patch. It includes cold starts and recycling, rather than comparing one early baseline with a long later batch. Sampled peak process-tree RSS was {summary['peak_process_tree_rss_bytes'] / 2**20:.1f} MiB.", "",
            "## Correctness controls", "",
            "An initial uncontrolled check mixed pipeline reproducibility with oscillator phase randomization. That was an invalid correctness test: identical settings may legitimately generate different waveforms when random phases change. Its speed numbers are not the final comparison. The corrected gate disables oscillator randomization in diagnostic copies, preserves fixed phase knobs and the actual training pool, and requires exact audio equality on this pinned local runtime.", "",
            f"A separate causal probe accepted {probe['accepted']}/{probe['native_attempts']} native attempts. Turning off oscillator phase randomization alone produced identical audio and state for all ten reused renders across five families. The full lifetime check then reproduced this over 32 jobs and a restart. Random-phase variation remains a separate training/sampling concern; no tolerance was loosened to make the reproducibility gate pass.", "",
            "The pilot and alpha codec correctness checks now use explicit phase-controlled diagnostic variants as well. They compare codec controls and nested state before applying phase controls, register the derived states under training lineage, and save a distinct check identity. Ordinary training patches and audio-quality evaluation requests retain their requested phase settings.", "",
            f"The local renderer/training suite passed **{args.tests_passed} tests**. Dedicated tests cover fresh renderer instances, bounded process lifetimes, QA rejection, crashes, deadlines, cancellation, shared-memory cleanup, queue limits, cursor compatibility and exact optimizer/RNG checkpoint recovery with reuse. Existing one-job contracts and exports remain compatible. GitHub Actions stays disabled.", "",
            "## Scope", "",
            "This qualifies the pinned Windows Vital/DawDreamer path on five authored training families with a 32-job lifetime. It does not qualify Linux/Colab, unrestricted full-schema presets, random-phase bit reproducibility or finished-model quality. A sampled RSS maximum over this bounded run is not an unlimited-duration leak test. The previous one-job default remains unchanged; new runs opt in and freeze their worker policy.", "",
            "[Usage](../RENDER_WORKER.md) · [Frozen plan](data/render_worker_plan.json) · [Aggregate evidence](data/render_worker_summary.json)", ""]
    (reports / "RENDER_WORKER_RESULTS.md").write_text("\n".join(rows))
    print(json.dumps({key: summary[key] for key in ("qualified", "gates", "mean_wall_reduction", "accepted")}, indent=2))


if __name__ == "__main__":
    main()
