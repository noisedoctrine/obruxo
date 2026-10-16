"""Publish the bounded check of the deployed audio-only ranker and its separate oracle ceiling."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

RESEARCH = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(RESEARCH / "data_generation")]
from obruxo_data.hashing import canonical_sha256  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--training-tests-passed", type=int, required=True)
    parser.add_argument("--renderer-tests-passed", type=int, required=True)
    args = parser.parse_args()
    plan = json.loads((args.run / "plan.json").read_text())
    summary = json.loads((args.run / "summary.json").read_text())
    progress = json.loads((args.run / "progress.json").read_text())
    events = [json.loads(line) for line in (args.run / "renders.jsonl").read_text().splitlines()]
    cases = [json.loads((args.run / f"case-{index:03d}.json").read_text()) for index in range(len(plan["validation_states"]))]
    qualification = summary["qualification"]
    if (qualification["plan_id"] != canonical_sha256(plan) or progress["plan_sha256"] != canonical_sha256(plan)
            or [row["state_id"] for row in cases] != plan["validation_states"] or plan["test_split_evaluated"]
            or summary["renders_attempted"] != len(events) or progress["renders_sha256"] != canonical_sha256(events)
            or len(events) > plan["maximum_attempts_with_recovery"]):
        raise ValueError("ranking evidence or frozen validation panel changed")
    for entry in progress["completed_cases"]:
        if hashlib.sha256((args.run / f"case-{entry['case']:03d}.json").read_bytes()).hexdigest() != entry["case_sha256"]:
            raise ValueError("completed ranking case changed")
    for name, expected in plan["source"].items():
        if hashlib.sha256((RESEARCH / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"ranking source changed: {name}")
    for row in cases:
        report = row.get("ranking")
        if report is not None and (report["input"] != "audio_only" or report["source_midi_supplied"]
                                   or row["deployed"] != report["selected"]):
            raise ValueError("deployed choice differs from the actual audio-only ranker")
    means = qualification["means"]
    change = 100 * (means["deployed"]["mean_mrstft"] / means["model"]["mean_mrstft"] - 1)
    summary.update(deployed_relative_error_change_percent=change,
                   deployed_choices=dict(Counter(row.get("deployed", "source_failed") for row in cases)),
                   artifact_sha256=hashlib.sha256((args.run / "candidate-model.pt").read_bytes()).hexdigest(),
                   verification={"training_suite_passed": args.training_tests_passed, "renderer_stream_suite_passed": args.renderer_tests_passed,
                                 "github_actions": "disabled"},
                   compute_note="Local CPU; independent regression suites and cached conditioning training overlapped this check. Latency is not an idle-machine benchmark.")
    reports = Path(__file__).resolve().parents[1]
    for name, value in (("audio_ranking_plan.json", plan), ("audio_ranking_summary.json", summary)):
        (reports / "data" / name).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    paired = summary["paired_deployed_vs_model"]["checkpoints"]["deployed"]["against_model"]
    rows = ["# Audio-only candidate ranking", "",
            f"Selecting a patch using only the input recording changed mean native reference/transfer error by **{change:+.2f}%** against ordinary prediction in this fixed ten-patch alpha check. The saved candidate bundle is **{'qualified for this bounded optional path' if qualification['qualified'] else 'unqualified and refused by ordinary loading'}**. This does not promote a finished release model.", "",
            "## What was measured", "",
            "The public ranker inferred notes from the input audio, proposed at most four legal patches and rendered them under the same inferred performance. Ground-truth MIDI and source settings never entered selection. Evaluation then rendered the chosen patch under known reference and reserved transfer performances. Evaluator-only best-of-four is an upper bound, reported separately from the deployed choice.", "",
            "| Method | Scored/planned states | Mean reference/transfer MR-STFT |", "|---|---:|---:|"]
    rows += [f"| {name} | {value['scored']}/{value['planned']} | {value['mean_mrstft']:.4f} |" for name, value in means.items()]
    rows += ["", f"Deployed-minus-model paired mean: {paired['mean_delta']:+.4f}; descriptive 95% state-bootstrap interval: {paired['ci95']}. Wins: {paired['wins']}/{paired['planned']}; ties: {paired['ties']}. The two performances of each patch stay together. These are reused development states, not a fresh final test.", "",
             f"Ranked coverage: {100 * qualification['ranked_coverage']:.1f}%. Model-warm ranking p95: {qualification['warm_p95_seconds']:.2f}s; bundle cold load: {summary['cold_bundle_load_seconds']:.2f}s. Each candidate render still starts a fresh native process. Accepted native attempts: {summary['renders_accepted']}/{summary['renders_attempted']}. Sampled process-tree peak: {summary['peak_process_tree_rss_bytes'] / 2**20:.1f} MiB. Elapsed check time excludes initial bundle creation and includes partial recovery when present.", "",
             "Independent local checks overlapped this CPU run, so these times describe that workload rather than an idle-machine throughput benchmark. No audio archive, test-split evaluation or paid compute was used. The ordinary top-1 path remains available. Inferred notes and heuristic velocities can limit ranking quality; diagnostic best-of-four must not be mistaken for deployed performance.", "",
             f"Validation: the training suite passed {args.training_tests_passed} tests; the renderer/stream suite passed {args.renderer_tests_passed} tests. Counts describe separate overlapping suites. Dedicated checks cover audio-only note estimation, four-render caps, invalid/no-note fallbacks, mono WAV handling, frontend equivalence, state/identity rejection, qualification guards and recovery caps. Ruff/diff checks pass; GitHub Actions stays disabled.", "",
             "[Usage and gates](../AUDIO_CANDIDATES.md) · [Frozen plan](data/audio_ranking_plan.json) · [Aggregate evidence](data/audio_ranking_summary.json)", ""]
    (reports / "AUDIO_RANKING_RESULTS.md").write_text("\n".join(rows))
    print(json.dumps({"qualified": qualification["qualified"], "relative_error_change_percent": change, "paired": paired}, indent=2))


if __name__ == "__main__":
    main()
