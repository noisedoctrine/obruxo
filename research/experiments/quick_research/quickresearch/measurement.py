from __future__ import annotations

from itertools import combinations
import math
from pathlib import Path

from .common import BudgetExpired, Context, digest, file_hash, finish_report, outcome_counts, read_json, write_csv, write_json
from .fixtures import load_manifest
from .renders import bank_identity, render_key, render_one, renderer_for, usable


def jobs_for(fixtures: list[dict]) -> list[tuple]:
    jobs = []
    for fixture in fixtures:
        jobs.extend((fixture, "held_c4", "baseline", repeat) for repeat in range(3))
        for control in fixture["controls"]:
            jobs.extend((fixture, "held_c4", control["name"], repeat) for repeat in range(2))
    return jobs


def analyze(context: Context, fixtures: list[dict], rows: list[dict]) -> tuple[list[dict], list[dict]]:
    from scipy.io import wavfile
    from .metrics import CONFIG, distances, separation
    index = {row["key"]: row for row in rows}
    comparisons, ranges = [], []
    cache_dir = context.output / "distance_cache"
    cache_dir.mkdir(exist_ok=True)

    def score(a, b):
        key = digest({"audio": [a["audio_sha256"], b["audio_sha256"]], "metrics": CONFIG,
                      "implementation": a["identity"]["implementation"]})
        cached = cache_dir / f"{key}.json"
        if cached.exists():
            return read_json(cached)
        context.check(3)
        rates_audio = [wavfile.read(row["wav"]) for row in (a, b)]
        if [item[0] for item in rates_audio] != [44100, 44100]:
            raise ValueError("metric audio rate mismatch")
        values = distances(rates_audio[0][1], rates_audio[1][1])
        write_json(cached, values)
        return values

    for fixture in fixtures:
        baseline = [index.get(render_key(fixture, "held_c4", "baseline", repeat)) for repeat in range(3)]
        baseline_ok = all(row is not None and usable(row) for row in baseline)
        unchanged = []
        if baseline_ok:
            for left, right in combinations(range(3), 2):
                values = score(baseline[left], baseline[right])
                unchanged.append(values)
                comparisons.append({"fixture_id": fixture["fixture_id"], "control": "baseline", "kind": "unchanged",
                                    "left_repeat": left, "right_repeat": right, **values})
        for control in fixture["controls"]:
            changed_rows = [index.get(render_key(fixture, "held_c4", control["name"], repeat)) for repeat in range(2)]
            valid = baseline_ok and all(row is not None and usable(row) for row in changed_rows)
            changes = []
            if valid:
                for repeat, row in enumerate(changed_rows):
                    values = score(baseline[0], row)
                    changes.append(values)
                    comparisons.append({"fixture_id": fixture["fixture_id"], "control": control["name"], "kind": "changed",
                                        "left_repeat": 0, "right_repeat": repeat, **values})
            for metric in ("waveform_rmse", "spectral_distance", "envelope_mae"):
                result = separation([value[metric] for value in unchanged], [value[metric] for value in changes])
                ranges.append({"fixture_id": fixture["fixture_id"], "category": fixture["category"], "control": control["name"],
                               "metric": metric, **result})
    return comparisons, ranges


def plot_ranges(output: Path, ranges: list[dict]) -> str | None:
    available = [row for row in ranges if "changed_min" in row]
    if not available:
        return None
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, max(4, len(available) / 9)))
    for axis, metric in zip(axes, ("waveform_rmse", "spectral_distance", "envelope_mae")):
        group = [row for row in available if row["metric"] == metric]
        for index, row in enumerate(group):
            axis.plot([row["unchanged_min"], row["unchanged_max"]], [index + .12] * 2, "o-", color="gray", markersize=3)
            axis.plot([row["changed_min"], row["changed_max"]], [index - .12] * 2, "o-", color="tab:blue", markersize=3)
        axis.set(yticks=range(len(group)), yticklabels=[f"{row['fixture_id']} / {row['control']}" for row in group],
                 title=metric, xlabel="gray: repeat; blue: changed")
        positive = [row[key] for row in group for key in ("unchanged_min", "unchanged_max", "changed_min", "changed_max")
                    if row[key] > 0]
        if positive:
            low = math.floor(math.log10(min(positive)))
            exponents = range(low, math.floor(math.log10(max(positive))) + 1)
            ticks = [0, *(10**exponent for exponent in exponents)]
            axis.set_xscale("symlog", linthresh=10**low)
            axis.set_xlim(left=0)
            axis.set_xticks(ticks, ["0", *(f"$10^{{{exponent}}}$" for exponent in exponents)])
        axis.grid(alpha=.2)
    fig.tight_layout()
    target = output / "images/measurement_ranges.png"
    target.parent.mkdir(exist_ok=True)
    fig.savefig(target, dpi=140)
    plt.close(fig)
    return "images/measurement_ranges.png"


def run(context: Context, args) -> dict:
    from .metrics import CONFIG
    fixtures = load_manifest(Path(args.fixtures))
    renderer = renderer_for(args)
    jobs = jobs_for(fixtures["fixtures"])
    directory = context.output / "renders"
    rows, timings = [], []
    reason = None
    write_json(context.output / "bank.json", {"fixtures_hash": fixtures["fixtures_hash"], "fixtures": fixtures["fixtures"],
               "bank_id": bank_identity(fixtures, renderer, args.implementation_hash), "renderer_id": renderer.renderer_id,
               "renderer_config_sha256": file_hash(Path(args.renderer_config)),
               "implementation_hash": args.implementation_hash, "metrics": CONFIG, "expected_renders": len(jobs)})
    try:
        for index, job in enumerate(jobs):
            context.check(20)
            if timings:
                missing = sum(not (directory / f"{render_key(*pending)}.json").exists() for pending in jobs[index:])
                estimate = 1.25 * max(timings) * missing + 20
                if missing and estimate > context.remaining:
                    reason = "projected_budget_exceeded"
                    break
            row = render_one(context, renderer, *job, directory, args.implementation_hash)
            rows.append(row)
            if not row["cache_hit"]:
                timings.append(row["render_seconds"])
            write_json(context.output / "render_manifest.json", rows)
            context.progress(completed_renders=len(rows), expected_renders=len(jobs), outcomes=outcome_counts(rows))
    except BudgetExpired:
        reason = "budget_exhausted"
    comparisons, ranges = [], []
    image = None
    try:
        comparisons, ranges = analyze(context, fixtures["fixtures"], rows)
        write_csv(context.output / "comparisons.csv", comparisons)
        write_csv(context.output / "measurement_ranges.csv", ranges)
        context.check(2)
        image = plot_ranges(context.output, ranges)
    except BudgetExpired:
        reason = "budget_exhausted"
    write_json(context.output / "render_manifest.json", rows)
    render_manifest_path = context.output / "render_manifest.json"
    bank_path = context.output / "bank.json"
    complete = len(rows) == len(jobs) and reason is None
    good = sum(usable(row) for row in rows)
    state = reason or ("complete" if good == len(jobs) else "complete_with_failures")
    summary = {"complete": complete, "state": state, "expected_renders": len(jobs), "attempted": len(rows), "valid": good,
               "outcomes": outcome_counts(rows), "metrics": CONFIG, "observed_separations": sum(row["interpretation"] == "observed_separation" for row in ranges),
               "comparison_rows": len(comparisons), "rehearsal_render_seconds": timings[:3],
               "render_manifest_sha256": file_hash(render_manifest_path), "bank_sha256": file_hash(bank_path)}
    write_json(context.output / "measurement_summary.json", summary)
    report = (f"# Measurement signal and variability\n\nStatus: {state}. Attempted {len(rows)}/{len(jobs)} renders; "
              f"{good} passed QA. Observed separation in {summary['observed_separations']} fixture/control/metric combinations.\n\n"
              "The unchanged range uses all three baseline pairs. Changed comparisons share baseline repeat 0; comparisons are not independent replicates. "
              "QA failures and missing comparisons cannot establish separation. Silent or clipped perturbations remain treatment outcomes.\n\n"
              "Separation establishes detection on these fixtures at a 5% raw-range perturbation, not perceptual importance or a universally superior loss. "
              "Overlap can reflect stochastic variance, inaudible routing, or a small response; it does not establish unidentifiability.\n\n")
    if image:
        report += f"![Observed unchanged and changed distance ranges]({image})\n\nGray: repeat variation; blue: parameter change. "
        report += "The x-axis is diagnostic distance on a symmetric-log scale. Separation supports local detectability; lower is not automatically better.\n"
    report += "\nFull local evidence: measurement_ranges.csv, comparisons.csv, render_manifest.json, and measurement_summary.json.\n"
    return finish_report(context, "MEASUREMENT_SIGNAL_REPORT.md", report, state, expected=len(jobs), attempted=len(rows), valid=good)
