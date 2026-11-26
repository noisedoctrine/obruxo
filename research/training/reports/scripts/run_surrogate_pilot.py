"""Run the matched basic and expanded Vital surrogate comparison."""

# ruff: noqa: E402
import argparse
from collections import Counter
from importlib.metadata import version
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import torch

TRAINING = Path(__file__).resolve().parents[2]
RESEARCH = TRAINING.parent
sys.path.insert(0, str(RESEARCH / "data_generation"))
sys.path.insert(0, str(TRAINING))

from obruxo_data.hashing import canonical_sha256, sha256_bytes

from obruxo_training.audit import write_json
from obruxo_training.benchmark import MemoryMonitor
from obruxo_training.codec import PilotCodec, PilotTarget
from obruxo_training.evaluate import audio_distances
from obruxo_training.performances import initial_bank
from obruxo_training.surrogate_backbones import BACKBONES
from obruxo_training.surrogate_codec import SurrogateCodec
from obruxo_training.surrogate_pilot import Progress, collect, fit, native_session, source_fingerprints
from obruxo_training.surrogate_pilot_data import make_cohorts
from obruxo_training.surrogate_pilot_eval import evaluate_panel


def probe_native(codec, render):
    default = codec.from_basic(codec.basic.default_target())
    performance = initial_bank()[0].performance()

    def patch(*, shape=0, filtered=0, blend=0, osc2=0, wet=0):
        values = list(default.continuous)
        if filtered:
            values[3] = .38
        if wet:
            values[-2:] = [1., 1.]
        return codec.canonical(PilotTarget(tuple(values), (0, filtered, shape, blend, osc2, 0, 0, wet)))

    variants = {"init": default, "sine": patch(shape=1), "triangle": patch(shape=2), "square": patch(shape=3),
                "low-pass": patch(filtered=1), "band-pass": patch(filtered=1, blend=1),
                "high-pass": patch(filtered=1, blend=2), "oscillator-2": patch(osc2=1), "reverb": patch(wet=1)}
    outcomes = render([(f"preflight:{key}", target, performance) for key, target in variants.items()])
    if any(audio is None for audio, _ in outcomes):
        failures = {key: meta for key, (audio, meta) in zip(variants, outcomes, strict=True) if audio is None}
        raise RuntimeError(f"expanded native qualification failed: {failures}")
    audio = [value for value, _ in outcomes]
    distances = audio_distances(np.stack([audio[0]] * (len(audio) - 1)), np.stack(audio[1:]))
    differences = {key: item["mrstft"] for key, item in zip(tuple(variants)[1:], distances, strict=True)}
    if any(value <= .01 for value in differences.values()):
        raise RuntimeError(f"expanded control produced no audible preflight difference: {differences}")
    first, last = round(.35 * 44100), round(1.2 * 44100)
    peak_bins = {}
    for key, audio_value in zip(variants, audio, strict=True):
        if not key.endswith("-pass"):
            continue
        excerpt = audio_value[first:last].mean(1)
        power = np.abs(np.fft.rfft(excerpt * np.hanning(len(excerpt)))) ** 2
        peak_bins[key] = float(np.dot(np.fft.rfftfreq(len(excerpt), 1 / 44100), power) / power.sum())
    if not peak_bins["low-pass"] < peak_bins["band-pass"] < peak_bins["high-pass"]:
        raise RuntimeError(f"filter modes did not produce ordered spectral centroids: {peak_bins}")
    wet_qa = outcomes[-1][1]["qa"]
    if wet_qa["tail_rms"] > wet_qa["tail_rms_warning"]:
        raise RuntimeError("the maximum reverb setting needs a longer render tail")
    return {"native_mrstft_from_init": differences, "filter_spectral_centroid_hz": peak_bins,
            "wet_tail_rms": wet_qa["tail_rms"], "wet_tail_warning_threshold": wet_qa["tail_rms_warning"],
            "settled_tail_seconds": codec.basic.contract.tail_seconds}


def run(args):
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    progress = Progress(output)
    stages = ("basic", "expanded")
    original = PilotCodec()
    codecs = {stage: SurrogateCodec(original, expanded=stage == "expanded") for stage in stages}
    started = perf_counter()
    monitor = MemoryMonitor()
    monitor.start()
    try:
        progress.set("Preparing data and control contracts", 1, 0)
        plan_path = output / "plan.json"
        profiles = {"format": "matched_two_stage_surrogate_v1", "steps_per_arm_per_stage": args.steps,
                    "inversion_steps": args.inversion_steps, "device": str(args.device), "threads": args.threads,
                    "render_workers": args.workers, "torch": str(torch.__version__), "numpy": np.__version__,
                    "vita": version("vita"), "dawdreamer": version("dawdreamer"),
                    "preflight_only": args.probe, "smoke": args.smoke,
                    "sources": {**source_fingerprints(), Path(__file__).name: sha256_bytes(Path(__file__).read_bytes())}}
        cohorts = make_cohorts(args.pool, args.reservation, codecs["basic"], codecs["expanded"], smoke=args.smoke)
        plan = {**profiles, **cohorts}
        plan_id = canonical_sha256(plan)
        if plan_path.exists():
            prior = json.loads(plan_path.read_text(encoding="utf-8"))
            if prior["id"] != plan_id or canonical_sha256(prior["plan"]) != plan_id:
                raise ValueError("run plan differs from the frozen manifest")
        else:
            write_json(plan_path, {"id": plan_id, "plan": plan})
        inventory = write_control_inventory(output, codecs)
        counts = {}
        for stage, cohort in cohorts["stages"].items():
            roots = {row["state_id"]: row["target"] for row in cohort["pairs"]}
            evaluation = cohort["cases"]
            counts[stage] = {"new_training_roots": len(roots), "new_training_clips": 2 * len(cohort["pairs"]),
                             "wet_new_training_roots": sum(bool(target["categorical"][7]) for target in roots.values())
                             if stage == "expanded" else 0,
                             "training_midi": len({row["midi_id"] for row in cohort["pairs"]}),
                             "evaluation_roots": len({row["state_id"] for row in evaluation}),
                             "evaluation_cases": dict(Counter(row["midi_role"] for row in evaluation)),
                             "evaluation_midi": {role: len({row["midi_id"] for row in evaluation if row["midi_role"] == role})
                                                 for role in ("seen", "new")}}
        write_json(output / "cohort-counts.json", counts)
        progress.update(1)
        preflight_path = output / "preflight.json"
        if not preflight_path.exists():
            with native_session(output, codecs["expanded"], args.workers) as render:
                write_json(preflight_path, probe_native(codecs["expanded"], render))
        if args.probe:
            progress.update(1, state="complete", message="Native capability preflight")
            return
        accepted_basic = []
        summaries = {}
        for phase, stage in enumerate(stages):
            codec = codecs[stage]
            counts = cohorts["stages"][stage]
            offset = 5 * phase
            progress.set(f"{stage.title()} training pairs", len(counts["pairs"]), 1 + offset)
            with native_session(output, codec, args.workers) as render:
                new_rows, collection = collect(output, counts["pairs"], render, progress)
            if stage == "basic":
                accepted_basic = new_rows
            all_rows = accepted_basic if stage == "basic" else accepted_basic + new_rows
            models, fits = {}, {}
            for index, backbone in enumerate(BACKBONES):
                progress.set(f"{stage.title()} training · {backbone}", args.steps, 2 + index + offset)
                models[backbone], fits[backbone] = fit(output, stage, backbone, codec, all_rows, steps=args.steps,
                                                       plan_id=plan_id, device=args.device, progress=progress)
            sizes = [item["parameters"] for item in fits.values()]
            if max(sizes) > 1.1 * min(sizes):
                raise RuntimeError("matched pilot parameter counts differ by more than 10%")
            progress.set(f"{stage.title()} real-Vital evaluation", len(counts["cases"]), 5 + offset)
            with native_session(output, codec, args.workers) as render:
                evaluation = evaluate_panel(output, stage, counts["cases"], models, codec, codec, render,
                                            device=args.device, steps=args.inversion_steps, progress=progress)
            summaries[stage] = {"collection": collection, "fit": fits, "evaluation": evaluation,
                                "total_training_pairs": len(all_rows)}
            write_json(output / f"{stage}-summary.json", summaries[stage])
        basic_cases = cohorts["stages"]["basic"]["cases"]
        progress.set("Expanded models on the basic panel", len(basic_cases), 11)
        with native_session(output, codecs["basic"], args.workers) as render:
            summaries["preservation"] = evaluate_panel(output, "preservation", basic_cases, models, codecs["expanded"],
                                                        codecs["basic"], render, device=args.device,
                                                        steps=args.inversion_steps, progress=progress)
        progress.set("Writing results", 1, 12)
        result = {"plan_id": plan_id, "smoke": args.smoke, "panels": summaries,
                  "peak_process_tree_bytes": monitor.peak_bytes, "elapsed_seconds_this_invocation": perf_counter() - started}
        write_json(output / "summary.json", result)
        write_report(output, result, inventory)
        progress.update(1, state="complete", message="Summary and report saved")
    except BaseException as error:
        progress.update(progress.done, state="failed", message=f"{type(error).__name__}: {error}")
        raise
    finally:
        monitor.close()


def write_report(output, result, inventory):
    summaries = result["panels"]
    paragraphs = ["# Vital surrogate pilot", ""]
    if result["smoke"]:
        paragraphs += ["This reduced smoke run checks the native and software paths.", ""]
    for label, panel in (("Basic", summaries["basic"]["evaluation"]), ("Expanded", summaries["expanded"]["evaluation"]),
                         ("Expanded on the basic panel", summaries["preservation"])):
        candidate = panel["candidate"]
        signal = f"{candidate} led the eligible backbones" if candidate else "no backbone met the real-Vital inversion thresholds"
        paragraphs += [f"## {label}", "", f"On this panel, {signal}. Positive improvement means the optimized patch rendered "
                       "closer to the target in stereo MR-STFT.", "",
                       "| Backbone | Seen MIDI improved | New MIDI improved | New MIDI mean native improvement | "
                       "New MIDI gain over forward-only | New MIDI forward error |",
                       "|---|---:|---:|---:|---:|---:|"]
        for name, groups in panel["backbones"].items():
            seen, new = groups["seen"], groups["new"]
            gain = new.get("mean_native_improvement")
            forward_only = new.get("mean_gain_over_forward_only")
            error = new.get("mean_forward_error")
            numbers = " | ".join(f"{value:+.4f}" if value is not None else "—" for value in (gain, forward_only, error))
            paragraphs.append(f"| {name} | {seen.get('improved', 0)}/{seen['planned']} | "
                              f"{new.get('improved', 0)}/{new['planned']} | {numbers} |")
        paragraphs += ["", f"Complete comparisons on **{panel['scorable_cases']}/{panel['planned_cases']}** planned cases. "
                       "Patch-root resampling intervals, wet/dry breakdowns, and forward/backward timings are in [summary.json](summary.json).", ""]
    for stage, rows in inventory.items():
        paragraphs += [f"## {stage.title()} Vital control scope", "",
                       "The representations below are inputs to the forward surrogate. Direct inversion optimizes the active "
                       "continuous inputs and searches legal categorical candidates.", "",
                       "| Control family | Changeable / total | Surrogate input slots |",
                       "|---|---:|---:|"]
        groups = {}
        for row in rows:
            groups.setdefault(row["scope_family"], []).append(row)
        for name, group in groups.items():
            active = [row for row in group if row["changeable"]]
            paragraphs.append(f"| {name} {'✅' if active else '❌'} | {len(active)} / {len(group)} | "
                              f"{sum(row['model_input_width'] for row in active)} |")
        paragraphs += ["", "[Exact per-control inventory](control-scope.jsonl)", ""]
    paragraphs += ["## Execution and reproducibility", "",
                   "Training used paired original and perturbed patches under identical MIDI. Expanded training reused the basic pairs "
                   "and widened each fitted model before adding the new renders. Evaluation used the same held-out root IDs "
                   "and exact seen/new MIDI assignments across stages. The confirmation and test partitions were reserved.", "",
                   "[Frozen plan](plan.json) · [Run metrics](summary.json) · [Control inventory](control-scope.jsonl)", ""]
    (output / "REPORT.md").write_text("\n".join(paragraphs), encoding="utf-8")


def scope_family(row):
    key = row["control_key"]
    if key.startswith("osc_"):
        if key.endswith("_wavetable_asset"):
            return "Oscillator waveform assets"
        if key.endswith(("_level", "_pan", "_on", "_transpose", "_tune", "_unison_detune", "_unison_voices")):
            return "Oscillator pitch, level, pan, enable and unison"
        return "Oscillator phase, warp, frame and routing controls"
    if key.startswith("filter_") and not key.startswith("filter_fx"):
        return "Main filters"
    if key.startswith(("env_", "envelope_")):
        return "Envelopes"
    if key.startswith(("lfo_", "random_")):
        return "LFOs and random sources"
    if key.startswith("modulation"):
        return "Modulation routing and maps"
    if key.startswith(("sampler_", "sample_")) or key == "sample":
        return "Sampler and sample assets"
    if key.startswith("reverb_"):
        return "Reverb"
    if key.startswith(("chorus_", "compressor_", "delay_", "distortion_", "eq_", "filter_fx_", "flanger_", "phaser_")):
        return "Chorus, compression, delay, distortion, EQ, FX filter, flanger and phaser"
    return "Global, voice and performance controls"


def write_control_inventory(output, codecs):
    source = RESEARCH / "modelling/control_scope/stage_a_controls.jsonl"
    aliases = {"filter_1_response": "filter_1_blend", "osc_1_waveform": "osc_1_wavetable_asset",
               "osc_2_waveform": "osc_2_wavetable_asset"}
    template = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines()]
    inventory = {}
    with (output / "control-scope.jsonl").open("w", encoding="utf-8") as handle:
        for stage, codec in codecs.items():
            inventory[stage] = []
            continuous = {name: (lo, hi, default) for name, lo, hi, default in codec.continuous}
            categories = {aliases.get(name, name): options for name, options in codec.categorical}
            for old in template:
                row = {**old, "variant": stage, "experiment_id": "issue172_surrogate_pilot", "head_outputs": 0,
                       "scope_family": scope_family(old)}
                name = row["control_key"]
                if name in continuous:
                    lo, hi, default = continuous[name]
                    row.update(changeable=True, model_input_width=1, minimum=lo, maximum=hi, default_value=default,
                               fixed_value=None, fixed_reason=None, output_type="normalized_continuous_input",
                               update_method="projected autograd on surrogate input",
                               gradient_path="spectral/envelope error through frozen surrogate to active control",
                               training_signal="native absolute spectra and paired sound-change targets")
                elif name in categories:
                    options = list(categories[name])
                    row.update(changeable=True, model_input_width=1 if len(options) == 2 else len(options), choices=options,
                               fixed_value=None, fixed_reason=None, output_type="hard_category_input",
                               update_method="discrete candidate search with native verification", gradient_path="none",
                               training_signal="native absolute spectra under recorded categorical state")
                else:
                    row.update(changeable=False, model_input_width=0, fixed_value=row["default_value"], output_type="fixed")
                if name == "filter_1_blend":
                    row["active_condition"] = "filter_1_on"
                if name in {"osc_2_level", "osc_2_tune", "osc_2_transpose", "osc_2_wavetable_asset"}:
                    row["active_condition"] = "osc_2_on"
                if name in {"reverb_dry_wet", "reverb_decay_time"}:
                    row["active_condition"] = "reverb_on"
                inventory[stage].append(row)
                handle.write(json.dumps(row, sort_keys=True) + "\n")
            if sum(row["model_input_width"] for row in inventory[stage]) != codec.width:
                raise ValueError(f"{stage} control inventory and codec widths differ")
    return inventory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    default = TRAINING / "outputs/basic-confirmation-reserved-20261026"
    parser.add_argument("--pool", type=Path, default=default / "pool.json")
    parser.add_argument("--reservation", type=Path, default=default / "reservation.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--inversion-steps", type=int, default=16)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--probe", action="store_true")
    args = parser.parse_args()
    if min(args.steps, args.inversion_steps, args.threads, args.workers) < 1 or args.workers > 4:
        parser.error("counts must be positive and at most four renderer workers are supported")
    args.device = torch.device("cpu")
    torch.set_num_threads(args.threads)
    run(args)


if __name__ == "__main__":
    main()
