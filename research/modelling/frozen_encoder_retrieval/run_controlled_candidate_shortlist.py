from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys

PACKAGE = Path(__file__).resolve().parent
REPO = PACKAGE.parents[2]
for directory in (REPO / "research/data_generation", REPO / "research/training", PACKAGE):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import run_controlled_perturbation as controlled
from frozen_retrieval import evaluation
from frozen_retrieval.dataset import validate_dataset_manifests
from frozen_retrieval.encoders import file_sha256


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Screen active Vital controls and prepare a private listening shortlist")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--renderer-config", type=Path, required=True)
    parser.add_argument("--plugin-path", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-control", type=int, default=3)
    return parser


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _set_variant(preset, field: str, delta: float, family: str, route: dict | None = None) -> tuple[object, dict]:
    value = preset.to_dict()
    changed = preset.__class__(value, preset.schema)
    before_raw, before_norm = changed.get_raw(field), changed.get_normalized(field)
    after_norm = before_norm + delta if before_norm <= 1.0 - delta else before_norm - delta
    changed.set_normalized(field, after_norm)
    if changed.get_raw(field) == before_raw:
        raise ValueError(f"{field} perturbation was a no-op")
    details = {"family": family, "field": field, "before_raw": before_raw,
        "after_raw": changed.get_raw(field), "before_normalized": round(before_norm, 6),
        "after_normalized": round(after_norm, 6), "normalized_delta": round(after_norm - before_norm, 6)}
    if route:
        details.update(route)
    return changed, details


def _preset_candidates(preset, row: dict, source_root: Path) -> list[dict]:
    path = source_root / row["relative_path"]
    if hashlib.sha256(path.read_bytes()).hexdigest() != row["raw_sha256"]:
        raise ValueError(f"source preset changed after the frozen dataset was created: {row['state_id']}")
    validation = [item.to_dict() for item in preset.validate().diagnostics]
    settings = preset.to_dict()["settings"]
    candidates = []
    for index in (1, 2):
        on_field, field = f"filter_{index}_on", f"filter_{index}_cutoff"
        if settings.get(on_field, 0) <= 0.5 or field not in settings:
            continue
        variant, details = _set_variant(preset, field, 0.20, "filter")
        candidates.append({"state_id": row["state_id"], "preset_row": row, "preset": preset,
            "variant": variant, "details": details, "sort_value": preset.get_normalized(field),
            "diagnostics": validation})
        break
    field = "env_1_attack"
    if field in settings:
        variant, details = _set_variant(preset, field, 0.20, "envelope")
        candidates.append({"state_id": row["state_id"], "preset_row": row, "preset": preset,
            "variant": variant, "details": details, "sort_value": preset.get_normalized(field),
            "diagnostics": validation})
    routes = []
    for index, route in enumerate(settings.get("modulations", []), start=1):
        source, destination = route.get("source", ""), route.get("destination", "")
        match = re.fullmatch(r"lfo_(\d+)", source)
        if not match or not destination:
            continue
        if abs(float(settings.get(f"modulation_{index}_amount", 0))) <= 0.05:
            continue
        if float(settings.get(f"modulation_{index}_bypass", 0)) > 0.5:
            continue
        if float(settings.get(f"{source}_sync", 0)) > 0.5:
            continue
        field = f"{source}_frequency"
        if field not in settings:
            continue
        routes.append((abs(float(settings[f"modulation_{index}_amount"])), index, route, field))
    if routes:
        amount, index, route, field = max(routes, key=lambda item: (item[0], -item[1]))
        variant, details = _set_variant(preset, field, 0.20, "routed_lfo", {
            "route_source": route["source"], "route_destination": route["destination"],
            "route_amount": float(settings[f"modulation_{index}_amount"]), "route_index": index,
            "lfo_sync": False})
        candidates.append({"state_id": row["state_id"], "preset_row": row, "preset": preset,
            "variant": variant, "details": details, "sort_value": amount, "diagnostics": validation})
    return candidates


def _select_balanced(rows: list[dict], count: int, used_states: set[str]) -> list[dict]:
    available = sorted((row for row in rows if row["state_id"] not in used_states),
                       key=lambda row: (row["sort_value"], row["state_id"]))
    selected = []
    if not available or count <= 0:
        return selected
    targets = [round((index + 0.5) * (len(available) - 1) / count) for index in range(count)]
    for target in targets:
        if len(selected) >= count:
            break
        unused = [row for row in available if row not in selected]
        selected.append(min(unused, key=lambda row: (abs(available.index(row) - target), row["state_id"])))
    for row in available:
        if len(selected) >= count:
            break
        if row not in selected:
            selected.append(row)
    return selected


def _select_contexts(performances: list[dict]) -> list[dict]:
    def pick(articulation: str, extreme: str, used: set[str]) -> dict:
        rows = [row for row in performances if row.get("stats", {}).get("articulation") == articulation]
        if len(rows) < 2:
            raise ValueError(f"candidate screen needs both held and phrased MIDI examples; missing {articulation}")
        rows.sort(key=lambda row: (float(row.get("stats", {}).get("note_duration_median_seconds", 0)), row["midi_id"]))
        for row in (rows[0], rows[-1]) if extreme == "both" else ((rows[0],) if extreme == "short" else (rows[-1],)):
            if row["midi_id"] not in used:
                used.add(row["midi_id"])
                return row
        raise ValueError("candidate screen could not select four distinct short/long note examples")

    used: set[str] = set()
    return [pick("held", "short", used), pick("held", "long", used),
            pick("phrased", "short", used), pick("phrased", "long", used)]


def _source_wav(dataset: Path, receipt: dict) -> Path:
    path = dataset / receipt["wav_relative_path"]
    if not path.is_file() or file_sha256(path) != receipt.get("audio_sha256"):
        raise ValueError(f"cached baseline WAV failed receipt identity check: {receipt.get('request_id')}")
    if not receipt.get("accepted"):
        raise ValueError(f"candidate MIDI baseline was not accepted by frozen QA: {receipt.get('qa_status')}")
    return path


def _render_one(renderer, preset, midi: dict, output: Path, seed_id: str) -> dict:
    from obruxo_data.midi import Performance
    from obruxo_data.render.batch import run_batch
    from obruxo_data.render.base import RenderRequest
    from frozen_retrieval.rendering import _accepted, _qa_warnings

    request = RenderRequest(preset=preset, performance=Performance.from_dict(midi["excerpt"]),
        sample_rate=44_100, channels=2, tail_seconds=2.0, renderer_id=renderer.renderer_id)
    run_batch(renderer, [request], output, workers=1, resume=True)
    metadata_path, wav_path = output / f"{request.request_id}.json", output / f"{request.request_id}.wav"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    accepted, qa_status = _accepted(metadata)
    if not accepted:
        raise RuntimeError(f"candidate screen render failed frozen QA: {request.request_id}: {qa_status}")
    return {"seed_id": seed_id, "midi_id": midi["midi_id"], "request_id": request.request_id,
            "audio_sha256": file_sha256(wav_path), "wav_path": wav_path,
            "qa_status": qa_status, "qa_warnings": _qa_warnings(metadata)}


def _copy_example(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        destination.hardlink_to(source)
    except OSError:
        shutil.copy2(source, destination)


def run(args: argparse.Namespace) -> dict:
    import numpy as np
    import yaml
    from obruxo_data.render.base import FULL_CORPUS_RENDERER_ID
    from obruxo_data.render.vita import VitalRenderer
    from obruxo_data.vital import VitalPreset

    dataset = args.dataset.resolve(strict=True)
    summary = json.loads((dataset / "dataset_summary.private.json").read_text(encoding="utf-8"))
    validate_dataset_manifests(dataset, summary)
    if summary.get("source_split") != "validation" or summary.get("performance_source_split") != "train":
        raise ValueError("candidate screen requires the approved validation preset gallery and held-out train-MIDI cohort")
    presets = _jsonl(dataset / "presets.private.jsonl")
    plans = _jsonl(dataset / "render_plan.private.jsonl")
    receipts = evaluation._render_receipts(dataset, plans)
    performances = [row for row in _jsonl(dataset / "performances.private.jsonl")
                    if row.get("performance_source_split") == summary.get("performance_source_split")]
    qualified = json.loads((dataset / "candidate_sets.render_qualified.private.json").read_text(encoding="utf-8"))
    eligible_states = {state_id for candidate in qualified["candidate_sets"].values() for state_id in candidate["state_ids"]}
    source_root = Path(summary["source_root"]).resolve(strict=True)

    inventory: dict[str, list[dict]] = defaultdict(list)
    for row in sorted(presets, key=lambda item: item["state_id"]):
        if row["state_id"] not in eligible_states:
            continue
        preset = VitalPreset.load(source_root / row["relative_path"])
        for candidate in _preset_candidates(preset, row, source_root):
            inventory[candidate["details"]["family"]].append(candidate)
    if args.per_control < 1 or args.per_control * 3 < 5:
        raise ValueError("per-control screening must yield at least five candidate presets")

    chosen, used_states = [], set()
    for family in ("filter", "envelope", "routed_lfo"):
        selected = _select_balanced(inventory[family], args.per_control, used_states)
        chosen.extend(selected)
        used_states.update(row["state_id"] for row in selected)
    if not 5 <= len(chosen) <= 10:
        raise ValueError(f"active controls yielded {len(chosen)} candidates; expected a 5–10 preset shortlist")

    plan_by_pair = {(row["state_id"], row["midi_id"]): row for row in plans}
    common_performances = [midi for midi in performances if all(
        (plan := plan_by_pair.get((candidate["state_id"], midi["midi_id"]))) and
        receipts.get(plan["request_id"], {}).get("accepted") for candidate in chosen)]
    contexts = _select_contexts(common_performances)
    renderer_config = yaml.safe_load(args.renderer_config.read_text(encoding="utf-8"))
    if renderer_config.get("renderer_id") != FULL_CORPUS_RENDERER_ID:
        raise ValueError("candidate screen requires the corpus-qualified Vital renderer")
    output = args.output.resolve()
    render_dir = output / "renders"
    render_dir.mkdir(parents=True, exist_ok=True)
    renderer = VitalRenderer.from_config(args.renderer_config, plugin_path=args.plugin_path)
    if renderer.renderer_id != FULL_CORPUS_RENDERER_ID or renderer.max_workers != 1:
        raise ValueError("candidate screen renderer differs from the qualified single-worker Vital profile")

    candidates = []
    for index, candidate in enumerate(chosen, start=1):
        preset, variant = candidate["preset"], candidate["variant"]
        control = candidate["details"]
        original_codes = Counter(json.dumps(item, sort_keys=True) for item in candidate["diagnostics"])
        variant_codes = Counter(json.dumps(item.to_dict(), sort_keys=True) for item in variant.validate().diagnostics)
        if variant_codes - original_codes:
            raise ValueError(f"candidate {candidate['state_id']} control adds Vital validation diagnostics")
        repeat = controlled._seeded_preset(preset, 1, VitalPreset)
        measured = []
        for midi in contexts:
            plan = plan_by_pair[(candidate["state_id"], midi["midi_id"])]
            before = _source_wav(dataset, receipts[plan["request_id"]])
            repeated = _render_one(renderer, repeat, midi, render_dir, "independent_repeat")
            after = _render_one(renderer, variant, midi, render_dir, "changed_source_seed")
            before_descriptor = controlled._spectral_descriptor(before, np)
            repeat_descriptor = controlled._spectral_descriptor(repeated["wav_path"], np)
            after_descriptor = controlled._spectral_descriptor(after["wav_path"], np)
            if before_descriptor is None or repeat_descriptor is None or after_descriptor is None:
                raise RuntimeError(f"candidate acoustic descriptor missing for {candidate['state_id']}/{midi['midi_id']}")
            repeat_distance = 1.0 - controlled._cosine(before_descriptor, repeat_descriptor, np)
            change_distance = 1.0 - controlled._cosine(before_descriptor, after_descriptor, np)
            measured.append({"midi_id": midi["midi_id"], "articulation": midi["stats"]["articulation"],
                "note_duration_median_seconds": midi["stats"]["note_duration_median_seconds"],
                "duration_seconds": midi["stats"]["duration_seconds"], "note_count": midi["stats"]["note_count"],
                "change_spectral_distance": change_distance, "repeat_spectral_distance": repeat_distance,
                "change_to_repeat_ratio": change_distance / max(repeat_distance, 1e-8),
                "before_wav": str(before), "after_wav": str(after["wav_path"]),
                "repeat_wav": str(repeated["wav_path"]), "after_request_id": after["request_id"],
                "repeat_request_id": repeated["request_id"]})
        change_values = [row["change_spectral_distance"] for row in measured]
        repeat_values = [row["repeat_spectral_distance"] for row in measured]
        ratios = [row["change_to_repeat_ratio"] for row in measured]
        candidate_id = f"candidate_{index:02d}"
        # Listening files use short- and long-note contexts. Before audio is hard-linked from the verified frozen render.
        examples = []
        measurement_by_midi = {row["midi_id"]: row for row in measured}
        for midi in (contexts[0], contexts[1]):
            midi_id = midi["midi_id"]
            paths = {}
            for kind in ("before", "after", "repeat"):
                source_path = Path(measurement_by_midi[midi_id][f"{kind}_wav"])
                destination = output / "listening_examples.private" / candidate_id / f"{kind}_{midi_id}.wav"
                _copy_example(source_path, destination)
                paths[kind] = str(destination)
            examples.append({"midi_id": midi_id, "articulation": midi["stats"]["articulation"],
                             "note_duration_median_seconds": midi["stats"]["note_duration_median_seconds"],
                             **paths})
        candidates.append({"candidate_id": candidate_id, "state_id": candidate["state_id"],
            "family_id": candidate["preset_row"].get("family_id"), "display_name": candidate["preset_row"].get("display_name"),
            "control": control, "midi_measurements": measured,
            "mean_change_spectral_distance": sum(change_values) / len(change_values),
            "mean_repeat_spectral_distance": sum(repeat_values) / len(repeat_values),
            "median_change_to_repeat_ratio": sorted(ratios)[len(ratios) // 2],
            "performances_exceeding_repeat_variation": sum(change > repeat for change, repeat in zip(change_values, repeat_values, strict=True)),
            "listening_examples": examples})

    summary = {"policy": "issue155_controlled_candidate_screen_v1", "status": "shortlist_prepared_waiting_for_user_selection",
        "selected_count": len(candidates), "scanned_render_qualified_presets": len(eligible_states),
        "presets_with_valid_control_candidates": {family: len(rows) for family, rows in sorted(inventory.items())},
        "shortlisted_by_control_family": dict(Counter(row["control"]["family"] for row in candidates)),
        "midi_context_count": len(contexts), "midi_contexts": [{"midi_id": row["midi_id"], **row["stats"]} for row in contexts],
        "new_vital_renders": len(candidates) * len(contexts) * 2,
        "acoustic_measure": "1 - cosine of normalized 64-band log-frequency spectral descriptors; compare source-seed control change with unchanged-preset independent-seed repeat",
        "candidate_selection": "three deterministic parameter-value quantiles per control family, avoiding repeat states; encoder retrieval scores were not used",
        "candidates": candidates}
    output.mkdir(parents=True, exist_ok=True)
    json_path = output / "controlled_candidate_shortlist.private.json"
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = ["# Controlled-preset candidate shortlist", "",
        f"Status: **{len(candidates)} automatically screened candidates; awaiting user choice.** The definitive controlled diagnostic has not been run.", "",
        f"Scanned {len(eligible_states)} render-qualified presets. Screened {sum(summary['presets_with_valid_control_candidates'].values())} active control instances; generated {summary['new_vital_renders']} new repeat/changed Vital renders across {len(contexts)} existing train-MIDI contexts. Baseline audio was reused from SHA-verified cached renders.", "",
        "Candidates were sampled at deterministic low/middle/high parameter quantiles within each control family, with distinct preset states where possible. Acoustic scores did not influence which presets were included. The spectral distance is descriptive and uses four performances; values are not encoder retrieval results or inferential claims.", ""]
    for candidate in candidates:
        control = candidate["control"]
        lines.extend([f"## {candidate['candidate_id']} — {control['family']}", "",
            f"- Preset state: `{candidate['state_id']}`" + (f" ({candidate['display_name']})" if candidate.get("display_name") else ""),
            f"- Control: `{control['field']}` raw {control['before_raw']} → {control['after_raw']}; normalized {control['before_normalized']:.3f} → {control['after_normalized']:.3f}" +
                (f"; route {control.get('route_source')} → {control.get('route_destination')} (amount {control.get('route_amount'):.3f})" if control.get("route_source") else ""),
            f"- Mean changed-audio spectral distance: {candidate['mean_change_spectral_distance']:.5f}; mean unchanged-preset repeat distance: {candidate['mean_repeat_spectral_distance']:.5f}; median change/repeat ratio: {candidate['median_change_to_repeat_ratio']:.2f}×. Change exceeded repeat distance on {candidate['performances_exceeding_repeat_variation']}/{len(contexts)} MIDI excerpts.",
            "- MIDI-specific measurements:",
            "", "| MIDI | Articulation | Median note (s) | Change distance | Repeat distance | Ratio |", "|---|---|---:|---:|---:|---:|"])
        lines.extend(f"| `{row['midi_id']}` | {row['articulation']} | {row['note_duration_median_seconds']:.3f} | {row['change_spectral_distance']:.5f} | {row['repeat_spectral_distance']:.5f} | {row['change_to_repeat_ratio']:.2f}× |" for row in candidate["midi_measurements"])
        lines.extend(["", "Listening WAVs (before / changed / unchanged-preset repeat):"])
        for example in candidate["listening_examples"]:
            lines.append(f"- `{example['midi_id']}` ({example['articulation']}, median note {example['note_duration_median_seconds']:.3f}s): " +
                " · ".join(f"[{kind}]({example[kind]})" for kind in ("before", "after", "repeat")))
        lines.append("")
    (output / "controlled_candidate_shortlist.private.md").write_text("\n".join(lines), encoding="utf-8")
    return summary


def main() -> int:
    args = _parser().parse_args()
    summary = run(args)
    print(json.dumps({key: value for key, value in summary.items() if key != "candidates"}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
