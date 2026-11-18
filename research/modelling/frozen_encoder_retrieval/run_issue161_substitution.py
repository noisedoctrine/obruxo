from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[2]
sys.path[:0] = [str(ROOT), str(REPO / "research/data_generation"), str(REPO / "research/training")]

from frozen_retrieval.issue161_substitution import (  # noqa: E402
    EXPECTED_VARIANTS, MODEL_ORDER, SEED, build_results, challenger_reference_choices, load_scoring_midi, preselect_triplets, raw_mrstft_matrix,
    retrieve_alternatives, score_secondary_metrics, verify_triplet_wavs,
)
from frozen_retrieval.dataset import validate_dataset_manifests  # noqa: E402
from frozen_retrieval.sensitivity import _jsonl, verify_embedding_cache  # noqa: E402


DATASET = ROOT / "outputs" / "issue152-validation-train-midi-v3-50"
PRIVATE = ROOT / "outputs" / "issue161-audible-substitute"
PUBLIC_RESULTS = ROOT / "reports" / "issue161_substitution_results.json"
REPORT = ROOT / "reports" / "FROZEN_ENCODER_RETRIEVAL_RESULTS.md"


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _private_plan(performance_rows: list[dict], seed: int) -> dict:
    selected = preselect_triplets(performance_rows, seed=seed)
    return selected


def _resume_inputs(dataset: Path, output: Path, summary: dict, plan: dict) -> tuple[dict, dict, dict, dict]:
    """Reuse choices made after the full cache audit while rechecking their source contract."""
    validate_dataset_manifests(dataset, summary)
    evaluation = json.loads((dataset / "evaluation.private.json").read_text(encoding="utf-8"))
    expected_arms = {f"{model}:{variant}" for model, variants in EXPECTED_VARIANTS.items() for variant in variants}
    if (set(evaluation.get("available_arms", [])) != expected_arms or
            evaluation.get("held_out_reserved_sources_used") is not False):
        raise ValueError("saved retrieval choices do not match the audited eleven-arm evaluation contract")

    family_by_state = {row["state_id"]: row["family_id"] for row in _jsonl(dataset / "presets.private.jsonl")}
    if len(family_by_state) < 129:
        raise ValueError("preset family manifest is incomplete")
    anchor_arm = sorted(expected_arms)[0]
    metric_rows = [row for row in _jsonl(dataset / "episode_metrics.private.jsonl")
                   if row.get("condition") == "cross_midi_common_gallery"]
    rows_by_arm = {arm: [row for row in metric_rows if row.get("arm") == arm] for arm in expected_arms}
    anchor_pairs = {(row["target_state_id"], row["query_midi_id"]) for row in rows_by_arm[anchor_arm]}
    if len(rows_by_arm[anchor_arm]) != 6_450 or len(anchor_pairs) != 6_450 or any(
            len(rows) != 6_450 or {(row["target_state_id"], row["query_midi_id"]) for row in rows} != anchor_pairs
            for rows in rows_by_arm.values()):
        raise ValueError("saved evaluation arm episodes no longer share the complete 129 by 50 matrix")
    cohort_pairs = anchor_pairs
    preset_ids = sorted({state_id for state_id, _ in cohort_pairs})
    midi_ids = sorted({midi_id for _, midi_id in cohort_pairs})
    if len(cohort_pairs) != 6_450 or len(preset_ids) != 129 or len(midi_ids) != 50:
        raise ValueError("saved evaluation no longer identifies the complete 129 by 50 cohort")
    family_ids = {family_by_state[state_id] for state_id in preset_ids}
    if len(family_ids) != 129:
        raise ValueError("saved comparison states do not map one-to-one to source families")
    if not set(item["midi_id"] for triplet in plan["triplets"] for item in triplet["roles"]) <= set(midi_ids):
        raise ValueError("preselected triplet IDs are not in the audited common MIDI matrix")

    render_plans = _jsonl(dataset / "render_plan.private.jsonl")
    receipts = {row["request_id"]: row for row in _jsonl(dataset / "render_receipts.private.jsonl")}
    audit = {"preset_ids": preset_ids, "midi_ids": midi_ids, "family_ids": sorted(family_ids),
             "family_by_state": family_by_state, "receipts": receipts,
             "common_cells": len(cohort_pairs), "evaluation_arms": len(expected_arms),
             "source_split": summary.get("source_split"), "performance_source_split": summary.get("performance_source_split"),
             "reserved_split_access": summary.get("reserved_split_access")}

    choices_path = output / "retrieval_choices.private.json"
    choices = json.loads(choices_path.read_text(encoding="utf-8"))
    if set(choices) != expected_arms:
        raise ValueError("saved retrieval choices do not contain all eleven representation arms")
    target_set = set(preset_ids)
    for arm, rows in choices.items():
        expected_pairs = {(triplet["triplet_index"], state_id) for triplet in plan["triplets"] for state_id in preset_ids}
        actual_pairs = {(row.get("triplet_index"), row.get("target_state_id")) for row in rows}
        if len(rows) != 1_548 or actual_pairs != expected_pairs:
            raise ValueError(f"saved retrieval denominator is incomplete or duplicated for {arm}")
        for row in rows:
            triplet = plan["triplets"][row["triplet_index"]]
            midi1, midi2, midi3 = triplet["roles"]
            target, candidate = row["target_state_id"], row["candidate_state_id"]
            if (target not in target_set or candidate not in target_set or
                    family_by_state[target] != row.get("target_family_id") or
                    family_by_state[candidate] == family_by_state[target] or
                    row.get("candidate_rank") != 1 or row.get("eligible_candidates") != 128 or
                    not np.isfinite(row.get("cosine_similarity", np.nan)) or
                    (row.get("midi1_pattern"), row.get("midi2_pattern"), row.get("midi3_pattern"),
                     row.get("midi3_register"), row.get("midi3_articulation")) !=
                    (midi1["pattern"], midi2["pattern"], midi3["pattern"], midi3["register"], midi3["articulation"])):
                raise ValueError(f"saved retrieval choice violates the common different-family pool for {arm}")

    prior_wav_path = output / "wav_verification.private.json"
    prior_wav = json.loads(prior_wav_path.read_text(encoding="utf-8"))
    if prior_wav.get("selected_matrix_wavs_verified") != 4_644 or prior_wav.get("selected_unique_midis") != 36:
        raise ValueError("resume requires the saved successful 4,644-WAV triplet verification")
    prior_wav["midi3_scoring_wavs_loaded"] = 0
    return audit, render_plans, choices, {"evaluation": evaluation, "prior_wav_coverage": prior_wav}


def _format_ci(value: dict, *, scale: float = 1.0, digits: int = 3) -> str:
    estimate = value["mean"] * scale
    low, high = value["clustered_95_ci"]
    return f"{estimate:.{digits}f} [{low * scale:.{digits}f}, {high * scale:.{digits}f}]"


def _arm_label(arm: str) -> str:
    labels = {
        "basic_pitch_trunk:masked_mean": "Basic Pitch — mean",
        "basic_pitch_trunk:temporal_pyramid_8": "Basic Pitch — temporal",
        "beats_iter3_plus:masked_mean": "BEATs — mean",
        "beats_iter3_plus:temporal_pyramid_8": "BEATs — temporal",
        "cir_ast:official_clip": "CIR",
        "clap_music:official_clip": "CLAP",
        "dac_44khz:masked_mean": "DAC — mean",
        "dac_44khz:temporal_pyramid_8": "DAC — temporal",
        "encodec_24khz:masked_mean": "EnCodec — mean",
        "encodec_24khz:temporal_pyramid_8": "EnCodec — temporal",
        "muq_mulan:official_clip": "MuQ-MuLan",
    }
    return labels[arm]


def _markdown_section(results: dict) -> str:
    arms = results["arms"]
    ordered = sorted(arms, key=lambda arm: arms[arm]["original_level_mrstft"]["mean"])
    best, runner_up = ordered[:2]
    best_result = arms[best]
    top_delta = results["paired_comparisons_vs_cir"][best]
    coverage = results["triplet_coverage"]
    available = coverage["available_pattern_counts"]
    by_role = coverage["selected_pattern_counts_by_role"]
    unavailable = ", ".join(coverage["unavailable_patterns"]) or "none"
    wav_coverage = results.get("selected_wav_coverage", {})
    lines = [
        "<!-- issue161-audible-substitute-results:start -->",
        "### Issue #161 — Three-MIDI audible-substitute retrieval",
        "",
        (f"When the exact target preset is withheld, **{_arm_label(best)} had the lowest measured mean error** "
         f"({best_result['original_level_mrstft']['mean']:.3f} original-level MR-STFT) among 11 cached representations, "
         "on 1,548 choices per arm (129 families × 12 triplets; 36 distinct MIDI performances) from the same "
         "128-alternative pool."),
        (f"Its paired mean difference versus CIR was {_format_ci(top_delta)}; because that interval crosses zero, "
         "this exploratory sample does not establish a clear winner."),
        "",
        (f"The selected alternatives' mean error was {_format_ci(best_result['excess_over_oracle'])} above the "
         f"MIDI-3 oracle ({results['oracle']['mean']:.3f}), an evaluation-only best choice after the held-out sound test."),
        "",
        "A lower MR-STFT score means the retrieved preset's audio is closer to the target recording across three time-frequency scales. It is an acoustic distance, not a listener judgment.",
        "",
        "For example, `A1` is target preset A playing query MIDI 1. The encoder picks a different-family preset B by comparing its MIDI-2 embedding; then `A3` and `B3` are compared on the held-out third MIDI. B's MIDI-3 sound never affects which preset is retrieved.",
        "",
        "Recall@1 and Issue #157 columns are existing measurements over 6,450 common-gallery queries and 50 MIDI performances; the new acoustic columns use 1,548 matched target/triplet choices. They answer different questions and use different denominators.",
        "",
        "| Frozen representation | Exact-preset R@1 | #157 preset-dominant | Original-level MR-STFT ↓ | Excess over oracle ↓ | Mean rank / 128 ↓ | Better than CIR | Gain-matched MR-STFT ↓ | 10 ms envelope error ↓ | RMS error ↓ |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in ordered:
        row = arms[arm]
        lines.append(f"| {_arm_label(arm)} | {row['exact_preset_recall_at_1']:.1%} | "
            f"{row['issue157_preset_dominant_fraction']:.1%} | {_format_ci(row['original_level_mrstft'])} | "
            f"{_format_ci(row['excess_over_oracle'])} | {row['rank_among_128_alternatives']['mean']:.1f} | "
            f"{results['pairwise_wins'][arm]['cir_ast:official_clip']:.1%} | "
            f"{row['gain_matched_mrstft']['mean']:.3f} | {row['envelope_relative_rmse']['mean']:.3f} | "
            f"{row['global_rms_relative_error']['mean']:.3f} |")
    random = results["random_baseline"]
    oracle = results["oracle"]
    lines += [
        "| Fixed random alternative | — | — | " + _format_ci(random) + " | — | — | — | — | — | — |",
        "| MIDI-3 oracle (evaluation reference) | — | — | " + _format_ci(oracle) + " | 0.000 | 1.0 | — | — | — | — |",
        "",
        "The **Better than CIR** column is the paired fraction of identical target/triplet cases where that arm's selected preset had lower original-level error than CIR; ties count one half. The aggregate JSON contains the full paired 11 × 11 win matrix.",
        "",
        (f"**Sample and uncertainty.** Each arm made {results['episodes_per_arm']:,} matched choices: "
         f"129 independent preset families × 12 selected triplets, with 128 eligible alternative families per choice. "
         f"The triplets use {coverage['unique_selected_performances']} distinct MIDI performances across MIDI 1, 2, and 3. "
         "Intervals resample preset families and whole triplets in paired two-way draws (2,000 replicates); the 12-triplet "
         "axis is small, and intervals do not quantify new encoders, new MIDI populations, or an independent confirmation cohort."),
        "",
        "The oracle picks, for evaluation only, the lowest-error alternative after both candidates play the same held-out third performance. It never affects embeddings, triplet selection, or retrieval. The fixed random baseline chooses one eligible alternative per target/triplet from the same 128-state pool. Excess over oracle is the encoder choice's error minus that best available catalog choice; it cannot be negative.",
        "",
        "### Triplet diversity and performance breakdowns",
        "",
        "Triplets were selected from the actual MIDI event records before encoder rankings were read. The event tags include single-note melody/bass, simultaneous-note chords/polyphony, and repeating sequences; tags can overlap. A repeating-sequence tag requires repeated three/four-onset pitch-class motifs or a long, wide-register monophonic run. The selector rotates available types through the three roles and greedily balances articulation, register, velocity variation, and pitch span. Unavailable types are reported rather than inferred from audio.",
        "",
        "| MIDI pattern (overlapping event tags) | Available in 50-performance pool | MIDI 1 query | MIDI 2 retrieval | MIDI 3 sound test |",
        "|---|---:|---:|---:|---:|",
    ]
    for pattern in ("single_note_melody_bass", "chords_polyphonic", "repeating_sequence"):
        lines.append(f"| {pattern.replace('_', ' ')} | {available[pattern]} | {by_role['midi1_query'].get(pattern, 0)} | "
                     f"{by_role['midi2_retrieval'].get(pattern, 0)} | {by_role['midi3_sound_test'].get(pattern, 0)} |")
    midi3_patterns = results["strata"]["midi3_pattern"]
    midi3_registers = results["strata"]["midi3_register"]
    pattern_names = ("single_note_melody_bass", "chords_polyphonic", "repeating_sequence")
    register_names = tuple(name for name in ("low", "mid", "high") if name in midi3_registers)
    lines += [
        "",
        "Mean original-level MR-STFT by MIDI 3 sound-test pattern (descriptive; four triplets per pattern):",
        "",
        "| Representation | Melody/bass | Chords/poly | Repeating sequence |",
        "|---|---:|---:|---:|",
    ]
    for arm in ordered:
        cells = [midi3_patterns[name]["arms"][arm]["mean_original_level_mrstft"] for name in pattern_names]
        lines.append(f"| {_arm_label(arm)} | " + " | ".join(f"{value:.3f}" for value in cells) + " |")
    lines += [
        "",
        "Mean original-level MR-STFT by MIDI 3 target register (descriptive; triplet count varies):",
        "",
        "| Representation | " + " | ".join(f"{name} (n={midi3_registers[name]['episodes']}, {midi3_registers[name]['triplets']} triplets)" for name in register_names) + " |",
        "|---|" + "---:|" * len(register_names),
    ]
    for arm in ordered:
        cells = [midi3_registers[name]["arms"][arm]["mean_original_level_mrstft"] for name in register_names]
        lines.append(f"| {_arm_label(arm)} | " + " | ".join(f"{value:.3f}" for value in cells) + " |")
    lines += [
        (f"Selected role coverage: {coverage['selected_articulation_counts'].get('held', 0)} held and "
         f"{coverage['selected_articulation_counts'].get('phrased', 0)} phrased excerpts; register counts low/mid/high "
         f"{coverage['selected_register_counts'].get('low', 0)}/{coverage['selected_register_counts'].get('mid', 0)}/"
         f"{coverage['selected_register_counts'].get('high', 0)}; velocity-varied excerpts "
         f"{coverage['selected_velocity_varied_count']}/36; pitch spans of at least 24 semitones "
         f"{coverage['selected_wide_pitch_span_count']}/36. Unavailable pattern categories: {unavailable}."),
        "",
        "Equivalent means by MIDI 1 query and MIDI 2 retrieval pattern are in the sanitized aggregate. The subgroup means are descriptive: pattern strata have only the triplets shown above, and register groups are not independent of MIDI content.",
        "",
        "### Interpretation and limits",
        "",
        (f"The primary ordering is { _arm_label(best) } then { _arm_label(runner_up) }; the second arm's original-level "
         f"error is {_format_ci(arms[runner_up]['original_level_mrstft'])}. The paired difference for the leading arm "
         f"versus CIR is {_format_ci(top_delta)} (negative favors the leading arm). Exact-preset Recall@1 and Issue #157 "
         "preset-dominant fraction answer different questions: recognizing A across performances or moving more for a "
         "preset change does not guarantee that the nearest different preset sounds closest. The cross-arm table keeps "
         "those earlier measures beside the new acoustic result so disagreements remain visible."),
        "",
        "This is a decoder-free, training-free catalog-selection test. It does not show that any representation predicts Vital controls, builds a new patch, or produces a better reconstruction. No independent confirmation cohort was available or accessed; the selected 12 triplets are a fixed exploratory sample. No listening panel was run, so acoustic distances should not be described as human preference.",
        "",
        "**Recommendation.** Treat the lowest-error arm as a provisional catalog-search candidate only. Repeat with a separately preselected, permitted confirmation cohort and listening review before making a broader recommendation.",
        "",
        "### Execution and reproducibility",
        "",
        "The run reuses the accepted `issue152-validation-train-midi-v3-50` WAV and frozen embedding caches. The full cache audit verified all 11 representation arms, 129 × 50 common cells, render identities and accepted receipts, feature archive hashes, checkpoint/tap/pooling/preprocessing provenance, finite nonzero vectors, and the `validation` preset / `train` performance boundary. It found no missing required matrix cells. Selected MIDI 1/2/3 WAVs were then checked against receipt bytes and render metadata; no Vital rendering, feature extraction, or encoder inference was performed.",
        (f"The triplet matrix check verified {wav_coverage.get('selected_matrix_wavs_verified', 0):,} selected preset/MIDI WAV cells "
         f"and loaded {wav_coverage.get('midi3_scoring_wavs_loaded', 0):,} MIDI-3 score WAVs. Receipt QA warnings were retained "
         f"without waveform clipping or repair: {wav_coverage.get('qa_warning_counts', {})}."),
        "",
        "The original-level primary score uses sample-zero alignment and the full saved stereo 44.1 kHz float32 WAV, including the renderer tail, with no time shift, trim, or padding. Files with clipping warnings remain at their saved level and are counted in the private execution summary; silent or nonfinite waveforms fail closed. Gain matching applies one stereo-RMS scalar to the candidate only. The envelope measure is the existing 10 ms per-channel stereo RMS relative RMSE with a zero-padded final frame; level is global stereo RMS relative error.",
        "",
        "Triplet IDs, selected WAV paths, per-case selections and errors, and intermediate acoustic matrices remain under ignored `outputs/issue161-audible-substitute/`. The committed [sanitized aggregate results](issue161_substitution_results.json) contain no raw preset or MIDI identifiers.",
        "",
        "<!-- issue161-audible-substitute-results:end -->",
        "",
    ]
    return "\n".join(lines)


def _update_report(results: dict, report_path: Path) -> None:
    text = report_path.read_text(encoding="utf-8")
    section = _markdown_section(results)
    start, end = "<!-- issue161-audible-substitute-results:start -->", "<!-- issue161-audible-substitute-results:end -->"
    if start in text and end in text:
        before, rest = text.split(start, 1)
        _, after = rest.split(end, 1)
        text = before + after.lstrip("\r\n")
    if "## Results and recommendation" in text:
        before, after = text.split("## Results and recommendation", 1)
        updated = before + "## Results and recommendation\n\n" + section + after.lstrip("\r\n")
    elif "### How to read these tests" in text:
        updated = text.replace("### How to read these tests", section + "### How to read these tests", 1)
    else:
        updated = text.rstrip() + "\n\n" + section
    updated = updated.replace("**Primary expanded-cohort result:** CIR", "**Existing cross-MIDI retrieval baseline:** CIR", 1)
    report_path.write_text(updated, encoding="utf-8", newline="\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate three-MIDI audible-substitute retrieval from frozen caches")
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--checkpoint-root", type=Path, default=ROOT / "local_checkpoints")
    parser.add_argument("--output", type=Path, default=PRIVATE)
    parser.add_argument("--report", type=Path, default=REPORT)
    parser.add_argument("--bootstrap-replicates", type=int, default=2_000)
    parser.add_argument("--prepare-only", action="store_true", help="freeze triplets and verify source WAVs without scoring")
    parser.add_argument("--resume-scoring", action="store_true", help="reuse choices from a completed full cache audit and resume acoustic scoring")
    parser.add_argument("--device", choices=("cpu", "xpu"), default="xpu", help="device for shared STFT scoring")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args(argv)
    if args.seed != SEED or args.bootstrap_replicates < 1:
        parser.error("issue #161 pins seed 161 and requires positive bootstrap replicates")
    if args.prepare_only and args.resume_scoring:
        parser.error("--prepare-only and --resume-scoring are mutually exclusive")
    args.output.mkdir(parents=True, exist_ok=True)
    progress = lambda message: print(message, flush=True)
    summary = json.loads((args.dataset / "dataset_summary.private.json").read_text(encoding="utf-8"))
    if (summary.get("source_split") != "validation" or summary.get("performance_source_split") != "train" or
            summary.get("reserved_split_access") != "none" or summary.get("unique_midi_excerpts") != 50):
        raise ValueError("triplet selection source is not the authorized validation-preset/train-MIDI cohort")
    performances = [row for row in _jsonl(args.dataset / "performances.private.jsonl")
                    if row.get("performance_source_split") == "train" and row.get("split") == "validation"]
    if len(performances) != 50:
        raise ValueError("triplet selection requires all 50 permitted MIDI event records")
    plan = _private_plan(performances, args.seed)
    plan_path = args.output / "triplets.private.json"
    if plan_path.exists() and json.loads(plan_path.read_text(encoding="utf-8")) != plan:
        raise ValueError("saved MIDI triplet manifest differs from the reproducible event-only selection")
    _write_json(plan_path, plan)
    progress("Preselected 12 MIDI triplets from MIDI events alone, before cache metrics were loaded.")

    if args.resume_scoring:
        audit, render_plans, choices, resume_data = _resume_inputs(args.dataset, args.output, summary, plan)
        midi3_ids = {triplet["roles"][2]["midi_id"] for triplet in plan["triplets"]}
        audio_rows, current_wav_coverage = verify_triplet_wavs(args.dataset, audit, render_plans, plan,
                                                                midi_ids_override=midi3_ids)
        if current_wav_coverage["selected_matrix_wavs_verified"] != len(audit["preset_ids"]) * len(midi3_ids):
            raise ValueError("resume failed to reverify every selected MIDI-3 scoring WAV")
        wav_coverage = resume_data["prior_wav_coverage"]
        wav_coverage["midi3_scoring_wavs_loaded"] = 0
        wav_coverage["midi3_wavs_reverified_for_resume"] = current_wav_coverage["selected_matrix_wavs_verified"]
        progress("Reused the saved post-audit retrieval choices; reverified all selected MIDI-3 WAVs against receipts.")
    else:
        audit = verify_embedding_cache(args.dataset, args.checkpoint_root, progress=progress)
        if (audit["reserved_split_access"] != "none" or audit["common_cells"] != 6_450 or
                audit["evaluation_arms"] != 11 or audit["source_split"] != "validation" or
                audit["performance_source_split"] != "train"):
            raise ValueError("frozen cache audit does not match issue #161's authorized cohort")
        if not {item["midi_id"] for triplet in plan["triplets"] for item in triplet["roles"]} <= set(audit["midi_ids"]):
            raise ValueError("selected MIDI identities are absent from the audited embedding matrix")
        render_plans = _jsonl(args.dataset / "render_plan.private.jsonl")
        audio_rows, wav_coverage = verify_triplet_wavs(args.dataset, audit, render_plans, plan)
        _write_json(args.output / "wav_verification.private.json", wav_coverage)
        progress(f"Frozen {len(plan['triplets'])} MIDI triplets; verified {wav_coverage['selected_matrix_wavs_verified']:,} selected WAVs.")
    if args.prepare_only:
        print("Preparation complete; no retrieval or acoustic scoring was run.")
        return 0

    if not args.resume_scoring:
        choices = retrieve_alternatives(audit, plan)
        _write_json(args.output / "retrieval_choices.private.json", choices)
        progress("Completed all 11 MIDI 1-to-MIDI 2 retrieval arms; retrieval read only MIDI 1/2 embeddings.")
    preset_ids = audit["preset_ids"]
    raw_matrices = {}
    gain_matrices = {}
    references = {}
    secondary = {}
    acoustic_contract = None
    for triplet in plan["triplets"]:
        index = triplet["triplet_index"]
        midi3_id = triplet["roles"][2]["midi_id"]
        samples = load_scoring_midi(audio_rows, preset_ids, midi3_id)
        matrix, gain_matrix, acoustic_contract = raw_mrstft_matrix(samples, output_dir=args.output / "tmp", device_name=args.device,
            progress=lambda message: progress(f"Triplet {index + 1}/{len(plan['triplets'])}: {message}"))
        raw_matrices[index] = matrix
        gain_matrices[index] = gain_matrix
        reference_rows = challenger_reference_choices(preset_ids, audit["family_by_state"], {index: matrix},
                                                      [index], seed=args.seed)
        references.update(reference_rows)
        triplet_choices = {arm: [row for row in rows if row["triplet_index"] == index]
                           for arm, rows in choices.items()}
        triplet_secondary = score_secondary_metrics(triplet_choices, reference_rows, triplets=plan,
            preset_ids=preset_ids, raw_matrices={index: matrix}, gain_matrices={index: gain_matrix},
            audio_by_midi3={midi3_id: samples})
        secondary.update(triplet_secondary)
        wav_coverage["midi3_scoring_wavs_loaded"] += len(samples)
        del samples
        progress(f"Scored the shared MIDI 3 challenger matrix for triplet {index + 1}/{len(plan['triplets'])}.")
    np.savez_compressed(args.output / "acoustic_matrices.private.npz",
                        **{f"raw_triplet_{index:02d}": matrix for index, matrix in raw_matrices.items()},
                        **{f"gain_matched_triplet_{index:02d}": matrix for index, matrix in gain_matrices.items()})
    sensitivity_doc = json.loads((ROOT / "reports" / "issue157_sensitivity_results.json").read_text(encoding="utf-8"))
    sensitivity = {arm: {"preset_dominant_fraction": sensitivity_doc["arms"][arm]["preset_dominant_fraction"]}
                   for arm in choices}
    existing_eval = resume_data["evaluation"] if args.resume_scoring else json.loads((args.dataset / "evaluation.private.json").read_text(encoding="utf-8"))
    existing_recall = {arm: existing_eval["model_results"][arm]["cross_midi_common_gallery"]["recall_at_1"]
                       for arm in choices}
    results, private_cases = build_results(choices, raw_matrices, secondary, plan, existing_recall, sensitivity,
                                           seed=args.seed, bootstrap_replicates=args.bootstrap_replicates)
    results["acoustic_contract"] = acoustic_contract
    results["selected_wav_coverage"] = wav_coverage
    _write_json(args.output / "case_results.private.json", private_cases)
    _write_json(args.output / "aggregate_results.private.json", results)
    _write_json(args.output / "wav_verification.private.json", wav_coverage)
    sanitized = {key: value for key, value in results.items() if key != "selected_wav_coverage"}
    _write_json(PUBLIC_RESULTS, sanitized)
    _update_report(results, args.report)
    print(f"Saved sanitized aggregates to {PUBLIC_RESULTS} and updated {args.report}.")
    print(f"Top measured arm: {_arm_label(min(results['arms'], key=lambda arm: results['arms'][arm]['original_level_mrstft']['mean']))}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
