"""Publish paired Stage 1 reconstruction scores and prepare level-consistent examples."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.stats import spearmanr


MODELS = ("open-small", "same-s", "same-l", "same-l-embedded", "dac")
METRICS = ("mel_log1p", "stft_log1p", "visqol_audio", "cir_cosine", "si_sdr_db", "ccpc",
           "mrstft", "replay_mrstft", "spectral_loss", "spectral_convergence", "log_magnitude_mae",
           "octave_band_energy_error_db", "monophonic_octave_energy_error_db",
           "attack_envelope_error_db", "decay_envelope_error_db", "sustain_envelope_error_db",
           "release_envelope_error_db", "rms_relative", "side_to_mid_error_db",
           "encode_seconds", "decode_seconds", "latent_bytes")
DIRECTIONS = {"visqol_audio": 1, "cir_cosine": 1, "si_sdr_db": 1, "ccpc": 1}


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_model(run, name):
    model = load(run / f"rescore-{name}.json")
    cir_path = run / f"cir-{name}.json"
    cir = load(cir_path)["cases"] if cir_path.is_file() else {}
    replay_path = run / f"replay-mrstft-{name}.json"
    replay = load(replay_path) if replay_path.is_file() else {}
    rows = {r["case"]: r for r in model["cases"]}
    if len(rows) != 256 or (cir and set(cir) != set(rows)):
        raise ValueError(f"{name}: unexpected CIR or waveform panel")
    for case, row in rows.items():
        legacy = row.pop("legacy")
        row.update({key: legacy[key] for key in METRICS if key in legacy})
        row["cir_cosine"] = cir.get(case)
        row["replay_mrstft"] = replay.get(case)
    return model["revision"], rows


def interval(rows, key, reference=None):
    groups = {}
    for case, row in rows.items():
        value = row.get(key)
        other = reference[case].get(key) if reference is not None else 0
        if value is not None and other is not None:
            groups.setdefault(row["state_id"], []).append(value - other)
    if not groups:
        return None
    sums = np.asarray([np.sum(group) for group in groups.values()])
    counts = np.asarray([len(group) for group in groups.values()])
    indices = np.random.default_rng(187).integers(len(groups), size=(5000, len(groups)))
    means = sums[indices].sum(axis=1) / counts[indices].sum(axis=1)
    return {"mean": float(sums.sum() / counts.sum()), "valid": int(counts.sum()), "roots": len(groups),
            "lower_95": float(np.quantile(means, .025)), "upper_95": float(np.quantile(means, .975))}


def listening_examples(run, output, models):
    import soundfile as sf

    left, right = models["open-small"], models["same-l"]
    paired = [(case, left[case]["mel_log1p"] - row["mel_log1p"],
               left[case]["mrstft"] - row["mrstft"]) for case, row in right.items()]
    split = sorted((item for item in paired if item[1] > 0 and item[2] < 0), key=lambda item: item[1])
    candidates = {"mel_favors_same_l": max(paired, key=lambda item: item[1]),
                  "mel_favors_open_small": min(paired, key=lambda item: item[1])}
    if split:
        candidates["mel_and_mrstft_disagree"] = split[len(split) // 2]
    manifests = {name: {r["case"]: r for r in load(run / f"waveforms-{name}.json")}
                 for name in ("open-small", "same-l", "same-l-embedded")}
    output.mkdir(parents=True, exist_ok=True)
    examples = {}
    for title, (case, _, _) in candidates.items():
        waves = {name: wavfile.read(rows[case]["wave_path"])[1] for name, rows in manifests.items()}
        waves["reference"] = np.load(manifests["open-small"][case]["original_path"], allow_pickle=False)
        peak = max(float(np.abs(wave).max()) for wave in waves.values())
        gain = min(1, .98 / max(peak, 1e-16))
        files = {}
        for name, wave in waves.items():
            path = output / f"{title}-{name}.flac"
            sf.write(path, wave * gain, 44100, subtype="PCM_24")
            files[name] = path.name
        examples[title] = {"case": case, "shared_playback_gain": gain, "files": files}
    return examples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--models", nargs="+", choices=MODELS, default=MODELS)
    args = parser.parse_args()
    revisions, models = {}, {}
    for name in args.models:
        revisions[name], models[name] = load_model(args.run, name)
    reference = models["open-small"]
    panel_key = lambda rows: {case: (row["state_id"], row["midi_id"], row["original_sha256"])
                              for case, row in rows.items()}
    if any(panel_key(rows) != panel_key(reference) for rows in models.values()):
        raise ValueError("candidate rows do not share an identical validation panel")
    keys = ("mel_log1p", "stft_log1p", "visqol_audio", "cir_cosine", "si_sdr_db", "ccpc", "mrstft",
            "replay_mrstft", "spectral_loss", "octave_band_energy_error_db", "rms_relative")
    summaries = {name: {key: interval(rows, key) for key in METRICS} for name, rows in models.items()}
    differences = {name: {key: interval(rows, key, reference) for key in keys}
                   for name, rows in models.items() if name != "open-small"}
    agreement = {}
    same_l = models["same-l"]
    for key in keys:
        pairs = [(reference[case]["mel_log1p"] - row["mel_log1p"],
                  DIRECTIONS.get(key, -1) * (row[key] - reference[case][key]))
                 for case, row in same_l.items() if row.get(key) is not None and reference[case].get(key) is not None]
        if not pairs:
            continue
        primary, alternative = np.asarray(pairs).T
        rho = spearmanr(primary, alternative).statistic
        agreement[key] = {"spearman_rho": float(rho) if np.isfinite(rho) else None,
                          "same_winner": int(np.sum(primary * alternative > 0)),
                          "opposite_winner": int(np.sum(primary * alternative < 0)), "valid": len(pairs)}
    fields = ("codec", "case", "state_id", "midi_id", "original_sha256", "origin", *METRICS)
    with (args.report_dir / "STAGE1_RESCORE_CASES.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows({"codec": name, **row} for name, rows in models.items() for row in rows.values())
    result = {
        "codec_revisions": revisions,
        "panel": {"clips": len(reference), "patch_roots": len({r["state_id"] for r in reference.values()}),
                  "midi_performances": len({r["midi_id"] for r in reference.values()})},
        "interval_method": "5,000 paired patch-root cluster bootstrap replicates, seed 187",
        "means_and_intervals": summaries, "paired_differences_from_open_small": differences,
        "same_l_vs_open_small_metric_agreement": agreement,
        "listening": listening_examples(args.run, args.report_dir / "listening", models),
    }
    (args.report_dir / "STAGE1_RESCORE.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n",
                                                        encoding="utf-8")


if __name__ == "__main__":
    main()
