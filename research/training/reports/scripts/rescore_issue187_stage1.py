"""Reconstruct missing frozen validation audio and score the Issue 187 codecs."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.io import wavfile
import torch
from torch.nn import functional as F

import run_issue187_surrogates as experiment
from obruxo_training.reconstruction_metrics import VisqolAudio, waveform_scores


BASE = experiment.baseline
DAC = experiment.snapshot("descript/dac_44khz", "c1bc521685adf9cfe247bc39a5ca58917eda1ac4")
CHECKPOINTS = {**experiment.CHECKPOINTS, "same-l-embedded": experiment.GENERATORS["same-l"], "dac": DAC}


class DacCodec:
    """Use the original mono checkpoint independently for left and right channels."""

    def __init__(self):
        from transformers import DacModel

        self.model = DacModel.from_pretrained(DAC, local_files_only=True).eval().requires_grad_(False).to(experiment.DEVICE)
        self.hop = self.model.config.hop_length

    def encode(self, audio, _seed):
        audio = F.pad(audio, (0, -audio.shape[-1] % self.hop)).transpose(0, 1)
        return self.model.encode(audio).quantized_representation

    def decode(self, latent):
        return self.model.decode(latent).audio_values[None]


def validation_panel(old_run):
    source = experiment.PROXY_SOURCE
    plan = experiment.read_json(source / "plan.json")["plan"]
    rows = plan["evaluation"]["validation_proxy"]["pair_recipes"]
    if (len(rows), len({r["state_id"] for r in rows}), len({r["midi_id"] for r in rows})) != (128, 64, 16):
        raise ValueError("unexpected frozen validation panel")
    ledger = experiment.receipts(source)
    for index, row in enumerate(rows):
        for offset, (role, _) in enumerate(experiment.ROLES):
            label = f"{row['pair_id']}-{role}"
            path = old_run / "validation-audio" / f"{label}.npy"
            audio = np.load(path, allow_pickle=False)
            receipt = ledger[f"{row['pair_id']}:{role}"]
            experiment.require_audio(audio, receipt, label)
            yield row, role, label, path, experiment.SEED + 2 * index + offset, receipt["qa"]["audio_float32_sha256"]


def read_reconstruction(path, reference):
    rate, wave = wavfile.read(path)
    if rate != BASE.SAMPLE_RATE or wave.dtype != np.float32 or wave.shape != reference.shape:
        raise ValueError(f"unexpected cached waveform format: {path}")
    return wave


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@torch.no_grad()
def reconstruct(panel, old_run, output, name, progress):
    cached = output / "reconstructions" / name
    cached.mkdir(parents=True, exist_ok=True)
    checkpoint = CHECKPOINTS[name]
    legacy_path = old_run / f"reconstruction-{name}.json"
    legacy = experiment.read_json(legacy_path) if name != "dac" else None
    if legacy is not None and legacy["summary"]["revision"] != checkpoint.name:
        raise ValueError(f"{name}: prior reconstruction uses a different checkpoint")
    previous_cases = {(r["pair_id"], r["clip_role"]): r for r in legacy["cases"]} if legacy else {}
    if legacy is not None and set(previous_cases) != {(r["pair_id"], role) for r, role, *_ in panel}:
        raise ValueError(f"{name}: prior reconstruction uses a different panel")
    codec = None
    manifest = []
    controls = BASE.SurrogateCodec(BASE.PilotCodec(), expanded=True) if name == "dac" else None
    pairs = experiment.PROXY_SOURCE / "validation-pairs"
    loader = BASE._pair_loader(pairs) if controls is not None else None
    progress.set(f"Stage 1 · Reconstruct with {name}", len(panel), 1)
    for index, (row, role, label, original_path, seed, original_hash) in enumerate(panel, 1):
        original = np.load(original_path, allow_pickle=False)
        listening = old_run / "listening" / f"{label}-{name}.wav"
        wave_path = cached / f"{label}.wav"
        receipt_path = wave_path.with_suffix(".json")
        if legacy is not None and listening.is_file():
            wave = read_reconstruction(listening, original)
            metrics, origin = previous_cases[(row["pair_id"], role)], "prior_listening"
            wave_path = listening
        elif wave_path.is_file() and receipt_path.is_file():
            saved = experiment.read_json(receipt_path)
            if (saved["revision"], saved["seed"], saved["original_sha256"], saved["wav_sha256"]) != (
                checkpoint.name, seed, original_hash, digest(wave_path),
            ):
                raise ValueError(f"{label}: rescore cache receipt mismatch")
            wave = read_reconstruction(wave_path, original)
            metrics, origin = saved["legacy"], "rescore_cache"
        else:
            if codec is None:
                codec = DacCodec() if name == "dac" else experiment.Codec(name, checkpoint)
            audio = experiment.samples(original)
            torch.xpu.synchronize()
            started = perf_counter()
            latent = codec.encode(audio, seed)
            torch.xpu.synchronize()
            encoded = perf_counter()
            predicted = codec.decode(latent)[..., :len(original)]
            torch.xpu.synchronize()
            decoded = perf_counter()
            wave = predicted[0].T.float().cpu().numpy()
            if wave.shape != original.shape or not np.isfinite(wave).all():
                raise ValueError(f"{name}/{label}: invalid reconstruction")
            metrics = previous_cases.get((row["pair_id"], role))
            if controls is not None:
                batch, metas = BASE.collate_pairs([row], pairs, controls, experiment.DEVICE, loader)
                i = int(role == "perturbed")
                metrics = experiment.measured_case(predicted, original, experiment.split_batch(batch, i), metas[i])
                metrics.update(encode_seconds=encoded - started, decode_seconds=decoded - encoded,
                               latent_shape=list(latent.shape), latent_bytes=latent.nbytes)
            temporary = wave_path.with_suffix(".tmp")
            wavfile.write(temporary, BASE.SAMPLE_RATE, wave)
            temporary.replace(wave_path)
            BASE.write_json(receipt_path, {"revision": checkpoint.name, "seed": seed, "original_sha256": original_hash,
                                           "wav_sha256": digest(wave_path), "legacy": metrics})
            origin = "encoded"
        manifest.append({"case": label, "pair_id": row["pair_id"], "clip_role": role, "state_id": row["state_id"],
                         "midi_id": row["midi_id"], "original_path": str(original_path), "original_sha256": original_hash,
                         "wave_path": str(wave_path), "origin": origin, "legacy": metrics})
        progress.update(index)
    return manifest


def score(manifest, name, output, visqol):
    cases = []
    for row in manifest:
        original = np.load(row["original_path"], allow_pickle=False)
        wave = read_reconstruction(Path(row["wave_path"]), original)
        values = waveform_scores(original, wave)
        values["visqol_audio"] = visqol(original, wave) if visqol else None
        cases.append({k: v for k, v in row.items() if not k.endswith("_path")} | values)
    BASE.write_json(output / f"rescore-{name}.json", {"revision": CHECKPOINTS[name].name, "cases": cases})
    print(f"{name}: completed {len(manifest)} waveform scores", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prior-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--models", choices=CHECKPOINTS, nargs="+", default=list(CHECKPOINTS))
    parser.add_argument("--visqol-model", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(4)
    old_run, output = args.prior_run.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    panel = list(validation_panel(old_run))
    visqol = VisqolAudio(args.visqol_model) if args.visqol_model else None
    progress = experiment.Progress(output)
    progress.stage_count = 1
    try:
        with ThreadPoolExecutor(max_workers=1) as scorer:
            scoring = []
            for name in args.models:
                manifest = reconstruct(panel, old_run, output, name, progress)
                BASE.write_json(output / f"waveforms-{name}.json", manifest)
                torch.xpu.empty_cache()
                scoring.append(scorer.submit(score, manifest, name, output, visqol))
                for completed in scoring:
                    if completed.done():
                        completed.result()
            progress.set("Stage 1 · Finish CPU scoring", len(scoring), 1)
            for index, completed in enumerate(scoring, 1):
                completed.result()
                progress.update(index)
        progress.update(progress.total, state="complete")
    except Exception as exc:
        progress.failures += 1
        progress.update(progress.done, state="failed", log=True, message=str(exc))
        raise


if __name__ == "__main__":
    main()
