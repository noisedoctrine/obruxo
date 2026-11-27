"""Run the gated Issue 187 experiment using the existing Vital cohorts."""

# ruff: noqa: E402
import argparse
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
import json
import math
import os
from pathlib import Path
import sys
from time import perf_counter
from types import SimpleNamespace

import numpy as np
from scipy.io import wavfile
import torch
from torch.nn import functional as F

import run_issue183_flow_matching as baseline
from check_issue187_pretrained_xpu import load_codec

sys.path.insert(0, str(baseline.RESEARCH / "modelling/frozen_encoder_retrieval"))

from frozen_retrieval.core import _rank_metrics, mixed_gallery_midi_id
from frozen_retrieval.dataset import validate_dataset_manifests
from frozen_retrieval.rendering import accepted_receipts
from obruxo_data.midi import Performance
from obruxo_data.render.qa import audio_float32_sha256
from obruxo_training.evaluate import audio_distance
from obruxo_training.surrogate_pilot import Progress, native_session


DEVICE = "xpu:0"
SEED = 183
PREFETCH_WORKERS = 4
PREFETCH_DEPTH = 8
ROLES = (("original", "target"), ("perturbed", "changed"))
HUB = Path(os.environ.get("HF_HUB_CACHE", Path.home() / "HuggingFace/hub"))
PROXY_SOURCE = baseline.VALIDATION_SOURCE
RETRIEVAL_SOURCE = (baseline.MAIN_ROOT / "research/modelling/frozen_encoder_retrieval/outputs/"
                    "issue167-v7/dataset")
RETRIEVAL_BASELINE = baseline.RESEARCH / "modelling/frozen_encoder_retrieval/reports/issue167_v7_results.json"


def snapshot(repo, revision):
    return HUB / f"models--{repo.replace('/', '--')}" / "snapshots" / revision


CHECKPOINTS = {
    "open-small": snapshot("stabilityai/stable-audio-open-small", "ea2772170fc06f80dfd02bea6e57a44d9d4bc9d3"),
    "same-s": snapshot("stabilityai/SAME-S", "63a63b7193ee9b6ee9836776c5f3c52a2f9600df"),
    "same-l": snapshot("stabilityai/SAME-L", "54b0fe9c9fc27aa1d5eac36402788efea15f73eb"),
}
GENERATORS = {
    "same-s": snapshot("stabilityai/stable-audio-3-small-music", "8bdfb53c6b0fbc62113cd571fd37373efe15fa14"),
    "same-l": snapshot("stabilityai/stable-audio-3-medium", "d4d3579395a8ac0df4ab13b5b1d7138ce7290f81"),
}


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_rows(path):
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def receipts(source):
    return {row["label"]: row for row in read_rows(source / "renders.jsonl")
            if row["event"] == "outcome" and row.get("qa", {}).get("audio_float32_sha256")}


def require_audio(audio, recorded, label):
    digest = audio_float32_sha256(audio)
    if digest != recorded["qa"]["audio_float32_sha256"]:
        raise ValueError(f"{label}: waveform differs from the original native render receipt")


def pair_audio(rows, directory, render, ledger):
    jobs = [(f"{row['pair_id']}:{role}", baseline.PilotTarget.from_dict(row[field]),
             Performance.from_dict(row["performance"])) for row in rows for role, field in ROLES]
    if directory is not None and all((directory / f"{label.replace(':', '-')}.npy").is_file()
                                     for label, _, _ in jobs):
        audio = [np.load(directory / f"{label.replace(':', '-')}.npy", allow_pickle=False)
                 for label, _, _ in jobs]
    else:
        audio = []
        for job, (wave, receipt) in zip(jobs, render(jobs), strict=True):
            if wave is None or receipt["provenance"] != ledger[job[0]]["provenance"]:
                raise ValueError(f"{job[0]}: native render or provenance differs from the frozen run")
            audio.append(wave)
    for (label, _, _), wave in zip(jobs, audio, strict=True):
        require_audio(wave, ledger[label], label)
        if directory is not None:
            path = directory / f"{label.replace(':', '-')}.npy"
            if not path.exists():
                temporary = path.with_suffix(".tmp")
                with temporary.open("wb") as stream:
                    np.save(stream, wave, allow_pickle=False)
                temporary.replace(path)
    return audio


class Codec:
    def __init__(self, name, checkpoint=None):
        self.name = name
        self.checkpoint = CHECKPOINTS[name] if checkpoint is None else checkpoint
        self.model = load_codec("open-small-codec" if name == "open-small" else "same-codec",
                                self.checkpoint, DEVICE)
        self.hop = self.model.downsampling_ratio

    def encode(self, audio, seed=SEED):
        audio = F.pad(audio, (0, -audio.shape[-1] % self.hop))
        if self.name == "open-small":
            torch.manual_seed(seed)
        return self.model.encode_audio(audio)

    def decode(self, latent):
        return self.model.decode_audio(latent)


def samples(wave):
    return torch.as_tensor(wave.T.copy(), dtype=torch.float32, device=DEVICE)[None]


def frame_spectra(audio, times, *, size=2048):
    window = torch.hann_window(size, device=audio.device)
    batch, channels, length = audio.shape
    spectra = torch.stft(audio.reshape(-1, length), size, hop_length=size // 4, window=window,
                         pad_mode="constant", return_complex=True).abs().reshape(batch, channels, size // 2 + 1, -1)
    indices = (times * (baseline.SAMPLE_RATE / (size // 4))).round().long().clamp(0, spectra.shape[-1] - 1)
    return spectra.gather(-1, indices[:, None, None].expand(-1, channels, size // 2 + 1, -1)) / math.sqrt(size)


def measured_case(predicted, original, batch, meta):
    spectra = frame_spectra(predicted, batch["times"])
    values = spectra.permute(0, 3, 1, 2).reshape(len(spectra), spectra.shape[-1], -1)
    row = baseline.score_batch(values, batch, [meta])[0]
    wave = predicted[0].T.detach().float().cpu().numpy()
    row.update(audio_distance(original, wave))
    rms = lambda x: float(np.sqrt(np.mean(x ** 2)))
    row["side_to_mid_error_db"] = abs(20 * math.log10(
        (rms(wave[:, 0] - wave[:, 1]) + 1e-8) / (rms(wave[:, 0] + wave[:, 1]) + 1e-8)
        * (rms(original[:, 0] + original[:, 1]) + 1e-8) / (rms(original[:, 0] - original[:, 1]) + 1e-8)))
    return row


def split_batch(batch, index):
    return {key: value[index:index + 1] for key, value in batch.items()}


def validation(output, rows, codec, progress):
    waves = output / "validation-audio"
    waves.mkdir(exist_ok=True)
    ledger = receipts(PROXY_SOURCE)
    progress.set("Stage 1 · Recover frozen validation waveforms", len(rows), 1)
    with native_session(output, codec, 4) as render:
        for index in range(0, len(rows), 2):
            pair_audio(rows[index:index + 2], waves, render, ledger)
            progress.update(min(index + 2, len(rows)))
    return waves


@torch.no_grad()
def reconstruct(output, name, codec, rows, controls, waves, progress):
    phase_path = output / f"reconstruction-{name}.json"
    if phase_path.exists():
        return read_json(phase_path)["summary"]
    progress.set(f"Stage 1 · Reconstruct with {name}", 2 * len(rows), 1)
    data = PROXY_SOURCE / "validation-pairs"
    loader = baseline._pair_loader(data)
    cases = []
    for index, row in enumerate(rows):
        batch, metas = baseline.collate_pairs([row], data, controls, DEVICE, loader)
        for role_index, ((role, _), meta) in enumerate(zip(ROLES, metas, strict=True)):
            original = np.load(waves / f"{row['pair_id']}-{role}.npy", allow_pickle=False)
            audio = samples(original)
            torch.xpu.synchronize()
            started = perf_counter()
            latent = codec.encode(audio, SEED + 2 * index + role_index)
            torch.xpu.synchronize()
            encoded = perf_counter()
            predicted = codec.decode(latent)[..., :len(original)]
            torch.xpu.synchronize()
            decoded = perf_counter()
            result = measured_case(predicted, original, split_batch(batch, role_index), meta)
            result.update(encode_seconds=encoded - started, decode_seconds=decoded - encoded,
                          latent_shape=list(latent.shape[1:]), latent_bytes=latent.nbytes)
            if index in (0, 32, 64, 96):
                listening = output / "listening"
                listening.mkdir(exist_ok=True)
                wavfile.write(listening / f"{row['pair_id']}-{role}-{name}.wav", baseline.SAMPLE_RATE,
                              predicted[0].T.float().cpu().numpy())
            cases.append(result)
        progress.update(len(cases))
    summary = baseline.aggregate_cases(cases)
    summary.update({key: float(np.mean([row[key] for row in cases])) for key in
                    ("mrstft", "rms_relative", "side_to_mid_error_db", "encode_seconds", "decode_seconds", "latent_bytes")})
    with torch.enable_grad():
        latent = latent.detach().requires_grad_(True)
        codec.decode(latent).float().square().mean().backward()
    summary["decoder_gradient_norm"] = float(latent.grad.norm()) if latent.grad is not None else 0.
    summary["revision"] = codec.checkpoint.name
    baseline.write_json(phase_path, {"summary": summary, "cases": cases})
    return summary


def retrieval_clip(dataset, receipt):
    path = dataset / receipt["wav_relative_path"]
    if baseline.sha256_file(path) != receipt["audio_sha256"]:
        raise ValueError("retrieval WAV differs from its QA receipt")
    rate, audio = wavfile.read(path)
    if rate != baseline.SAMPLE_RATE or audio.ndim != 2 or audio.shape[1] != 2:
        raise ValueError("retrieval requires native 44.1 kHz stereo WAVs")
    if np.issubdtype(audio.dtype, np.integer):
        audio = audio.astype(np.float32) / (-np.iinfo(audio.dtype).min)
    return audio


def prefetch_audio(requests, load_audio, *, workers=PREFETCH_WORKERS, depth=PREFETCH_DEPTH):
    requests = iter(requests)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = deque()
        exhausted = False

        def fill():
            nonlocal exhausted
            while not exhausted and len(pending) < depth:
                try:
                    request = next(requests)
                except StopIteration:
                    exhausted = True
                    break
                pending.append((request, pool.submit(load_audio, request)))

        fill()
        while pending:
            request, future = pending.popleft()
            yield request, future.result()
            fill()


@torch.no_grad()
def retrieve(output, name, codec, progress):
    result_path = output / f"retrieval-{name}.json"
    if result_path.exists():
        return read_json(result_path)["summary"]
    dataset = RETRIEVAL_SOURCE
    validate_dataset_manifests(dataset, read_json(dataset / "dataset_summary.private.json"))
    receipts_by_id = accepted_receipts(dataset)
    pair_map = {(row["state_id"], row["midi_id"]): row["request_id"]
                for row in read_rows(dataset / "render_plan.private.jsonl")
                if row["source_kind"] == "newly_rendered_original_preset"}
    candidate_data = read_json(dataset / "candidate_sets.render_qualified.private.json")
    candidates = candidate_data["candidate_sets"]
    episodes = list(read_rows(dataset / "episodes_qa_qualified.private.jsonl"))
    gallery_lookup = {}
    needed = set()
    for episode in episodes:
        group = candidates[episode["candidate_set_id"]]
        query = pair_map[episode["target_state_id"], episode["query_midi_id"]]
        gallery = [pair_map[state, episode["gallery_midi_id"] if episode["condition"] == "cross_midi_common_gallery"
                            else mixed_gallery_midi_id(episode, state, candidates)] for state in group["state_ids"]]
        gallery_lookup[episode["episode_id"]] = query, gallery
        needed.update((query, *gallery))
    if len(needed) != 6450 or len(episodes) != 12900 or not needed.issubset(receipts_by_id):
        raise ValueError("retrieval must use the Issue 167 panel of 6,450 recordings and 12,900 episodes")
    progress.set(f"Stage 1 · Cross-MIDI features with {name}", len(needed), 1)
    cache = output / f"retrieval-features-{name}"
    cache.mkdir(exist_ok=True)
    vectors = {}
    ordered = sorted(needed)
    audio_rows = None
    for index, request in enumerate(ordered, 1):
        path = cache / f"{request}.npy"
        if path.exists():
            vectors[request] = np.load(path, allow_pickle=False)
        else:
            if audio_rows is None:
                missing = (item for item in ordered if not (cache / f"{item}.npy").exists())
                audio_rows = iter(prefetch_audio(
                    missing, lambda item: retrieval_clip(dataset, receipts_by_id[item]),
                ))
            prefetched_request, audio = next(audio_rows)
            if prefetched_request != request:
                raise RuntimeError("prefetched retrieval audio lost request order")
            latent = codec.encode(samples(audio), int(request[:8], 16))
            vector = torch.cat((latent.mean(-1), latent.std(-1, unbiased=False)), -1)[0]
            vectors[request] = F.normalize(vector, dim=0).cpu().numpy()
            temporary = path.with_suffix(".tmp")
            with temporary.open("wb") as stream:
                np.save(stream, vectors[request], allow_pickle=False)
            temporary.replace(path)
        progress.update(index)
    cases, blocks = [], defaultdict(list)
    for episode in episodes:
        query, gallery = gallery_lookup[episode["episode_id"]]
        blocks[tuple(gallery)].append((episode, query))
    for gallery, episodes_in_block in blocks.items():
        scores = np.stack([vectors[query] for _, query in episodes_in_block]) @ np.stack(
            [vectors[request] for request in gallery]).T
        for (episode, _), similarities in zip(episodes_in_block, scores, strict=True):
            states = candidates[episode["candidate_set_id"]]["state_ids"]
            rank = _rank_metrics(list(zip(states, similarities.tolist(), strict=True)), episode["target_state_id"],
                                 ks=(1, 5, 10))
            cases.append({"episode_id": episode["episode_id"], "condition": episode["condition"], **rank})
    summary = {}
    for condition in {row["condition"] for row in cases}:
        group = [row for row in cases if row["condition"] == condition]
        summary[condition] = {"episodes": len(group), **{key: float(np.mean([row[key] for row in group]))
                               for key in ("recall_at_1", "recall_at_5", "recall_at_10", "mrr")}}
    baseline.write_json(result_path, {"summary": summary, "cases": cases, "latent_pool": "mean and std over time"})
    return summary


def run(output, *, selection=None, prior_run=None, recovered=None, midi_checkpoint=None, smoke=False,
        qualification=None, target_step=500, inversion_decision=None):
    torch.set_num_threads(4)
    output.mkdir(parents=True, exist_ok=True)
    progress = Progress(output)
    progress.stage_count = 3
    progress.set("Stage 1 · Verify input receipts", 1, 1)
    try:
        def receipt_progress(*, completed=0, total=None, **_):
            if total is not None:
                progress.total = total
            progress.update(completed)

        source, plan, train_rows, rows, pair_receipts = baseline.load_sources(
            baseline.TRAINING_SOURCE, PROXY_SOURCE, output, SimpleNamespace(update=receipt_progress))
        baseline.write_json(output / "input-receipts.json", pair_receipts)
        controls = baseline.SurrogateCodec(baseline.PilotCodec(), expanded=True)
        if selection is not None:
            from issue187_surrogate_adapters import run_later_stages

            waves = prior_run / "validation-audio"
            state = run_later_stages(output, read_json(selection), plan, train_rows, rows, controls, waves,
                                     progress, recovered=recovered, prior_run=prior_run,
                                     midi_checkpoint=midi_checkpoint, smoke=smoke, qualification=qualification,
                                     target_step=target_step, inversion_decision=inversion_decision)
            if state != "blocked":
                progress.update(progress.done, state=state)
            return
        waves = validation(output, rows, controls, progress)
        models = {}
        for name in CHECKPOINTS:
            codec = Codec(name)
            models[name] = {"reconstruction": reconstruct(output, name, codec, rows, controls, waves, progress),
                            "retrieval": retrieve(output, name, codec, progress)}
            del codec
            torch.xpu.empty_cache()
        retrieval_baseline = read_json(RETRIEVAL_BASELINE)["retrieval"]["metrics"]
        baseline.write_json(output / "stage1.json", {"models": models, "retrieval_baseline": retrieval_baseline})
        path = output / "decisions.json"
        decision = read_json(path) if path.is_file() else {}
        reason = "Publish the revised Stage 1 comparison and pass its recorded codec selection explicitly"
        decision.update(stage2="awaiting_stage1_selection", stage3="not_started", reason=reason)
        baseline.write_json(path, decision)
        progress.update(progress.done, state="blocked", message=reason)
    except Exception as exc:
        progress.failures += 1
        progress.update(progress.done, state="failed", log=True, message=str(exc))
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--selection", type=Path, help="published Stage 1 rescore codec selection")
    parser.add_argument("--prior-run", type=Path, help="previous run with receipt-verified validation audio")
    parser.add_argument("--recovered", type=Path, help="exact receipt-matched training audio recovered from original workers")
    parser.add_argument("--midi-checkpoint", type=Path, help="extracted frozen SpanSynth-Edit MIDI encoder")
    parser.add_argument("--smoke", action="store_true", help="one disposable optimizer update and full validation")
    parser.add_argument("--qualification", type=Path, help="passing selected-model qualification decisions")
    parser.add_argument("--target-step", type=int, choices=(500, 8000), default=500)
    parser.add_argument("--inversion-decision", type=Path, help="reviewed Stage 2 selection to take to Stage 3")
    args = parser.parse_args()
    if args.selection and not all((args.prior_run, args.recovered, args.midi_checkpoint)):
        parser.error("--selection requires --prior-run, --recovered, and --midi-checkpoint")
    if args.smoke and args.selection is None:
        parser.error("--smoke requires a published --selection")
    if args.selection and not args.smoke and args.qualification is None:
        parser.error("formal execution requires --qualification with the smoke decision record")
    run(args.output.resolve(), selection=args.selection, prior_run=args.prior_run, recovered=args.recovered,
        midi_checkpoint=args.midi_checkpoint, smoke=args.smoke, qualification=args.qualification,
        target_step=args.target_step, inversion_decision=args.inversion_decision)
