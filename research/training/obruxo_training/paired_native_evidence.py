"""Retain auditable comparison audio and account for every native attempt."""

import numpy as np
import torch
from itertools import islice

from obruxo_data.hashing import sha256_bytes
from obruxo_data.render.qa import audio_float32_sha256

from .checkpoints import atomic_torch_save
from .evaluate import audio_distances
from .batching import BatchConfig
from .spectral import SpectralTargets, spectral_loss


def native_ledger(events, renderer_identity, *, known_process_losses=()):
    attempts, outcomes = {}, {}
    for row in events:
        key = row["phase"], row["attempt"]
        destination = attempts if row["event"] == "attempt" else outcomes
        if row["event"] not in ("attempt", "outcome") or key in destination:
            raise ValueError("native ledger contains an unknown or duplicate event")
        destination[key] = row
    if attempts.keys() != outcomes.keys():
        raise ValueError("native ledger has an unobserved result; preserve the interruption")
    fields = ("phase", "label", "request_id", "source_sha256", "controlled_sha256")
    losses = {(row["phase"], row["attempt"]): row for row in known_process_losses}
    if len(losses) != len(known_process_losses):
        raise ValueError("duplicate declared process loss")
    recovered, used_losses = [], set()
    for phase in {key[0] for key in attempts}:
        if sorted(key[1] for key in attempts if key[0] == phase) != list(range(sum(key[0] == phase for key in attempts))):
            raise ValueError("native attempt counters changed")
    for key, receipt in outcomes.items():
        if any(receipt.get(name) != attempts[key].get(name) for name in fields):
            raise ValueError("native receipt differs from its charged request")
        if receipt["status"] == "accepted":
            if receipt["renderer_identity"] != renderer_identity:
                raise ValueError("native renderer identity changed")
            continue
        if key in losses:
            declared = losses[key]
            if (key[0] != "qualification" or receipt["status"] != "unobserved" or receipt.get("error_type") != "ProcessLostBeforeReceipt"
                    or "qa" in receipt or receipt.get("observation_evidence_sha256") != declared["observation_evidence_sha256"]
                    or any(receipt.get(name) != declared.get(name) for name in fields)):
                raise ValueError("process loss differs from its frozen observation evidence")
            matches = [later for later in outcomes.values() if later["status"] == "accepted" and later["phase"] == key[0]
                       and later["attempt"] > key[1] and all(later.get(name) == receipt.get(name) for name in fields)]
            if not matches:
                raise ValueError("unobserved process-loss request has no later accepted identical request")
            recovered.append({"kind": "process_loss", "phase": key[0], "unobserved_attempt": key[1],
                              "accepted_attempt": matches[0]["attempt"], "request_id": receipt["request_id"]})
            used_losses.add(key)
            continue
        if receipt["status"] != "error" or receipt.get("error_type") != "RenderTimeout" or "qa" in receipt:
            raise ValueError("native audio rejection or unsupported failure cannot be recovered")
        matches = [later for later in outcomes.values() if later["status"] == "accepted" and later["phase"] == key[0]
                   and later["attempt"] > key[1] and all(later.get(name) == receipt.get(name) for name in fields)]
        if not matches:
            raise ValueError("native timeout has no later accepted identical request")
        recovered.append({"phase": key[0], "failed_attempt": key[1], "accepted_attempt": matches[0]["attempt"], "request_id": receipt["request_id"]})
    if used_losses != set(losses):
        raise ValueError("declared process loss is absent from the native ledger")
    return attempts, outcomes, recovered


class RetainedAuditSession:
    def __init__(self, session, output):
        self.session, self.output = session, output

    def __getattr__(self, name):
        return getattr(self.session, name)

    def render(self, candidate, performance, phase, label):
        audio, receipt = self.session.render(candidate, performance, phase, label)
        if audio is None or label.rsplit(":", 1)[-1] in ("source", "repeat"):
            return audio, receipt
        digest = receipt["qa"]["audio_float32_sha256"]
        if audio_float32_sha256(audio) != digest:
            raise ValueError("comparison waveform differs from its native receipt")
        path = self.output / "comparison_audio" / f"{digest}.pt"
        path.parent.mkdir(exist_ok=True)
        if path.exists():
            saved = torch.load(path, map_location="cpu", weights_only=True)
            if audio_float32_sha256(saved["audio"].numpy()) != digest:
                raise ValueError("retained comparison waveform changed")
        else:
            atomic_torch_save(path, {"audio": torch.from_numpy(audio.copy())})
        metadata = {"file": path.relative_to(self.output).as_posix(), "sha256": sha256_bytes(path.read_bytes())}
        return audio, {**receipt, "retained_audio": metadata}


def rescore_waveform(source, receipt, reference, tail_seconds):
    return rescore_waveforms([(source, receipt, reference, tail_seconds)])[0]


def _retained_audio(source, receipt, reference):
    retained = receipt["retained_audio"]
    expected = f"comparison_audio/{receipt['qa']['audio_float32_sha256']}.pt"
    if retained["file"] != expected:
        raise ValueError("comparison waveform path changed")
    path = source / expected
    if sha256_bytes(path.read_bytes()) != retained["sha256"]:
        raise ValueError("comparison waveform file changed")
    saved = torch.load(path, map_location="cpu", weights_only=True)
    audio = saved["audio"].numpy()
    if (set(saved) != {"audio"} or audio.dtype != np.float32 or audio.shape != reference.shape
            or audio_float32_sha256(audio) != receipt["qa"]["audio_float32_sha256"]):
        raise ValueError("comparison audio does not match the recorded render")
    return audio


def rescore_waveforms(comparisons, *, batch_size=16):
    """Audit bounded batches of retained pairs; group equal lengths without altering padding."""
    BatchConfig(spectral=batch_size)
    comparisons, results, measure = iter(comparisons), [], SpectralTargets()
    while chunk := list(islice(comparisons, batch_size)):
        audio = [_retained_audio(source, receipt, reference) for source, receipt, reference, _tail in chunk]
        groups, scores = {}, {}
        for index, value in enumerate(audio):
            groups.setdefault(len(value), []).append(index)
        for positions in groups.values():
            references = np.stack([chunk[index][2] for index in positions])
            candidates = np.stack([audio[index] for index in positions])
            spectral = spectral_loss(measure(torch.from_numpy(candidates).transpose(1, 2)),
                                     measure(torch.from_numpy(references).transpose(1, 2)))["loss"].tolist()
            distances = audio_distances(references, candidates)
            for index, loss, distance in zip(positions, spectral, distances, strict=True):
                scores[index] = {"spectral": loss, "mrstft": distance["mrstft"]}
        for index, (_source, _receipt, reference, tail_seconds) in enumerate(chunk):
            tail = max(1, int(tail_seconds * 44100))
            ref_rms = np.sqrt(np.mean(reference[-tail:].astype(np.float64) ** 2, axis=0))
            rms = np.sqrt(np.mean(audio[index][-tail:].astype(np.float64) ** 2, axis=0))
            results.append({**scores[index], "tail_error": float(np.mean(np.abs(rms - ref_rms) / np.maximum(ref_rms, .001)))})
    return results


class WaveformAudit:
    """Bounded batching for callers that verify independently recorded scores."""
    def __init__(self, batch_size=16):
        self.batch_size = BatchConfig(spectral=batch_size).spectral
        self.pending = []

    def add(self, source, receipt, reference, tail_seconds, expected):
        self.pending.append(((source, receipt, reference, tail_seconds), expected))
        if len(self.pending) >= self.batch_size:
            self.finish()

    def finish(self):
        if not self.pending:
            return
        actual = rescore_waveforms((row[0] for row in self.pending), batch_size=self.batch_size)
        if actual != [row[1] for row in self.pending]:
            raise ValueError("comparison waveform cannot reproduce its recorded score")
        self.pending.clear()
