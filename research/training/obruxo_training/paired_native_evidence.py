"""Retain auditable comparison audio and account for every native attempt."""

import numpy as np
import torch

from obruxo_data.hashing import sha256_bytes
from obruxo_data.render.qa import audio_float32_sha256

from .checkpoints import atomic_torch_save
from .evaluate import audio_distance
from .spectral import SpectralTargets, spectral_loss


def native_ledger(events, renderer_identity):
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
    recovered = []
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
        if receipt["status"] != "error" or receipt.get("error_type") != "RenderTimeout" or "qa" in receipt:
            raise ValueError("native audio rejection or unsupported failure cannot be recovered")
        matches = [later for later in outcomes.values() if later["status"] == "accepted" and later["phase"] == key[0]
                   and later["attempt"] > key[1] and all(later.get(name) == receipt.get(name) for name in fields)]
        if not matches:
            raise ValueError("native timeout has no later accepted identical request")
        recovered.append({"phase": key[0], "failed_attempt": key[1], "accepted_attempt": matches[0]["attempt"], "request_id": receipt["request_id"]})
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
    measure = SpectralTargets()
    spectral = float(spectral_loss(measure(torch.from_numpy(audio.copy()).T[None]), measure(torch.from_numpy(reference.copy()).T[None]))["loss"])
    tail = max(1, int(tail_seconds * 44100))
    ref_rms = np.sqrt(np.mean(reference[-tail:].astype(np.float64) ** 2, axis=0))
    rms = np.sqrt(np.mean(audio[-tail:].astype(np.float64) ** 2, axis=0))
    return {"spectral": spectral, "mrstft": audio_distance(reference, audio)["mrstft"],
            "tail_error": float(np.mean(np.abs(rms - ref_rms) / np.maximum(ref_rms, .001)))}
