"""Durable native costs; only initial/final comparison waveforms are retained."""

from collections import OrderedDict
import math

import torch

from obruxo_data.hashing import canonical_sha256
from obruxo_data.midi import Performance
from obruxo_data.render.qa import audio_float32_sha256

from .basic_direct_fit import tensor_bytes
from .codec import PilotTarget
from .native_fit import append_durable, digest
from .paired_native_evidence import RetainedAuditSession, rescore_waveform
from .spectral import SpectralTargets, spectral_loss
from .stream import REJECT_AUDIO_CODES
from .unison_fit_data import require_request
from .unison_sound_fit import lines


def measurement_query(codec, case, view, candidate, tag=None):
    return {"case": case, "view": view, "candidate": codec.canonical_target(candidate).to_dict(), "tag": tag}


def allowed_trial_rejection(receipt):
    codes, qa = set(receipt.get("diagnostic_codes", [])), receipt.get("qa", {})
    errors = {code for code in codes if code.startswith("audio.")}
    allowed = REJECT_AUDIO_CODES | {"vital.runtime.canonicalization", "vital.runtime.numeric_canonicalization", "vital.runtime.version_canonicalization"}
    return (receipt["status"] == "rejected" and bool(errors & REJECT_AUDIO_CODES) and codes <= allowed
            and qa.get("finite") is True and qa.get("actual_frames") == qa.get("expected_frames")
            and qa.get("actual_channels") == qa.get("expected_channels") == 2)


class DirectStore:
    def __init__(self, output, plan, codec, fit, native=None):
        self.output, self.plan, self.codec, self.fit, self.native = output, plan, codec, fit, native
        self.retained = RetainedAuditSession(native, output) if native else None
        rows = lines(output / "measurements.jsonl")
        self.rows = {canonical_sha256(row["query"]): row for row in rows}
        if len(rows) != len(self.rows):
            raise ValueError("duplicate direct measurement key")
        self.sources, self.used = OrderedDict(), set()
        self.offline = native is None

    def check_limits(self):
        if self.native:
            self.native.check_limits()
            if sum(row["event"] == "attempt" for row in self.native.events) >= self.plan["maximum_native_requests"]:
                raise RuntimeError("direct total native request allocation exhausted")
        archive = sum(path.stat().st_size for path in self.output.rglob("*") if path.is_file())
        if archive + 32 * 1024**2 > self.plan["limits"]["archive_bytes"]:
            raise RuntimeError("direct evidence archive allocation exhausted")

    def source(self, case, view):
        recipe = self.plan["cases"][case]["recipe"]
        sha = recipe["source_audio_sha256"][view]
        if sha not in self.sources:
            name = f"source_audio/{sha}.pt"
            path = self.fit / name
            if digest(path) != self.plan["inputs"]["fit/" + name]:
                raise ValueError("direct retained source file changed")
            audio = torch.load(path, map_location="cpu", weights_only=True)["audio"]
            if audio_float32_sha256(audio.numpy()) != sha:
                raise ValueError("direct retained source samples changed")
            self.sources[sha] = (audio.numpy(), SpectralTargets()(audio.T[None]))
            while len(self.sources) > 2:
                self.sources.popitem(last=False)
            if sum(audio.nbytes + tensor_bytes(targets) for audio, targets in self.sources.values()) > self.plan["limits"]["replay_bytes"]:
                raise RuntimeError("direct resident source allocation exhausted")
        self.sources.move_to_end(sha)
        return self.sources[sha]

    def measure(self, case, view, candidate, role, *, tag=None, retain=False):
        query = measurement_query(self.codec, case, view, candidate, tag)
        key = canonical_sha256(query)
        self.used.add(key)
        if key in self.rows:
            row = self.rows[key]
            if retain and "retained_audio" not in row["receipt"]:
                raise ValueError("comparison waveform was not retained at its declared collection point")
            if row["loss"] is None and role != "trial":
                raise ValueError("invalid direct audio cannot supply a gradient, category or anchor")
            return row
        if self.offline:
            raise ValueError("offline direct replay is missing a native measurement")
        self.check_limits()
        performance = Performance.from_dict(self.plan["cases"][case]["recipe"]["performances"][view])
        session = self.retained if retain else self.native
        audio, receipt = session.render(candidate, performance, "qualification" if retain else "training", f"direct:{key}")
        if audio is None:
            if role != "trial" or not allowed_trial_rejection(receipt):
                raise RuntimeError(f"direct {role} native render failed: {receipt.get('error_type', receipt['status'])}")
            row = {"query": query, "role": role, "loss": None, "scores": None, "receipt": receipt}
        else:
            require_request(self.codec, self.plan, candidate, performance, receipt)
            reference, targets = self.source(case, view)
            value = float(spectral_loss(SpectralTargets()(torch.from_numpy(audio.copy()).T[None]), targets)["loss"])
            if not math.isfinite(value) or value < 0:
                raise ValueError("direct native objective must be finite and nonnegative")
            scores = rescore_waveform(self.output, receipt, reference, self.codec.contract.tail_seconds) if retain else {"spectral": value}
            if scores["spectral"] != value:
                raise ValueError("retained direct sound score differs from its live measurement")
            row = {"query": query, "role": role, "loss": value, "scores": scores, "receipt": receipt}
        append_durable(self.output / "measurements.jsonl", row)
        self.rows[key] = row
        return row

    def evaluate(self, case, candidate, role):
        rows = [self.measure(case, view, candidate, role) for view in self.plan["cases"][case]["views"]]
        return None if any(row["loss"] is None for row in rows) else sum(row["loss"] for row in rows) / 2

    def comparisons(self, case, candidate, stage):
        rows = []
        for view in self.plan["cases"][case]["views"]:
            first = self.measure(case, view, candidate, "comparison", tag=None if stage == "initial" else stage, retain=True)
            repeat = self.measure(case, view, candidate, "comparison", tag=stage + "-repeat", retain=True)
            if first["receipt"]["qa"]["audio_float32_sha256"] != repeat["receipt"]["qa"]["audio_float32_sha256"]:
                raise ValueError("direct fixed-phase repeat is not exact")
            rows.append({"view": view, "first": canonical_sha256(first["query"]), "repeat": canonical_sha256(repeat["query"]), "scores": first["scores"]})
        return rows

    def require_receipt(self, row):
        query, receipt = row["query"], row["receipt"]
        candidate = PilotTarget.from_dict(query["candidate"])
        performance = Performance.from_dict(self.plan["cases"][query["case"]]["recipe"]["performances"][query["view"]])
        if receipt["status"] == "accepted":
            require_request(self.codec, self.plan, candidate, performance, receipt)
            if row["loss"] is None or not math.isfinite(row["loss"]) or row["loss"] < 0 or row["scores"]["spectral"] != row["loss"]:
                raise ValueError("direct accepted sound score changed or is invalid")
        elif row["role"] != "trial" or row["loss"] is not None or row["scores"] is not None or not allowed_trial_rejection(receipt):
            raise ValueError("unsupported direct native rejection")
        else:
            require_request(self.codec, self.plan, candidate, performance, {**receipt, "status": "accepted"})
            current = {name: value for name, value in receipt["provenance"].items() if name != "request_id"}
            if current != self.plan["expected_renderer_identity"]:
                raise ValueError("rejected trial belongs to a different renderer")
        return candidate, performance
