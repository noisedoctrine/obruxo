"""Collect paired real-Vital targets and fit the Issue 172 comparison."""

from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
from functools import lru_cache
import json
from pathlib import Path
from time import perf_counter

import torch

from obruxo_data.hashing import canonical_sha256, sha256_bytes
from obruxo_data.midi import Performance
from obruxo_data.render.base import RenderRequest

from .audit import write_json
from .codec import PilotTarget
from .event_frames import sample_event_frames
from .paired_spectral import paired_change_loss, require_aligned_pairs
from .proxy import performance_inputs
from .render_profile import BASIC_PREROLL_FRAMES
from .reproducibility import fixed_generator_phases
from .spectral import SpectralTargets, spectral_loss
from .stream import NativeRendererFactory, RenderWorkerPool
from .surrogate_backbones import VitalSurrogate
from .temporal_proxy import collate_frames
from .trainer import atomic_torch_save


class Progress:
    def __init__(self, output):
        self.output = output
        self.phase, self.done, self.total = "Preparing", 0, 1
        self.current_stage, self.stage_count = 0, 12
        self.failures, self.last_write = 0, 0.

    def set(self, phase, total, index):
        self.phase, self.total, self.current_stage = phase, total, index
        self.update(0, log=True)

    def update(self, done, *, log=False, state="running", message=None):
        self.done = done
        now = perf_counter()
        if log or now - self.last_write >= 2 or done == self.total or state != "running":
            updated = datetime.now(timezone.utc).isoformat(timespec="seconds")
            payload = {"phase": self.phase, "completed": done, "total": self.total, "failures": self.failures,
                       "stage_index": self.current_stage, "stage_count": self.stage_count, "state": state,
                       "updated_at": updated, "message": message}
            write_json(self.output / "status.json", payload)
            self.last_write = now
            if log or done == self.total or state != "running":
                event = f"{updated} {self.phase}: {done}/{self.total} ({state})"
                if message:
                    event += f" • {message}"
                with (self.output / "events.log").open("a", encoding="utf-8") as handle:
                    handle.write(event + "\n")
                print(event, flush=True)


@contextmanager
def native_session(output, codec, workers):
    path = output / "renders.jsonl"
    renderer_id = codec.basic.contract.renderer_id
    worker = RenderWorkerPool(renderer_factory=NativeRendererFactory(renderer_id=renderer_id), worker_count=workers)
    previous = None
    if path.exists():
        with path.open(encoding="utf-8") as ledger:
            for line in ledger:
                row = json.loads(line)
                if "provenance" in row:
                    previous = {key: value for key, value in row["provenance"].items() if key != "request_id"}
                    break

    def render(jobs):
        nonlocal previous
        requests = [RenderRequest(preset_json=fixed_generator_phases(codec.decode(target)).decode(),
                                  performance=performance, renderer_id=renderer_id,
                                  tail_seconds=codec.basic.contract.tail_seconds, preroll_frames=BASIC_PREROLL_FRAMES)
                    for _, target, performance in jobs]
        with path.open("a", encoding="utf-8", newline="\n") as ledger:
            for (label, _, _), request in zip(jobs, requests, strict=True):
                ledger.write(json.dumps({"event": "attempt", "label": label, "request_id": request.request_id}) + "\n")
            ledger.flush()
            results = worker.render_many(requests, timeout=90.)
            for (label, _, _), request, (audio, receipt) in zip(jobs, requests, results, strict=True):
                ledger.write(json.dumps({"event": "outcome", "label": label, "request_id": request.request_id, **receipt},
                                        allow_nan=False) + "\n")
                if audio is not None:
                    current = {key: value for key, value in receipt["provenance"].items() if key != "request_id"}
                    if previous is not None and current != previous:
                        raise ValueError("native renderer provenance changed")
                    previous = current
        return results

    try:
        yield render
    finally:
        worker.close()


def collect(output, rows, render, progress):
    directory = output / "pairs"
    directory.mkdir(exist_ok=True)
    measured = SpectralTargets()
    existing = {path.stem for path in directory.glob("*.pt")}
    pending = [row for row in rows if row["pair_id"] not in existing]
    completed = len(rows) - len(pending)
    for start in range(0, len(pending), 2):
        group = pending[start:start + 2]
        jobs = [(f"{row['pair_id']}:{role}", PilotTarget.from_dict(row[field]), Performance.from_dict(row["performance"]))
                for row in group for role, field in (("original", "target"), ("perturbed", "changed"))]
        outcomes = render(jobs)
        for index, row in enumerate(group):
            pair = outcomes[index * 2:index * 2 + 2]
            accepted = all(audio is not None for audio, _ in pair)
            entry = {"id": canonical_sha256(row), "records": [], "accepted": accepted}
            if accepted:
                perf = jobs[index * 2][2]
                played = performance_inputs(perf)
                seed = int(entry["id"][:15], 16)
                for (audio, _), field in zip(pair, ("target", "changed"), strict=True):
                    full = measured(torch.from_numpy(audio.copy()).T[None])
                    values, times = sample_event_frames(full, perf, torch.Generator().manual_seed(seed))
                    entry["records"].append({"measurements": values, "times": times, "target": row[field],
                                             "performance": played, "pair_id": row["pair_id"], "source_state_id": row["state_id"]})
                require_aligned_pairs(entry["records"])
            else:
                progress.failures += 1
            atomic_torch_save(directory / (row["pair_id"] + ".pt"), entry)
            completed += 1
            progress.update(completed)
    rows_by_id = {row["pair_id"]: row for row in rows}
    report = {"planned_pairs": len(rows), "accepted_pairs": 0, "failed_pairs": 0, "controls": Counter()}
    accepted_rows = []
    for pair_id, row in rows_by_id.items():
        path = directory / (pair_id + ".pt")
        entry = torch.load(path, map_location="cpu", weights_only=True)
        if entry["id"] != canonical_sha256(row):
            raise ValueError("saved pair and frozen render recipe differ")
        if entry["accepted"]:
            accepted_rows.append(row)
            report["accepted_pairs"] += 1
            report["controls"][row["control"]] += 1
        else:
            report["failed_pairs"] += 1
    if len(accepted_rows) < .9 * len(rows):
        raise RuntimeError(f"fewer than 90% of native pairs passed QA: {report}")
    return accepted_rows, report


def control_tensor(codec, targets, device):
    rows = [(*target.continuous, *codec.category_vector(target.categorical)) for target in targets]
    return torch.tensor(rows, dtype=torch.float32, device=device)


def finish_model(model):
    model.eval().requires_grad_(False)
    return model


def widen_model(model, parent):
    original = dict(parent)
    first = original.pop("input.weight")
    missing, unexpected = model.load_state_dict(original, strict=False)
    if missing != ["input.weight"] or unexpected:
        raise ValueError("unexpected model differences during control expansion")
    with torch.no_grad():
        weight = model.input.weight
        weight.zero_()
        weight[:, :9] = first[:, :9]
        weight[:, 13:17] = first[:, 9:13]
        weight[:, 33:] = first[:, 13:]


def fit(output, label, backbone, codec, rows, *, steps, plan_id, device, progress):
    torch.manual_seed(172)
    model = VitalSurrogate(backbone, codec.width).to(device)
    checkpoint = output / f"{label}-{backbone}.pt"
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
    step, trace, elapsed, cursor = 0, [], 0., 0
    shuffle = torch.Generator().manual_seed(172)
    order = torch.randperm(len(rows), generator=shuffle).tolist()
    if checkpoint.exists():
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if saved["plan_id"] != plan_id or saved["stage"] != label or saved["backbone"] != backbone:
            raise ValueError("checkpoint does not belong to this run")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        step, trace, elapsed, cursor, order = (saved[name] for name in ("step", "trace", "elapsed", "cursor", "order"))
        shuffle.set_state(saved["shuffle"])
    elif codec.expanded:
        parent = torch.load(output / f"basic-{backbone}.pt", map_location="cpu", weights_only=True)
        if parent["step"] != steps or parent["plan_id"] != plan_id:
            raise ValueError("incomplete or incompatible basic parent")
        widen_model(model, parent["model"])

    @lru_cache(maxsize=256)
    def load(pair_id):
        return torch.load(output / "pairs" / (pair_id + ".pt"), map_location="cpu", weights_only=True)["records"]

    started = perf_counter()
    for step_number in range(step + 1, steps + 1):
        if cursor >= len(order):
            order, cursor = torch.randperm(len(rows), generator=shuffle).tolist(), 0
        selected = order[cursor:cursor + 4]
        cursor += len(selected)
        records = [record for index in selected for record in load(rows[index]["pair_id"])]
        measurements, times = collate_frames(records)
        measurements = {key: tuple(item.to(device) for item in value) if isinstance(value, tuple) else value.to(device)
                        for key, value in measurements.items()}
        times = tuple(value.to(device) for value in times)
        targets = [PilotTarget.from_dict(record["target"]) for record in records]
        if codec.expanded:
            targets = [codec.from_basic(target) if len(target.continuous) == 9 else target for target in targets]
        controls = control_tensor(codec, targets, device)
        played = torch.stack([record["performance"] for record in records]).to(device)
        optimizer.zero_grad(set_to_none=True)
        prediction = model(controls, played, times, measurements["lengths"])
        absolute = spectral_loss(prediction, measurements)["loss"].mean()
        response = paired_change_loss(prediction, measurements).mean()
        (absolute + .25 * response).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
        optimizer.step()
        if step_number == 1 or step_number % 100 == 0 or step_number == steps:
            current_elapsed = elapsed + perf_counter() - started
            trace.append({"step": step_number, "spectral_envelope_error": float(absolute.detach()),
                          "control_change_error": float(response.detach()), "elapsed_seconds": current_elapsed})
            atomic_torch_save(checkpoint, {"plan_id": plan_id, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                                           "step": step_number, "trace": trace, "elapsed": current_elapsed, "backbone": backbone,
                                           "stage": label, "order": order, "cursor": cursor, "shuffle": shuffle.get_state()})
        progress.update(step_number, log=step_number % 500 == 0 and step_number < steps)
    load.cache_clear()
    if step == steps:
        progress.update(steps)
    return finish_model(model), {"trace": trace, "parameters": sum(p.numel() for p in model.parameters())}


def source_fingerprints():
    files = ("surrogate_codec.py", "surrogate_pilot_data.py", "surrogate_backbones.py", "surrogate_pilot.py",
             "surrogate_pilot_eval.py", "spectral.py", "stream.py", "event_frames.py", "paired_spectral.py")
    return {name: sha256_bytes(Path(__file__).with_name(name).read_bytes()) for name in files}
