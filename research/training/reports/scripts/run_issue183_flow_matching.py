"""Run the matched direct-prediction and conditional-flow pilot for issue 183."""

# ruff: noqa: E402
import argparse
from collections import defaultdict
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import torch

TRAINING = Path(__file__).resolve().parents[2]
RESEARCH = TRAINING.parent
WORKTREE = TRAINING.parents[1]
MAIN_ROOT = WORKTREE.parent.parent if WORKTREE.parent.name == "tmp" else WORKTREE
sys.path[:0] = [str(RESEARCH / "data_generation"), str(TRAINING)]

from obruxo_data.hashing import canonical_sha256
from obruxo_training.codec import PilotCodec, PilotTarget
from obruxo_training.flow_matching import (DirectSpectrumModel, FlowSpectrumModel, as_spectrogram,
                                           log_spectrum, one_resolution_spectral_loss)
from obruxo_training.spectral import SAMPLE_RATE
from obruxo_training.surrogate_codec import SurrogateCodec
from obruxo_training.surrogate_pilot import control_tensor
from obruxo_training.trainer import atomic_torch_save

BACKBONES = ("residual_mlp", "transformer", "conformer")
BACKBONE_LABELS = {"residual_mlp": "Residual MLP", "transformer": "Transformer", "conformer": "Conformer"}
MODES = ("direct", "flow")
FFT_SIZE, HOP = 2048, 512
FREQUENCY_BINS = FFT_SIZE // 2 + 1
FLOW_STEPS = (4, 8, 16)
VALIDATION_EVERY = 100
BATCH_PAIRS = 4
SEED = 183
TRAINING_PROBE_PAIRS = 16
TRAINING_PROBE_NOISE_SEEDS = (70183, 70184, 70185)
TRAINING_SOURCE = MAIN_ROOT / "research/training/outputs/issue172-surrogate-pilot-20261126"
VALIDATION_SOURCE = MAIN_ROOT / "research/training/outputs/issue173-model-selection-active-20261126-v4"


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _receipt_payload(receipts):
    return "".join(json.dumps(receipt, sort_keys=True) + "\n" for receipt in receipts).encode("utf-8")


def receipt_sha256(receipts):
    return hashlib.sha256(_receipt_payload(receipts)).hexdigest()


def write_receipts(path, receipts):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(_receipt_payload(receipts))
    temporary.replace(path)


def append_jsonl(path, value):
    with Path(path).open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def repo_reference(path, label):
    path = Path(path).resolve()
    try:
        return path.relative_to(MAIN_ROOT.resolve()).as_posix()
    except ValueError:
        return f"external:{label}"


class Progress:
    def __init__(self, output, step_total):
        self.output, self.step_total = output, step_total
        self.last_write = 0.
        self.status = {"phase": "Starting", "arm": None, "completed": 0, "total": step_total,
                       "arm_index": 0, "arm_count": len(BACKBONES) * len(MODES), "failures": 0,
                       "state": "running", "updated_at": None, "message": None}

    def update(self, *, phase=None, arm=None, completed=None, total=None, arm_index=None, state="running",
               message=None, force=False):
        if phase is not None:
            self.status["phase"] = phase
        if arm is not None:
            self.status["arm"] = arm
        if completed is not None:
            self.status["completed"] = int(completed)
        if total is not None:
            self.status["total"] = int(total)
        if arm_index is not None:
            self.status["arm_index"] = int(arm_index)
        self.status["state"] = state
        self.status["message"] = message
        now = perf_counter()
        if force or now - self.last_write >= 2 or state != "running":
            self.status["updated_at"] = __import__("datetime").datetime.now(
                __import__("datetime").timezone.utc).isoformat(timespec="seconds")
            write_json(self.output / "status.json", self.status)
            if force or state != "running":
                append_jsonl(self.output / "events.jsonl", {
                    "updated_at": self.status["updated_at"], "phase": self.status["phase"],
                    "arm": self.status["arm"], "completed": self.status["completed"],
                    "total": self.status["total"], "state": state, "message": message})
            self.last_write = now


def _load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _verify_shape(entry, source, pair_id):
    if not entry.get("accepted") or len(entry.get("records", [])) != 2:
        raise ValueError(f"expected an accepted original/perturbed cache pair: {pair_id}")
    for record in entry["records"]:
        measurements = record["measurements"]
        magnitude, mask = measurements["magnitudes"][1], measurements["masks"][1]
        if tuple(magnitude.shape) != (1, 2, FREQUENCY_BINS, 32) or tuple(mask.shape) != (1, 32):
            raise ValueError(f"unexpected 2048-point frame shape for {pair_id}")
        if not torch.all(mask) or not torch.isfinite(magnitude).all() or (magnitude < 0).any():
            raise ValueError(f"invalid spectral mask or magnitude for {pair_id}")
        if tuple(record["times"][1].shape) != (1, 32):
            raise ValueError(f"unexpected frame-time shape for {pair_id}")
        expected_width = 9 if source in {"basic"} else 13
        if len(record["target"]["continuous"]) != expected_width:
            raise ValueError(f"wrong {source} control representation in {pair_id}")


def load_sources(train_run, validation_run, output, progress):
    train_run, validation_run = Path(train_run), Path(validation_run)
    train_plan_file, validation_plan_file = train_run / "plan.json", validation_run / "plan.json"
    source_plan = _load_json(train_plan_file)["plan"]
    validation_plan = _load_json(validation_plan_file)["plan"]
    if source_plan.get("format") != "matched_two_stage_surrogate_v1":
        raise ValueError("training input is not the frozen issue 172 run")
    validation_recipes = validation_plan["evaluation"]["validation_proxy"]["pair_recipes"]
    training_receipts = {row["pair_id"]: row for row in validation_plan["training"]["pair_receipts"]}
    train_dir, val_dir = train_run / "pairs", validation_run / "validation-pairs"
    rows, receipts = [], []
    planned = sum(len(source_plan["stages"][stage]["pairs"]) for stage in ("basic", "expanded"))
    progress.update(phase="Verifying cached train and validation pairs", completed=0, total=planned + len(validation_recipes),
                    force=True)
    completed = 0
    for source in ("basic", "expanded"):
        for row in source_plan["stages"][source]["pairs"]:
            pair_id = row["pair_id"]
            path = train_dir / (pair_id + ".pt")
            expected = training_receipts.get(pair_id)
            if expected is None or expected["recipe_sha256"] != canonical_sha256(row):
                raise ValueError(f"missing or changed #173 recipe receipt: {pair_id}")
            actual_sha = sha256_file(path)
            if actual_sha != expected["cache_sha256"]:
                raise ValueError(f"#172 training cache differs from its #173 receipt: {pair_id}")
            entry = torch.load(path, map_location="cpu", weights_only=True)
            if entry["id"] != canonical_sha256(row) or bool(entry["accepted"]) != bool(expected["accepted"]):
                raise ValueError(f"#172 pair identity or acceptance differs from its receipt: {pair_id}")
            if entry["accepted"]:
                _verify_shape(entry, source, pair_id)
                rows.append({**row, "source_stage": source})
            receipts.append({"source": "issue172", "stage": source, "pair_id": pair_id,
                             "state_id": row["state_id"], "midi_id": row["midi_id"],
                             "accepted": bool(entry["accepted"]), "recipe_sha256": entry["id"],
                             "cache_sha256": actual_sha})
            completed += 1
            progress.update(phase="Verifying cached train and validation pairs", completed=completed, total=planned + len(validation_recipes))
    if len(rows) != 2046 or len({row["state_id"] for row in rows}) != 512:
        raise ValueError(f"issue 172 cache no longer has 2,046 accepted pairs across 512 patch roots: {len(rows)}")
    if len(training_receipts) != planned:
        raise ValueError("issue 173 receipt set does not cover the issue 172 pair plan")

    val_rows = []
    for index, row in enumerate(validation_recipes):
        pair_id = row["pair_id"]
        path = val_dir / (pair_id + ".pt")
        entry = torch.load(path, map_location="cpu", weights_only=True)
        if entry["id"] != canonical_sha256(row):
            raise ValueError(f"issue 173 validation recipe changed: {pair_id}")
        _verify_shape(entry, "expanded", pair_id)
        cache_sha = sha256_file(path)
        val_rows.append(row)
        receipts.append({"source": "issue173", "stage": "validation", "pair_id": pair_id,
                         "state_id": row["state_id"], "midi_id": row["midi_id"],
                         "accepted": True, "recipe_sha256": entry["id"], "cache_sha256": cache_sha})
        completed += 1
        progress.update(phase="Verifying cached train and validation pairs", completed=completed, total=planned + len(validation_recipes))
    if (len(val_rows) != 128 or len({row["state_id"] for row in val_rows}) != 64
            or len({row["midi_id"] for row in val_rows}) != 16):
        raise ValueError("issue 173 validation cache must contain 128 pairs, 64 roots and 16 MIDI performances")
    return source_plan, validation_plan, rows, val_rows, receipts


def collate_pairs(rows, pair_dir, codec, device, cache):
    records, metas = [], []
    for row in rows:
        for clip_role, record in zip(("original", "perturbed"), cache(row["pair_id"]), strict=True):
            target = PilotTarget.from_dict(record["target"])
            if len(target.continuous) == 9:
                target = codec.from_basic(target)
            records.append(record)
            metas.append({"pair_id": row["pair_id"], "state_id": row["state_id"],
                          "midi_id": row["midi_id"], "clip_role": clip_role,
                          "performance": record["performance"], "times_cpu": record["times"][1][0]})
            if target != PilotTarget.from_dict(record["target"]):
                record = {**record, "target": target.to_dict()}
                records[-1] = record
    spectra = torch.cat([record["measurements"]["magnitudes"][1] for record in records], dim=0).to(device)
    mask = torch.cat([record["measurements"]["masks"][1] for record in records], dim=0).to(device).bool()
    times = torch.cat([record["times"][1] for record in records], dim=0).to(device)
    lengths = torch.cat([record["measurements"]["lengths"] for record in records], dim=0).to(device)
    performance = torch.stack([record["performance"] for record in records]).to(device)
    targets = [PilotTarget.from_dict(record["target"]) for record in records]
    controls = control_tensor(codec, targets, device)
    log_values = log_spectrum(spectra).permute(0, 3, 1, 2).reshape(len(records), 32, -1)
    return {"spectra": spectra, "mask": mask, "times": times, "lengths": lengths,
            "performance": performance, "controls": controls, "log_values": log_values}, metas


def load_model(backbone, mode, control_width):
    model_type = DirectSpectrumModel if mode == "direct" else FlowSpectrumModel
    return model_type(backbone, control_width)


def cpu_generator(seed):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return generator


def _training_objective(model, mode, batch, noise_rng):
    expected = batch["spectra"]
    if mode == "direct":
        predicted = as_spectrogram(model(batch["controls"], batch["performance"], batch["times"], batch["lengths"]))
        return one_resolution_spectral_loss(predicted, expected, batch["mask"])["loss"].mean()
    clean = batch["log_values"]
    noise = torch.randn(clean.shape, generator=noise_rng).to(clean.device)
    time = torch.rand((len(clean),), generator=noise_rng)
    time = time.to(clean.device)
    noisy = (1 - time[:, None, None]) * noise + time[:, None, None] * clean
    context = model.context(batch["controls"], batch["performance"], batch["times"], batch["lengths"])
    velocity = model.velocity_from_context(context, noisy, time, batch["mask"], noise)
    squared = (velocity - (clean - noise)).square().mean(dim=-1)
    return (squared * batch["mask"]).sum() / batch["mask"].sum().clamp_min(1)


def _phase_at(performance, time):
    notes = performance[:80].reshape(16, 5)
    available = notes[:, 4] > 0.5
    start, duration = notes[:, 2] * 16., notes[:, 3] * 16.
    age = time - start
    active = available & (age >= 0) & (age < duration)
    if torch.any(active & (age < .05)):
        return "attack", int(active.sum()), int(torch.nonzero(active, as_tuple=False)[0])
    if torch.any(active & (age < .25)):
        return "decay", int(active.sum()), int(torch.nonzero(active, as_tuple=False)[0])
    if torch.any(active):
        return "sustain", int(active.sum()), int(torch.nonzero(active, as_tuple=False)[0])
    released = available & (age >= duration) & (age < duration + .30)
    if torch.any(released):
        return "release", 0, int(torch.nonzero(released, as_tuple=False)[0])
    return None, 0, None


def _octave_energy_error_db(predicted, expected, bands):
    """Mean absolute octave-band power error in dB, without band-share normalization."""
    weights = np.full(FREQUENCY_BINS, 2., dtype=np.float64)
    weights[0], weights[-1] = 1., 1.

    def energy(spectrum):
        return np.asarray([(np.square(spectrum[:, band], dtype=np.float64)
                            * weights[band][None]).sum() for band in bands])

    pred, target = energy(predicted), energy(expected)
    target_total = target.sum()
    if target_total <= 1e-12:
        return None
    floor = max(target_total * 1e-8, 1e-12)
    return float(np.abs(10. * np.log10((pred + floor) / (target + floor))).mean())


def _octave_band_energy_error_db(predicted, expected):
    frequencies = np.fft.rfftfreq(FFT_SIZE, 1 / SAMPLE_RATE)
    edges = (SAMPLE_RATE / FFT_SIZE) * 2. ** np.arange(11)
    bands = [(frequencies >= low) & ((frequencies <= high) if index == 9 else (frequencies < high))
             for index, (low, high) in enumerate(zip(edges[:-1], edges[1:], strict=True))]
    return _octave_energy_error_db(predicted, expected, bands)


def _monophonic_octave_energy_error_db(predicted, expected, performance, time):
    """Pitch-centered absolute band-energy error for monophonic frames only."""
    _, active_count, note_index = _phase_at(performance, time)
    if active_count != 1:
        return None
    notes = performance[:80].reshape(16, 5)
    midi_pitch = round(float(notes[note_index, 0]) * 127)
    fundamental = 440. * 2 ** ((midi_pitch - 69) / 12)
    frequencies = np.fft.rfftfreq(FFT_SIZE, 1 / SAMPLE_RATE)
    edges = fundamental * 2. ** np.arange(-2.5, 3., 1.)
    if edges[-1] > frequencies[-1]:
        return None
    bands = [(frequencies >= low) & ((frequencies <= high) if index == 4 else (frequencies < high))
             for index, (low, high) in enumerate(zip(edges[:-1], edges[1:], strict=True))]
    if not all(band.any() for band in bands):
        return None
    return _octave_energy_error_db(predicted, expected, bands)


def score_batch(prediction, batch, metas):
    predicted = as_spectrogram(prediction)
    expected, frame_mask = batch["spectra"], batch["mask"]
    loss = one_resolution_spectral_loss(predicted, expected, frame_mask)
    pred_cpu, target_cpu = predicted.detach().float().cpu().numpy(), expected.detach().float().cpu().numpy()
    mask_cpu = frame_mask.detach().cpu().numpy()
    time_cpu = batch["times"].detach().float().cpu().numpy()
    performance_cpu = batch["performance"].detach().float().cpu()
    pred_weight = np.ones(FREQUENCY_BINS, dtype=np.float64) * 2
    pred_weight[0], pred_weight[-1] = 1, 1
    window_energy = float(torch.hann_window(FFT_SIZE).square().sum())
    rows = []
    for index, meta in enumerate(metas):
        record = {key: meta[key] for key in ("pair_id", "state_id", "midi_id", "clip_role")}
        record["spectral_loss"] = float(loss["loss"][index])
        record["spectral_convergence"] = float(loss["spectral_convergence"][index])
        record["log_magnitude_mae"] = float(loss["log_magnitude_mae"][index])
        phase_errors, octave_errors, monophonic_errors = defaultdict(list), [], []
        monophonic_frames = 0
        for frame in np.flatnonzero(mask_cpu[index]):
            pred_frame = pred_cpu[index, :, :, frame]
            target_frame = target_cpu[index, :, :, frame]
            for phase, active, note_index in (_phase_at(performance_cpu[index], float(time_cpu[index, frame])),):
                if phase is not None:
                    pred_power = (pred_frame ** 2 * pred_weight[None]).sum(1).mean()
                    target_power = (target_frame ** 2 * pred_weight[None]).sum(1).mean()
                    pred_rms = math.sqrt(max(0., float(pred_power)) / window_energy)
                    target_rms = math.sqrt(max(0., float(target_power)) / window_energy)
                    phase_errors[phase].append(abs(20 * math.log10((pred_rms + 1e-8) / (target_rms + 1e-8))))
                error = _octave_band_energy_error_db(pred_frame, target_frame)
                if error is not None:
                    octave_errors.append(error)
                if active == 1:
                    monophonic_frames += 1
                    error = _monophonic_octave_energy_error_db(
                        pred_frame, target_frame, performance_cpu[index], float(time_cpu[index, frame]))
                    if error is not None:
                        monophonic_errors.append(error)
        record["octave_band_energy_error_db"] = float(np.mean(octave_errors)) if octave_errors else None
        record["octave_band_frames"] = len(octave_errors)
        record["octave_band_total_frames"] = int(mask_cpu[index].sum())
        record["monophonic_octave_energy_error_db"] = float(np.mean(monophonic_errors)) if monophonic_errors else None
        record["monophonic_frames"] = len(monophonic_errors)
        record["monophonic_frames_total"] = monophonic_frames
        for phase in ("attack", "decay", "sustain", "release"):
            record[f"{phase}_envelope_error_db"] = float(np.mean(phase_errors[phase])) if phase_errors[phase] else None
            record[f"{phase}_frames"] = len(phase_errors[phase])
        rows.append(record)
    return rows


def _deterministic_noise(shape, seed, device):
    return torch.randn(shape, generator=cpu_generator(seed), dtype=torch.float32).to(device)


@torch.no_grad()
def evaluate_model(model, mode, rows, pair_dir, codec, device, *, flow_steps=8, seed=SEED,
                   progress=None, phase="Validation"):
    model.eval()
    cache = _pair_loader(pair_dir)
    results = []
    batch_count = math.ceil(len(rows) / BATCH_PAIRS)
    for index in range(0, len(rows), BATCH_PAIRS):
        selected = rows[index:index + BATCH_PAIRS]
        batch, metas = collate_pairs(selected, pair_dir, codec, device, cache)
        if mode == "direct":
            output = model(batch["controls"], batch["performance"], batch["times"], batch["lengths"])
        else:
            noise = _deterministic_noise((len(batch["spectra"]), 32, FREQUENCY_BINS * 2),
                                         seed + index // BATCH_PAIRS, device)
            context = model.context(batch["controls"], batch["performance"], batch["times"], batch["lengths"])
            output = model.sample_from_context(context, batch["mask"], noise, flow_steps)
        results.extend(score_batch(output, batch, metas))
        if progress is not None:
            progress.update(phase=phase, completed=min(index + BATCH_PAIRS, len(rows)),
                            total=len(rows), force=index + BATCH_PAIRS >= len(rows))
    return aggregate_cases(results), results


def fixed_training_probe(rows, count=TRAINING_PROBE_PAIRS):
    by_root = {}
    for row in rows:
        by_root.setdefault(row["state_id"], row)
    return [by_root[root] for root in sorted(by_root)[:count]]


@torch.no_grad()
def evaluate_flow_training_probe(model, rows, pair_dir, codec, device, *, seed, progress=None, phase="Training probe"):
    batch, _ = collate_pairs(rows, pair_dir, codec, device, _pair_loader(pair_dir))
    clean = batch["log_values"]
    noise = _deterministic_noise(clean.shape, seed, device)
    rng = cpu_generator(seed + 1)
    time = torch.rand((len(clean),), generator=rng).to(device)
    noisy = (1 - time[:, None, None]) * noise + time[:, None, None] * clean
    context = model.context(batch["controls"], batch["performance"], batch["times"], batch["lengths"])
    velocity = model.velocity_from_context(context, noisy, time, batch["mask"], noise)
    velocity_mse = ((velocity - (clean - noise)).square().mean(-1) * batch["mask"]).sum()
    velocity_mse /= batch["mask"].sum().clamp_min(1)
    sampled, _ = evaluate_model(model, "flow", rows, pair_dir, codec, device, flow_steps=8,
                                seed=seed, progress=progress, phase=phase)
    return {"recipes": len(rows), "clips": 2 * len(rows),
            "patch_roots": len({row["state_id"] for row in rows}), "noise_seed": seed,
            "velocity_mse": float(velocity_mse), "sampled_spectral_loss": sampled["spectral_loss"],
            "sampled_octave_band_energy_error_db": sampled["octave_band_energy_error_db"]}


def aggregate_cases(cases):
    keys = ("spectral_loss", "spectral_convergence", "log_magnitude_mae", "octave_band_energy_error_db",
            "monophonic_octave_energy_error_db",
            "attack_envelope_error_db", "decay_envelope_error_db", "sustain_envelope_error_db",
            "release_envelope_error_db")
    result = {key: float(np.mean([row[key] for row in cases if row.get(key) is not None]))
              if any(row.get(key) is not None for row in cases) else None for key in keys}
    result.update(clips=len(cases), patch_roots=len({row["state_id"] for row in cases}),
                  midi_performances=len({row["midi_id"] for row in cases}),
                   monophonic_frames=sum(row["monophonic_frames"] for row in cases),
                   monophonic_frames_total=sum(row["monophonic_frames_total"] for row in cases),
                   octave_band_frames=sum(row["octave_band_frames"] for row in cases),
                   octave_band_total_frames=sum(row["octave_band_total_frames"] for row in cases))
    for phase in ("attack", "decay", "sustain", "release"):
        result[f"{phase}_frames"] = sum(row[f"{phase}_frames"] for row in cases)
    return result


def _pair_root_values(cases, metric):
    grouped = defaultdict(list)
    for row in cases:
        value = row.get(metric)
        if value is not None and math.isfinite(value):
            grouped[row["state_id"]].append(float(value))
    return {root: float(np.mean(values)) for root, values in grouped.items()}


def paired_effect(direct, flow, metric, seed=SEED):
    a, b = _pair_root_values(direct, metric), _pair_root_values(flow, metric)
    roots = sorted(a.keys() & b.keys())
    if not roots:
        return {"metric": metric, "patch_roots": 0, "flow_minus_direct": None, "ci95": None}
    differences = np.asarray([b[root] - a[root] for root in roots])
    rng = np.random.default_rng(seed)
    samples = differences[rng.integers(0, len(differences), size=(10000, len(differences)))].mean(axis=1)
    return {"metric": metric, "patch_roots": len(roots),
            "flow_minus_direct": float(differences.mean()),
            "ci95": [float(x) for x in np.quantile(samples, [.025, .975])],
            "interval_scope": "patch-root bootstrap; MIDI-performance diversity is not resampled"}


def _pair_loader(pair_dir):
    @lru_cache(maxsize=64)
    def load(pair_id):
        return torch.load(Path(pair_dir) / (pair_id + ".pt"), map_location="cpu", weights_only=True)["records"]
    return load


def save_trace(output, arm, trace):
    trace_dir = output / "traces"
    trace_dir.mkdir(exist_ok=True)
    path = trace_dir / f"{arm}.jsonl"
    temporary = path.with_suffix(".jsonl.tmp")
    temporary.write_text("".join(json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in trace),
                         encoding="utf-8")
    temporary.replace(path)


def fit_arm(output, plan_id, backbone, mode, rows, validation_rows, codec, pair_dir, val_dir,
            train_probe_rows, device, progress, arm_index, steps, validation_every, seed):
    arm = f"{backbone}-{mode}"
    torch.manual_seed(seed)
    if device.type == "xpu":
        torch.xpu.manual_seed_all(seed)
    model = load_model(backbone, mode, codec.width).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
    checkpoint = output / f"{arm}.pt"
    shuffle = torch.Generator(device="cpu").manual_seed(seed)
    noise_rng = cpu_generator(seed + 1000)
    order = torch.randperm(len(rows), generator=shuffle).tolist()
    cursor, step, trace, train_elapsed, val_elapsed, probe_elapsed = 0, 0, [], 0., 0., 0.
    if checkpoint.exists():
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if saved["plan_id"] != plan_id or saved["arm"] != arm or saved["seed"] != seed:
            raise ValueError(f"checkpoint belongs to a different run: {arm}")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        for state in optimizer.state.values():
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    state[key] = value.to(device)
        step, trace, train_elapsed, val_elapsed = saved["step"], saved["trace"], saved["train_elapsed"], saved["val_elapsed"]
        probe_elapsed = saved.get("training_probe_elapsed", 0.)
        order, cursor = saved["order"], saved["cursor"]
        shuffle.set_state(saved["shuffle_state"])
        noise_rng.set_state(saved["noise_state"])

    cache = _pair_loader(pair_dir)

    def fixed_probe(update):
        if mode != "flow":
            return None
        progress.update(phase=f"{arm} · fixed training reconstruction probe", arm=arm,
                        completed=update, total=steps, arm_index=arm_index, force=True)
        result = evaluate_flow_training_probe(model, train_probe_rows, pair_dir, codec, device,
                                              seed=seed + 70000)
        progress.update(phase=f"Training {arm}", arm=arm, completed=update, total=steps,
                        arm_index=arm_index, force=True)
        return result

    if step == 0:
        start = perf_counter()
        metrics, _ = evaluate_model(model, mode, validation_rows, val_dir, codec, device, flow_steps=8,
                                    seed=seed + 90000, progress=progress, phase=f"{arm} · validation at update 0")
        val_elapsed += perf_counter() - start
        probe_start = perf_counter()
        probe = fixed_probe(0)
        probe_elapsed += perf_counter() - probe_start
        trace.append({"step": 0, "train_objective": None, "validation_spectral_loss": metrics["spectral_loss"],
                      "validation_octave_band_energy_error_db": metrics["octave_band_energy_error_db"], "validation_metrics": metrics,
                      "validation_flow_steps": 8 if mode == "flow" else None,
                      "training_probe": probe})
        save_trace(output, arm, trace)
    training_window, prior_train_elapsed, started = [], train_elapsed, perf_counter()
    for update in range(step + 1, steps + 1):
        if cursor + BATCH_PAIRS > len(order):
            order, cursor = torch.randperm(len(rows), generator=shuffle).tolist(), 0
        selected = [rows[position] for position in order[cursor:cursor + BATCH_PAIRS]]
        cursor += BATCH_PAIRS
        batch, _ = collate_pairs(selected, pair_dir, codec, device, cache)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = _training_objective(model, mode, batch, noise_rng)
        if not torch.isfinite(loss):
            raise ValueError(f"nonfinite training objective at {arm} update {update}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
        optimizer.step()
        training_window.append(float(loss.detach()))
        if update % validation_every == 0 or update == steps:
            if device.type == "xpu":
                torch.xpu.synchronize(device)
            train_elapsed = prior_train_elapsed + perf_counter() - started
            val_start = perf_counter()
            metrics, _ = evaluate_model(model, mode, validation_rows, val_dir, codec, device, flow_steps=8,
                                        seed=seed + 90000, progress=progress, phase=f"{arm} · validation at update {update}")
            if device.type == "xpu":
                torch.xpu.synchronize(device)
            val_elapsed += perf_counter() - val_start
            prior_train_elapsed, started = train_elapsed, perf_counter()
            probe_start = perf_counter()
            probe = fixed_probe(update)
            probe_elapsed += perf_counter() - probe_start
            started = perf_counter()
            trace.append({"step": update, "train_objective": float(np.mean(training_window)),
                          "validation_spectral_loss": metrics["spectral_loss"],
                          "validation_octave_band_energy_error_db": metrics["octave_band_energy_error_db"],
                          "validation_metrics": metrics, "validation_flow_steps": 8 if mode == "flow" else None,
                          "training_probe": probe,
                          "train_elapsed_seconds": train_elapsed, "validation_elapsed_seconds": val_elapsed})
            training_window.clear()
            save_trace(output, arm, trace)
            atomic_torch_save(checkpoint, {
                "plan_id": plan_id, "arm": arm, "backbone": backbone, "mode": mode, "seed": seed,
                "model": model.state_dict(), "optimizer": optimizer.state_dict(), "step": update,
                "trace": trace, "train_elapsed": train_elapsed, "val_elapsed": val_elapsed,
                "training_probe_elapsed": probe_elapsed,
                "order": order, "cursor": cursor, "shuffle_state": shuffle.get_state(),
                "noise_state": noise_rng.get_state()})
            progress.update(phase=f"Training {arm}", arm=arm, completed=update, total=steps,
                            arm_index=arm_index, force=True)
            if update == 100 and arm_index == 2:
                passes = math.ceil(steps / validation_every) + 1
                measured_passes = update // validation_every + 1
                estimated_train = train_elapsed / update * steps * len(BACKBONES) * len(MODES)
                estimated_validation = val_elapsed / measured_passes * passes * len(BACKBONES) * len(MODES)
                estimated_probe = probe_elapsed / measured_passes * passes * len(BACKBONES)
                write_json(output / "duration-estimate.json", {
                    "source_plan_id": plan_id, "measured_after_updates": update, "updates_per_second": update / train_elapsed,
                    "arm": arm, "elapsed_train_seconds": train_elapsed,
                    "elapsed_validation_seconds": val_elapsed, "elapsed_training_probe_seconds": probe_elapsed,
                    "estimated_training_seconds": estimated_train, "estimated_validation_seconds": estimated_validation,
                    "estimated_training_probe_seconds": estimated_probe,
                    "estimated_total_seconds": estimated_train + estimated_validation + estimated_probe,
                    "estimate_scope": "complete six-arm run, extrapolated from the first 100 updates and first validation/probe passes",
                    "assumption": "first 100 updates and first two validation/probe passes represent the matching arm groups"})
            progress.update(phase=f"Training {arm}", arm=arm, completed=update, total=steps,
                            arm_index=arm_index, force=update == steps)
        elif update % 5 == 0:
            progress.update(phase=f"Training {arm}", arm=arm, completed=update, total=steps,
                            arm_index=arm_index)
    del optimizer
    model.eval()
    return model, {"arm": arm, "backbone": backbone, "mode": mode, "steps": steps,
                   "seed": seed, "parameters": sum(param.numel() for param in model.parameters()),
                   "trainable_parameters": sum(param.numel() for param in model.parameters() if param.requires_grad),
                    "training_seconds": train_elapsed, "validation_seconds": val_elapsed,
                    "training_probe_seconds": probe_elapsed, "trace": trace}


@torch.no_grad()
def benchmark_latency(model, mode, row, pair_dir, codec, device):
    batch, _ = collate_pairs([row], pair_dir, codec, device, _pair_loader(pair_dir))
    model.eval()
    base_memory = torch.xpu.memory_allocated(device) if device.type == "xpu" else 0
    if mode == "flow":
        mask = batch["mask"][:1]
        noise = _deterministic_noise((1, 32, 2 * FREQUENCY_BINS), SEED + 404, device)
        fn = lambda k: model.sample(batch["controls"][:1], batch["performance"][:1],
                                    batch["times"][:1], batch["lengths"][:1], mask, noise, k)
        counts = FLOW_STEPS
    else:
        fn = lambda _k: model(batch["controls"][:1], batch["performance"][:1], batch["times"][:1], batch["lengths"][:1])
        counts = (1,)
    timing = {}
    for count in counts:
        for _ in range(3):
            fn(count)
        if device.type == "xpu":
            torch.xpu.synchronize(device)
            torch.xpu.reset_peak_memory_stats(device)
        samples = []
        for _ in range(10):
            start = perf_counter()
            fn(count)
            if device.type == "xpu":
                torch.xpu.synchronize(device)
            samples.append((perf_counter() - start) * 1000)
        peak = (torch.xpu.max_memory_allocated(device) - base_memory) if device.type == "xpu" else None
        timing[str(count)] = {"median_ms_per_clip_32_frames": float(np.median(samples)),
                              "p90_ms_per_clip_32_frames": float(np.quantile(samples, .9)),
                              "peak_temporary_bytes": int(max(0, peak)) if peak is not None else None}
    return timing


def model_summary(model, mode):
    model_params = sum(p.numel() for p in model.parameters())
    if mode == "direct":
        extra = {"output_head": "linear 64→2050; positive log1p magnitude", "head_parameters": sum(p.numel() for p in model.output.parameters())}
    else:
        extra = {"velocity_head": "framewise 2050→64 noisy spectrum, time embedding, two residual MLP blocks, 64→2050 clean endpoint minus full-dimensional initial-noise skip",
                 "head_parameters": sum(p.numel() for name, p in model.named_parameters() if not name.startswith("backbone."))}
    return {"parameters": model_params, **extra}


def make_plan(train_run, validation_run, source_plan, validation_plan, rows, val_rows, receipts,
              *, steps, validation_every, seed, device, threads):
    files = {}
    for name in ("flow_matching.py", "surrogate_backbones.py", "surrogate_codec.py", "spectral.py"):
        path = TRAINING / "obruxo_training" / name
        files[str(path.relative_to(TRAINING))] = sha256_file(path)
    script = Path(__file__)
    files[script.name] = sha256_file(script)
    return {
        "format": "issue183_direct_vs_flow_matching_v2", "issue": 183,
        "question": "Does conditional flow matching improve known-patch known-MIDI forward audio prediction, particularly monophonic octave energy and MIDI-aligned envelope fidelity?",
        "sources": {"issue172_run": repo_reference(train_run, "issue172_training_pool"), "issue172_plan_id": source_plan.get("pool_id"),
                    "issue173_validation_run": repo_reference(validation_run, "issue173_validation_panel"), "issue173_validation_plan_id": validation_plan.get("training", {}).get("pilot_plan_id"),
                    "training_pair_count": len(rows), "training_patch_roots": len({r["state_id"] for r in rows}),
                    "validation_pair_count": len(val_rows), "validation_clips": 2 * len(val_rows),
                    "validation_patch_roots": len({r["state_id"] for r in val_rows}),
                    "validation_midi_performances": len({r["midi_id"] for r in val_rows}),
                    "cache_receipt_count": len(receipts)},
        "data": {"training_recipe": "all accepted #172 basic and expanded pairs; basic labels are expanded with canonical default values before use",
                 "validation_recipe": "frozen #173 held-out expanded pairs; exact 2048-point spectral masks and recipe hashes verified",
                 "fft_size": FFT_SIZE, "hop": HOP, "sample_rate": SAMPLE_RATE, "channels": 2,
                 "magnitude_scaling": "abs(stft)/sqrt(2048), physical units; flow state uses log1p(magnitude/1e-4)",
                 "sampled_frames_per_clip": 32, "training_batch_pairs": BATCH_PAIRS, "training_clip_batch": 2 * BATCH_PAIRS},
        "arms": [{"backbone": backbone, "mode": mode, "steps": steps}
                 for backbone in BACKBONES for mode in MODES],
        "training": {"seed": seed, "validation_every_updates": validation_every, "optimizer": "AdamW",
                     "learning_rate": .001, "gradient_clip_norm": 5., "updates_per_arm": steps,
                     "starting_weights": "random initialization; no pretrained or #173 checkpoints loaded",
                     "device": str(device), "threads": threads,
                     "direct_loss": "single-resolution spectral convergence plus log-magnitude L1, equal weights",
                      "flow_loss": "MSE velocity in log1p magnitude space; z(t)=(1-t)e+t*x; velocity is predicted clean endpoint minus full-dimensional initial noise; t shared per clip; e~N(0,I)",
                     "flow_sampler": {"method": "fixed-grid Euler", "steps": list(FLOW_STEPS)},
                     "validation_curve_sampler_steps": 8},
        "evaluation": {"primary": "2048-point spectral convergence plus log-magnitude L1",
                        "octave": "mean absolute dB error in ten fixed one-octave band powers from 21.5 Hz to Nyquist on all non-silent valid frames; report a separate MIDI-centered five-band absolute dB diagnostic on supported monophonic frames",
                       "envelope": "Parseval RMS derived from stereo 2048 STFT; onset <50ms, decay 50–250ms, sustain after 250ms, release first 300ms after MIDI note-off",
                       "uncertainty": "10,000 paired patch-root bootstrap replicates; MIDI performances not resampled",
                       "euler_flow_steps": list(FLOW_STEPS)},
        "code_sha256": files, "data_receipts": "data-receipts.jsonl"}


def run(args):
    args.output.mkdir(parents=True, exist_ok=True)
    progress = Progress(args.output, args.steps)
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    device = torch.device(args.device)
    if device.type == "xpu" and not torch.xpu.is_available():
        raise RuntimeError("the configured PyTorch XPU is unavailable")
    hardware = {"torch": str(torch.__version__), "numpy": np.__version__, "device": str(device),
                "device_name": torch.xpu.get_device_name(device) if device.type == "xpu" else "CPU",
                "threads": args.threads}
    progress.update(phase="Preparing frozen source data", force=True)
    started = perf_counter()
    source_plan, validation_plan, rows, val_rows, receipts = load_sources(
        args.training_source, args.validation_source, args.output, progress)
    codec = SurrogateCodec(PilotCodec(), expanded=True)
    plan = make_plan(args.training_source, args.validation_source, source_plan, validation_plan, rows, val_rows,
                     receipts, steps=args.steps, validation_every=args.validation_every, seed=args.seed,
                     device=device, threads=args.threads)
    plan_id = canonical_sha256(plan)
    cache_receipts_sha256 = receipt_sha256(receipts)
    plan_path = args.output / "plan.json"
    receipts_path = args.output / "data-receipts.jsonl"
    frozen = {"id": plan_id, "plan": plan, "hardware": hardware,
              "data_receipts_sha256": cache_receipts_sha256}
    if plan_path.exists():
        saved = _load_json(plan_path)
        if saved.get("id") != plan_id or saved.get("plan") != plan:
            raise ValueError("existing issue 183 run plan differs; choose a new output directory")
        saved_receipt_sha256 = saved.get("data_receipts_sha256")
        if not saved_receipt_sha256:
            raise ValueError("existing run has no frozen receipt digest; refusing to resume or replace its evidence")
        if (not receipts_path.is_file() or sha256_file(receipts_path) != saved_receipt_sha256
                or cache_receipts_sha256 != saved_receipt_sha256):
            raise ValueError("cache receipts differ from the frozen run; original receipts were left untouched")
    else:
        if receipts_path.exists():
            raise ValueError("receipt file exists without a frozen plan; refusing to replace it")
        write_receipts(receipts_path, receipts)
        write_json(plan_path, frozen)
    if device.type == "xpu":
        torch.xpu.synchronize(device)
    write_json(args.output / "hardware.json", hardware)
    progress.update(phase="Training six matched arms", arm_index=1, force=True)
    arm_summaries, models = {}, {}
    arms = [(backbone, mode) for backbone in BACKBONES for mode in MODES]
    probe_rows = fixed_training_probe(rows)
    for index, (backbone, mode) in enumerate(arms, 1):
        arm = f"{backbone}-{mode}"
        checkpoint = args.output / f"{arm}.pt"
        if checkpoint.exists():
            saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
            if saved["plan_id"] == plan_id and saved["step"] == args.steps:
                model = load_model(backbone, mode, codec.width).to(device)
                model.load_state_dict(saved["model"])
                model.eval()
                models[arm] = model
                arm_summaries[arm] = {"arm": arm, "backbone": backbone, "mode": mode,
                                      "steps": args.steps, "seed": args.seed,
                                      "parameters": sum(p.numel() for p in model.parameters()),
                                      "trace": saved["trace"], "resumed_complete": True}
                progress.update(phase=f"Training {arm}", arm=arm, completed=args.steps, total=args.steps,
                                arm_index=index, force=True)
                continue
        progress.update(phase=f"Training {arm}", arm=arm, completed=0, total=args.steps,
                        arm_index=index, force=True)
        model, summary = fit_arm(args.output, plan_id, backbone, mode, rows, val_rows, codec,
                                 args.training_source / "pairs", args.validation_source / "validation-pairs",
                                 probe_rows, device, progress, index, args.steps, args.validation_every, args.seed)
        summary["model_head"] = model_summary(model, mode)
        models[arm], arm_summaries[arm] = model, summary
        write_json(args.output / "arm-summary.json", arm_summaries)
    source_checkpoint_sha256 = {arm: sha256_file(args.output / f"{arm}.pt") for arm in arm_summaries}
    duration_path = args.output / "duration-estimate.json"
    if duration_path.is_file():
        duration = _load_json(duration_path)
        duration["source_checkpoint_sha256"] = source_checkpoint_sha256
        write_json(duration_path, duration)
    write_json(args.output / "training-summary.json", {
        "plan_id": plan_id, "source_checkpoint_sha256": source_checkpoint_sha256,
        "hardware": hardware, "preparation_seconds": perf_counter() - started,
        "arms": training_summary_public(arm_summaries)})

    full_results, per_case, inference, effects = evaluate_arms(
        models, val_rows, args.validation_source / "validation-pairs", codec, device, args.seed, progress)
    summary = {"plan_id": plan_id, "source_checkpoint_sha256": source_checkpoint_sha256,
               "hardware": hardware, "training": training_summary_public(arm_summaries),
               "evaluation": full_results, "paired_effects": effects, "inference": inference,
               "validation_cases": len(val_rows) * 2, "validation_patch_roots": len({row["state_id"] for row in val_rows}),
               "validation_midi_performances": len({row["midi_id"] for row in val_rows}),
               "evaluation_before_phase_reconstruction": True}
    write_json(args.output / "summary.json", summary)
    write_json(args.output / "per-case-metrics.json", per_case)
    write_learning_curves(args.output, arms, arm_summaries)
    _write_svg(args.output, arms, arm_summaries)
    write_report(args.output, summary, plan, arm_summaries)
    progress.update(phase="Writing listening previews", completed=0, total=10, force=True)
    listening = write_listening_previews(args.output, args.validation_source, val_rows, per_case, models, codec,
                                         args.validation_source / "validation-pairs", device, args.seed,
                                         plan_id, source_checkpoint_sha256)
    summary["listening"] = listening
    write_json(args.output / "summary.json", summary)
    progress.update(phase="Complete", arm="", completed=args.steps, total=args.steps,
                    arm_index=len(arms), state="complete", message="Metrics, report, curves and listening previews saved", force=True)
    print(json.dumps({"plan_id": plan_id, "summary": str(args.output / "summary.json"),
                      "report": str(args.output / "REPORT.md"), "listening": listening}, indent=2), flush=True)


def evaluate_arms(models, val_rows, pair_dir, codec, device, seed, progress):
    arms = [(backbone, mode) for backbone in BACKBONES for mode in MODES]
    full_results, per_case, inference = {}, {}, {}
    for backbone, mode in arms:
        arm = f"{backbone}-{mode}"
        model = models[arm]
        if mode == "direct":
            metrics, cases = evaluate_model(model, mode, val_rows, pair_dir,
                                            codec, device, seed=seed + 90000, progress=progress,
                                            phase=f"Final evaluation · {arm}")
            full_results[arm] = {"metrics": metrics, "sampler_steps": 1}
            per_case[arm] = cases
            inference[arm] = benchmark_latency(model, mode, val_rows[0], pair_dir, codec, device)
        else:
            full_results[arm], per_case[arm] = {}, {}
            for step_count in FLOW_STEPS:
                metrics, cases = evaluate_model(model, mode, val_rows, pair_dir,
                                                codec, device, flow_steps=step_count, seed=seed + 90000,
                                                progress=progress, phase=f"Final evaluation · {arm} · Euler {step_count}")
                full_results[arm][str(step_count)] = metrics
                per_case[arm][str(step_count)] = cases
            inference[arm] = benchmark_latency(model, mode, val_rows[0], pair_dir, codec, device)
    effects = {}
    for backbone in BACKBONES:
        direct = per_case[f"{backbone}-direct"]
        for step_count in FLOW_STEPS:
            flow = per_case[f"{backbone}-flow"][str(step_count)]
            effects[f"{backbone}-flow-{step_count}"] = {
                metric: paired_effect(direct, flow, metric, seed + step_count)
                for metric in ("spectral_loss", "octave_band_energy_error_db", "monophonic_octave_energy_error_db",
                               "attack_envelope_error_db", "decay_envelope_error_db",
                               "sustain_envelope_error_db", "release_envelope_error_db")}
    return full_results, per_case, inference, effects


def verify_completed_run(source, validation_source, expected_arms=None):
    """Check a frozen run, its checkpoints, and the validation cache before reuse."""
    source, validation_source = Path(source), Path(validation_source)
    frozen = _load_json(source / "plan.json")
    plan, plan_id = frozen.get("plan"), frozen.get("id")
    if (not isinstance(plan, dict) or plan.get("format") != "issue183_direct_vs_flow_matching_v2"
            or canonical_sha256(plan) != plan_id):
        raise ValueError("invalid frozen issue 183 training plan")
    for name, digest in plan["code_sha256"].items():
        if Path(name).name == "run_issue183_flow_matching.py":
            continue
        if sha256_file(TRAINING / name.replace(chr(92), "/")) != digest:
            raise ValueError(f"checkpoint dependency changed: {name}")
    expected_receipts_sha256 = frozen.get("data_receipts_sha256")
    receipts_path = source / "data-receipts.jsonl"
    if (not expected_receipts_sha256 or not receipts_path.is_file()
            or sha256_file(receipts_path) != expected_receipts_sha256):
        raise ValueError("completed-run cache receipts differ from the frozen receipt digest")

    validation_frozen = _load_json(validation_source / "plan.json")
    validation_plan = validation_frozen.get("plan", validation_frozen)
    validation_plan_id = validation_plan.get("training", {}).get("pilot_plan_id")
    if plan["sources"].get("issue173_validation_plan_id") != validation_plan_id:
        raise ValueError("validation plan does not belong to the frozen training run")
    validation_rows = validation_plan["evaluation"]["validation_proxy"]["pair_recipes"]
    if len(validation_rows) != plan["sources"]["validation_pair_count"]:
        raise ValueError("validation panel size differs from the frozen training run")
    receipts = [json.loads(line) for line in
                (source / "data-receipts.jsonl").read_text(encoding="utf-8").splitlines() if line]
    receipts = {row["pair_id"]: row for row in receipts
                if row.get("source") == "issue173" and row.get("stage") == "validation"}
    if len(receipts) != len(validation_rows):
        raise ValueError("completed-run validation receipts are incomplete or duplicated")
    for row in validation_rows:
        pair_id = row["pair_id"]
        receipt = receipts.get(pair_id)
        cache = validation_source / "validation-pairs" / f"{pair_id}.pt"
        if (receipt is None or receipt.get("recipe_sha256") != canonical_sha256(row)
                or receipt.get("cache_sha256") != sha256_file(cache)):
            raise ValueError(f"validation cache does not match the completed-run receipt: {pair_id}")

    arms = expected_arms or [f"{row['backbone']}-{row['mode']}" for row in plan["arms"]]
    arm_specs = {f"{row['backbone']}-{row['mode']}": row for row in plan["arms"]}
    seed, steps, checkpoint_hashes = plan["training"]["seed"], plan["training"]["updates_per_arm"], {}
    for arm in arms:
        spec = arm_specs.get(arm)
        if spec is None:
            raise ValueError(f"arm is not included in the frozen plan: {arm}")
        path = source / f"{arm}.pt"
        saved = torch.load(path, map_location="cpu", weights_only=True)
        identity = (saved.get("plan_id"), saved.get("arm"), saved.get("step"), saved.get("seed"),
                    saved.get("backbone"), saved.get("mode"))
        if identity != (plan_id, arm, steps, seed, spec["backbone"], spec["mode"]):
            raise ValueError(f"wrong or incomplete checkpoint: {arm}")
        checkpoint_hashes[arm] = sha256_file(path)
    return {"frozen": frozen, "plan": plan, "plan_id": plan_id,
            "validation_rows": validation_rows, "checkpoint_sha256": checkpoint_hashes}


def reevaluate(args):
    """Evaluate existing completed checkpoints; never train or overwrite the original run."""
    source = args.evaluate_only_from.resolve()
    output = args.output.resolve()
    if source == output or (output.exists() and any(output.iterdir())):
        raise ValueError("evaluation requires a new, separate output directory")
    verified = verify_completed_run(source, args.validation_source)
    frozen, plan, plan_id = verified["frozen"], verified["plan"], verified["plan_id"]
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    if device.type == "xpu" and not torch.xpu.is_available():
        raise RuntimeError("configured XPU unavailable")
    progress = Progress(output, plan["training"]["updates_per_arm"])
    _, _, _, val_rows, receipts = load_sources(args.training_source, args.validation_source, output, progress)
    original_receipts = [json.loads(line) for line in
                         (source / "data-receipts.jsonl").read_text(encoding="utf-8").splitlines() if line]
    if receipts != original_receipts:
        raise ValueError("source cache receipts differ from the completed run")
    codec = SurrogateCodec(PilotCodec(), expanded=True)
    seed, steps = plan["training"]["seed"], plan["training"]["updates_per_arm"]
    models = {}
    checkpoint_hashes = {}
    for backbone in BACKBONES:
        for mode in MODES:
            arm = f"{backbone}-{mode}"
            path = source / f"{arm}.pt"
            saved = torch.load(path, map_location="cpu", weights_only=True)
            if (saved["plan_id"], saved["arm"], saved["step"], saved["seed"]) != (plan_id, arm, steps, seed):
                raise ValueError(f"wrong or incomplete checkpoint: {arm}")
            model = load_model(backbone, mode, codec.width).to(device)
            model.load_state_dict(saved["model"])
            models[arm] = model.eval()
            checkpoint_hashes[arm] = sha256_file(path)
    full_results, per_case, inference, effects = evaluate_arms(
        models, val_rows, args.validation_source / "validation-pairs", codec, device, seed, progress)
    summary = {
        "evaluation_version": "absolute_octave_band_energy_v3",
        "source_plan_id": plan_id, "source_checkpoint_sha256": checkpoint_hashes,
        "evaluator_sha256": sha256_file(Path(__file__)),
        "hardware": frozen.get("hardware", {}), "evaluation": full_results,
        "paired_effects": effects, "inference": inference,
        "validation_cases": len(val_rows) * 2,
        "validation_patch_roots": len({row["state_id"] for row in val_rows}),
        "validation_midi_performances": len({row["midi_id"] for row in val_rows}),
        "device": str(device), "torch": torch.__version__,
        "training_performed": False,
    }
    write_json(output / "summary.json", summary)
    write_json(output / "per-case-metrics.json", per_case)
    progress.update(phase="Corrected evaluation complete", state="complete", force=True)
    print(json.dumps({"summary": str(output / "summary.json"), "training_performed": False}, indent=2))


def training_summary_public(arms):
    result = {}
    for arm, summary in arms.items():
        public = {key: value for key, value in summary.items() if key != "trace"}
        probes = [point["training_probe"] for point in summary.get("trace", [])
                  if point.get("training_probe") is not None]
        if probes:
            public["training_reconstruction_probe"] = {"initial": probes[0], "final": probes[-1]}
        result[arm] = public
    return result


def write_learning_curves(output, arms, summaries):
    rows = []
    for backbone, mode in arms:
        arm = f"{backbone}-{mode}"
        for point in summaries[arm]["trace"]:
            rows.append({"backbone": backbone, "mode": mode, "step": point["step"],
                         "train_objective": point.get("train_objective"),
                         "validation_spectral_loss": point.get("validation_spectral_loss"),
                         "validation_octave_band_energy_error_db": point.get("validation_octave_band_energy_error_db"),
                         "training_probe_velocity_mse": (point.get("training_probe") or {}).get("velocity_mse"),
                         "training_probe_spectral_loss": (point.get("training_probe") or {}).get("sampled_spectral_loss"),
                         "training_probe_octave_band_energy_error_db": (point.get("training_probe") or {}).get(
                             "sampled_octave_band_energy_error_db")})
    with (output / "learning-curves.csv").open("w", encoding="utf-8", newline="") as handle:
        handle.write("backbone,mode,step,train_objective,validation_spectral_loss,validation_octave_band_energy_error_db,training_probe_velocity_mse,training_probe_spectral_loss,training_probe_octave_band_energy_error_db\n")
        for row in rows:
            values = [row[key] for key in ("backbone", "mode", "step", "train_objective",
                                           "validation_spectral_loss", "validation_octave_band_energy_error_db",
                                           "training_probe_velocity_mse", "training_probe_spectral_loss",
                                           "training_probe_octave_band_energy_error_db")]
            handle.write(",".join("" if value is None else str(value) for value in values) + "\n")


def _write_primary_comparison_chart(output, effects, preview_path=None):
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    labels = [BACKBONE_LABELS[backbone] for backbone in BACKBONES]
    rows = np.arange(len(BACKBONES))
    panels = (
        ("spectral_loss", "Spectral loss difference", "Δ spectral loss", 0.30, ".3f"),
        ("octave_band_energy_error_db", "Octave-band energy difference", "Δ octave-band error (dB)", 5.5, ".2f"),
    )
    figure, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    figure.suptitle("At 8 Euler steps, flow has higher validation error", fontsize=13, fontweight="bold")
    for axis, (metric, title, xlabel, xmax, number_format) in zip(axes, panels, strict=True):
        means, lows, highs = [], [], []
        for backbone in BACKBONES:
            effect = effects[backbone][metric]
            means.append(effect["flow_minus_direct"])
            lows.append(effect["ci95"][0])
            highs.append(effect["ci95"][1])
        means, lows, highs = map(np.asarray, (means, lows, highs))
        axis.errorbar(means, rows, xerr=np.vstack((means - lows, highs - means)), fmt="o",
                      color="#0072B2", ecolor="#0072B2", capsize=3.5, markersize=6, linewidth=1.8)
        for y, mean, low, high in zip(rows, means, lows, highs, strict=True):
            text = f"{mean:+{number_format}} [{low:+{number_format}}, {high:+{number_format}}]"
            axis.annotate(text, xy=(high, y), xytext=(7, 0), textcoords="offset points",
                          va="center", fontsize=8, color="#333333")
        axis.axvline(0, color="#555555", linewidth=1, linestyle="--")
        axis.set_xlim(-0.02 if metric == "spectral_loss" else -0.25, xmax)
        axis.set_ylim(len(rows) - 0.5, -0.5)
        axis.set_yticks(rows)
        if axis is axes[0]:
            axis.set_yticklabels(labels)
        else:
            axis.tick_params(labelleft=False)
        axis.set_title(title, fontsize=10)
        axis.set_xlabel(xlabel)
        axis.grid(axis="x", color="#dddddd", linewidth=0.7)
        axis.grid(axis="y", visible=False)
    axes[0].set_ylabel("Backbone")
    figure.text(0.5, 0.02, "Flow minus direct; positive means higher error for flow. Bars show 95% patch-root bootstrap intervals.",
                ha="center", fontsize=9)
    figure.tight_layout(rect=(0, 0.06, 1, 0.90), w_pad=2.4)
    chart_path = output / "primary-comparison.svg"
    figure.savefig(chart_path, format="svg", metadata={"Date": None})
    chart_path.write_text("\n".join(line.rstrip() for line in chart_path.read_text(encoding="utf-8").splitlines()) + "\n",
                          encoding="utf-8")
    if preview_path is not None:
        figure.savefig(preview_path, format="png", dpi=150)
    plt.close(figure)


def _write_sampler_quality_chart(output, result):
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    metrics = (
        ("spectral_loss", "Spectral loss", "error difference"),
        ("octave_band_energy_error_db", "General octave-band error", "dB difference"),
        ("monophonic_octave_energy_error_db", "Monophonic octave error", "dB difference"),
        ("attack_envelope_error_db", "Attack envelope error", "dB difference"),
        ("decay_envelope_error_db", "Decay envelope error", "dB difference"),
        ("sustain_envelope_error_db", "Sustain envelope error", "dB difference"),
        ("release_envelope_error_db", "Release envelope error", "dB difference"),
    )
    steps = FLOW_STEPS
    colors = {"residual_mlp": "#0072B2", "transformer": "#D55E00", "conformer": "#009E73"}
    figure, axes = plt.subplots(len(metrics), 1, figsize=(9.2, 14.5), sharex=True)
    figure.suptitle("Flow minus direct quality by Euler sampling steps", fontsize=14, fontweight="bold")
    for axis, (metric, title, unit) in zip(axes, metrics, strict=True):
        for backbone in BACKBONES:
            effects = [result["paired_effects"][f"{backbone}-flow-{step}"][metric] for step in steps]
            means = np.asarray([row["flow_minus_direct"] for row in effects])
            lows = np.asarray([row["ci95"][0] for row in effects])
            highs = np.asarray([row["ci95"][1] for row in effects])
            axis.errorbar(steps, means, yerr=np.vstack((means - lows, highs - means)), fmt="o-",
                          capsize=3, markersize=4.5, linewidth=1.5, color=colors[backbone],
                          label=BACKBONE_LABELS[backbone])
        axis.axhline(0, color="#555555", linewidth=0.9, linestyle="--")
        axis.set_ylabel(unit, fontsize=8)
        axis.set_title(title, fontsize=9, loc="left")
        axis.grid(axis="y", color="#dddddd", linewidth=0.65)
        axis.set_xticks(steps)
    axes[0].legend(ncol=3, frameon=False, fontsize=8, loc="upper right")
    axes[-1].set_xlabel("Flow Euler steps (direct baseline is one pass)")
    figure.text(0.015, 0.5, "Flow − direct; positive means higher error for flow. Error bars are 95% patch-root bootstrap intervals.",
                rotation="vertical", va="center", fontsize=8)
    figure.tight_layout(rect=(0.05, 0.01, 1, 0.97), h_pad=0.55)
    path = output / "sampler-quality-effects.svg"
    figure.savefig(path, format="svg", metadata={"Date": None})
    path.write_text("\n".join(line.rstrip() for line in path.read_text(encoding="utf-8").splitlines()) + "\n",
                    encoding="utf-8")
    plt.close(figure)


def _write_coverage_chart(output, metrics):
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    total = metrics["octave_band_total_frames"]
    labels = ["Non-silent octave", "Active monophonic", "Attack", "Decay", "Sustain", "Release"]
    values = [metrics["octave_band_frames"], metrics["monophonic_frames"],
              *(metrics[f"{phase}_frames"] for phase in ("attack", "decay", "sustain", "release"))]
    denominators = [total, metrics["monophonic_frames_total"], *(total for _ in range(4))]
    y = np.arange(len(labels))
    figure, axis = plt.subplots(figsize=(8.5, 3.4))
    axis.barh(y, denominators, color="#e8edf2", height=0.62, label="Candidate frames")
    axis.barh(y, values, color="#0072B2", height=0.42, label="Scored frames")
    for row, (value, denominator) in enumerate(zip(values, denominators, strict=True)):
        axis.text(denominator + max(denominators) * .012, row, f"{value:,} / {denominator:,}",
                  va="center", fontsize=8)
    axis.set_yticks(y, labels)
    axis.invert_yaxis()
    axis.set_xlim(0, max(denominators) * 1.25)
    axis.set_xlabel("Frames across the frozen validation panel")
    axis.set_title("Coverage of the octave and envelope diagnostics", loc="left", fontsize=11, fontweight="bold")
    axis.grid(axis="x", color="#dddddd", linewidth=0.65)
    axis.set_axisbelow(True)
    axis.spines[["top", "right", "left"]].set_visible(False)
    axis.legend(frameon=False, ncol=2, fontsize=8, loc="lower right")
    figure.tight_layout()
    path = output / "diagnostic-coverage.svg"
    figure.savefig(path, format="svg", metadata={"Date": None})
    path.write_text("\n".join(line.rstrip() for line in path.read_text(encoding="utf-8").splitlines()) + "\n",
                    encoding="utf-8")
    plt.close(figure)


def _write_inference_cost_chart(output, result):
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    steps = (1, *FLOW_STEPS)
    colors = {"residual_mlp": "#0072B2", "transformer": "#D55E00", "conformer": "#009E73"}
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.1))
    for backbone in BACKBONES:
        latency = [result["inference"][f"{backbone}-direct"]["1"]["median_ms_per_clip_32_frames"]]
        memory = [result["inference"][f"{backbone}-direct"]["1"]["peak_temporary_bytes"] / (1024 ** 2)]
        for step in FLOW_STEPS:
            row = result["inference"][f"{backbone}-flow"][str(step)]
            latency.append(row["median_ms_per_clip_32_frames"])
            memory.append(row["peak_temporary_bytes"] / (1024 ** 2))
        axes[0].plot(steps, latency, marker="o", linewidth=1.8, color=colors[backbone], label=BACKBONE_LABELS[backbone])
        axes[1].plot(steps, memory, marker="o", linewidth=1.8, color=colors[backbone], label=BACKBONE_LABELS[backbone])
    for axis, title, ylabel in ((axes[0], "Median inference latency", "ms per 32-frame clip"),
                                (axes[1], "Peak temporary memory", "MiB above loaded model")):
        axis.set_title(title, fontsize=10)
        axis.set_xlabel("Model evaluations (direct = 1; flow = Euler steps)")
        axis.set_ylabel(ylabel)
        axis.set_xticks(steps)
        axis.grid(axis="y", color="#dddddd", linewidth=0.65)
    axes[0].legend(frameon=False, fontsize=8)
    figure.suptitle("Inference cost across tested sampler steps", fontsize=13, fontweight="bold")
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    path = output / "inference-cost.svg"
    figure.savefig(path, format="svg", metadata={"Date": None})
    path.write_text("\n".join(line.rstrip() for line in path.read_text(encoding="utf-8").splitlines()) + "\n",
                    encoding="utf-8")
    plt.close(figure)


def write_report(output, result, plan, arm_summaries):
    evaluation = result["evaluation"]
    direct_metrics = evaluation[f"{BACKBONES[0]}-direct"]["metrics"]
    flow8_effects = {backbone: result["paired_effects"][f"{backbone}-flow-8"] for backbone in BACKBONES}
    output.mkdir(parents=True, exist_ok=True)
    _write_primary_comparison_chart(output, flow8_effects)
    _write_sampler_quality_chart(output, result)
    _write_coverage_chart(output, direct_metrics)
    _write_inference_cost_chart(output, result)
    lines = [
        "# Issue 183: Direct spectral prediction vs conditional flow matching", "",
        "We compared direct prediction with conditional flow matching at 4, 8 and 16 Euler steps for stereo Vital-spectrum prediction. At 8 steps, direct prediction had lower spectral and general octave-band error for all three backbones, so it remains the evidence-supported baseline for this panel. The single-seed result is exploratory and does not establish a best architecture overall.", "",
        "Spectral loss averages relative spectral-shape error and log-amplitude error. Octave-band error measures mean absolute dB error in fixed one-octave band powers from 21.5 Hz to Nyquist. Lower is better for both; positive flow-minus-direct differences indicate worse flow error.", "",
        f"The frozen validation panel contains **{result['validation_cases'] // 2} paired recipes ({result['validation_cases']} clips), {direct_metrics['patch_roots']} patch roots, and {result['validation_midi_performances']} distinct MIDI performances**. Each recipe pairs original and perturbed clips.", "",
        "## Primary comparison", "",
        "The chart summarizes paired flow-minus-direct effects. Intervals resample patch roots and keep the MIDI-performance set fixed.", "",
        "![Eight-step flow-minus-direct effects for spectral and general octave-band error.](primary-comparison.svg)", "",
        "| Backbone | Spectral loss (Direct → Flow-8) | Flow − Direct (95% CI) | Octave error dB (Direct → Flow-8) | Flow − Direct (95% CI) |", "|---|---:|---:|---:|---:|"]
    for backbone in BACKBONES:
        direct = evaluation[f"{backbone}-direct"]["metrics"]
        flow = evaluation[f"{backbone}-flow"]["8"]
        spectral_effect = flow8_effects[backbone]["spectral_loss"]
        octave_effect = flow8_effects[backbone]["octave_band_energy_error_db"]
        lines.append(f"| {BACKBONE_LABELS[backbone]} | {direct['spectral_loss']:.4f} → {flow['spectral_loss']:.4f} | {_effect_text(spectral_effect)} | {direct['octave_band_energy_error_db']:.2f} → {flow['octave_band_energy_error_db']:.2f} | {_effect_text(octave_effect)} |")
    octave_coverage = direct_metrics["octave_band_frames"]
    octave_total = direct_metrics["octave_band_total_frames"]
    mono_coverage = direct_metrics["monophonic_frames"]
    mono_total = direct_metrics["monophonic_frames_total"]
    phase_names = ("attack", "decay", "sustain", "release")
    phase_ranges = {}
    for phase in phase_names:
        differences = [flow8_effects[backbone][f"{phase}_envelope_error_db"]["flow_minus_direct"]
                      for backbone in BACKBONES]
        phase_ranges[phase] = f"{min(differences):+.2f} to {max(differences):+.2f} dB"
    lines += ["", "### Results across sampler steps", "",
              "![Flow-minus-direct paired effects for all seven quality metrics at 4, 8 and 16 steps.](sampler-quality-effects.svg)", "",
              "This chart includes spectral loss, general and monophonic octave error, and attack, decay, sustain and release envelope error. Intervals resample patch roots only; higher positive values mean larger flow error. At 8 steps, the envelope differences were attack "
              + phase_ranges["attack"] + ", decay " + phase_ranges["decay"] + ", sustain " + phase_ranges["sustain"]
              + ", and release " + phase_ranges["release"] + ".", "",
              "The octave diagnostic scores absolute one-octave band-power error. Envelope error is mean absolute dB error in spectra-derived RMS envelopes during each note phase.", "",
              "### Diagnostic coverage", "",
              "![Frame coverage for the octave and envelope diagnostics.](diagnostic-coverage.svg)", "",
              f"General octave scoring used {octave_coverage:,}/{octave_total:,} non-silent candidate frames; {octave_total - octave_coverage:,} silent frames were excluded. The MIDI-centered diagnostic covered {mono_coverage:,}/{mono_total:,} active monophonic frames. Envelope-phase counts are shown against the {octave_total:,} masked candidate frames.", "",
              "## Uncertainty, model scope, and missing evidence", "",
              f"The 95% intervals resample the {direct_metrics['patch_roots']} patch roots, keeping clips from each root together. They do not resample the {result['validation_midi_performances']} MIDI performances or training runs; there was one training seed. Treat these as exploratory panel-level comparisons, not independent confirmation or an estimate across performances and seeds.", "",
              "Pre-fix v1 estimates are excluded: that flow model used a restricted velocity head and the earlier mono-only octave score. This report shows the corrected v2 model and general absolute octave-band error.", "",
              "This experiment shows how these six trained arms performed on this frozen panel. It does not establish an overall best architecture, generalization to new performances, or end-to-end patch reconstruction.", "",
              "### Vital control scope", "",
              "No Vital controls are predicted or changed. All arms receive the same 21-control Vital codec and known MIDI timing as conditioning inputs, then predict a stereo magnitude spectrogram (2 × 1,025 frequency bins). The [canonical frozen-control inventory](../../../modelling/frozen_encoder_retrieval/reports/vital_control_inventory.jsonl) describes the broader fixed-control scope.", "",
              "### Missing evidence", "",
              "- Only Residual MLP, Transformer, and Conformer backbones and 4-, 8-, and 16-step Euler flow samplers were tested; no additional architectures or solvers were evaluated.",
              "- No independent training seed or fresh patch-root and MIDI-performance confirmation.",
              "- The reserved test set and transcribed-MIDI evaluation were not run.",
              "- No subjective listening ratings, phase-aware prediction, waveform prediction, or downstream patch reconstruction were measured.", "",
              "## Additional diagnostics", "",
              "These analyses add context but do not increase the independent validation sample.", "",
              f"Among direct arms, Residual MLP had the lowest mean spectral loss on this panel ({evaluation['residual_mlp-direct']['metrics']['spectral_loss']:.4f}; Transformer {evaluation['transformer-direct']['metrics']['spectral_loss']:.4f}; Conformer {evaluation['conformer-direct']['metrics']['spectral_loss']:.4f}). Patch-root intervals for these contrasts are in [direct-backbone-comparison.json](direct-backbone-comparison.json).", ""]
    step_rises = []
    for backbone in BACKBONES:
        first = evaluation[f"{backbone}-flow"]["4"]["spectral_loss"]
        last = evaluation[f"{backbone}-flow"]["16"]["spectral_loss"]
        step_rises.append(f"{BACKBONE_LABELS[backbone]} +{last - first:.4f}")
    lines.append(f"Increasing flow steps from 4 to 16 raised spectral loss for all three backbones ({'; '.join(step_rises)}). Exact values are in [summary.json](summary.json).")
    noise = result.get("noise_sensitivity")
    if noise:
        ranges = []
        for backbone in noise["finalists"]:
            arm = f"{backbone}-flow"
            row = noise["seed_ranges"][arm]["4"]["spectral_loss"]
            ranges.append(f"{BACKBONE_LABELS[backbone]} {row['min']:.4f}–{row['max']:.4f}")
        lines += [f"Three fixed Gaussian starts changed 4-step spectral loss only slightly for the finalists ({'; '.join(ranges)}). They were selected on this same validation panel, so this probes initialization sensitivity, not new patches or performances. See [noise-sensitivity.json](noise-sensitivity.json).", ""]
    probe_info = result.get("training", {}).get("residual_mlp-flow", {}).get("training_reconstruction_probe")
    probe_noise = result.get("training_probe_noise_sensitivity", {}).get("arms", {})
    if probe_info:
        initial_velocity = probe_info["initial"]["velocity_mse"]
        residual_noise = probe_noise.get("residual_mlp-flow", {}).get("velocity_mse")
        final_velocity = residual_noise["mean"] if residual_noise else probe_info["final"]["velocity_mse"]
        sampled = []
        for backbone in BACKBONES:
            arm_probe = result.get("training", {}).get(f"{backbone}-flow", {}).get("training_reconstruction_probe", {})
            noise_result = probe_noise.get(f"{backbone}-flow", {}).get("sampled_spectral_loss")
            final_loss = noise_result["mean"] if noise_result else arm_probe.get("final", {}).get("sampled_spectral_loss")
            if final_loss is not None:
                sampled.append(f"{BACKBONE_LABELS[backbone]} {final_loss:.4f}")
        lines += ["### Fixed training-subset fit", "",
                  f"On a fixed probe of {probe_info['initial']['recipes']} training recipes ({probe_info['initial']['clips']} clips, {probe_info['initial']['patch_roots']} roots), Residual MLP flow velocity MSE moved from {initial_velocity:.2f} at initialization to {final_velocity:.2f} averaged across {len(TRAINING_PROBE_NOISE_SEEDS)} Gaussian starts. Final sampled spectral losses averaged "
                  + ", ".join(sampled) + ". This shows fit on the probe only, not validation generalization; see [training-probe-noise-sensitivity.json](training-probe-noise-sensitivity.json).", ""]
    lines += ["", "### Learning curves", "",
              "Validation loss uses the 8-step Euler sampler for flow. Training objectives differ across methods, so compare them only within the same objective.", "",
              "![Validation learning curves for the six model arms.](validation-loss-curves.svg)", "",
              "[Learning curves and fixed-noise training probes](learning-curves.csv)", "",
              "## Execution and reproducibility", "",
              f"All {len(plan['arms'])} arms started from random initialization and ran {plan['training']['updates_per_arm']} updates with matched data order, batch size, optimizer, and validation cadence. Training used {plan['sources']['training_pair_count']:,} accepted pairs across {plan['sources']['training_patch_roots']} roots. The reserved test set was not used.", "",
              f"Device: {result['hardware']['device_name']} ({result['hardware']['device']}); PyTorch {result['hardware']['torch']}. The flow velocity is a predicted clean endpoint minus the full-dimensional initial-noise residual. Denoising blocks process frames independently, so the sampled training frames need not be evenly spaced.", "",
              "Ten frozen #173 examples were exported as listening previews with deterministic 48-iteration Griffin–Lim. Their case IDs, source manifest, checkpoint hashes and asset receipts are recorded in summary.json; the audio stays with the local run. Audio is peak-normalized for audition and was not used for scoring.", "",
              "Trainable parameters were: " + "; ".join(
                  f"{BACKBONE_LABELS[backbone]} {result['training'][f'{backbone}-direct']['trainable_parameters']:,} direct vs {result['training'][f'{backbone}-flow']['trainable_parameters']:,} flow"
                  for backbone in BACKBONES) + ".", "",
              "### Inference cost", "",
              "Latency includes conditioning and prediction for one 32-frame clip. Peak temporary memory is above the loaded-model baseline.", "",
              "![Median latency and peak temporary memory across direct and 4-, 8- and 16-step flow inference.](inference-cost.svg)", ""]
    throughput = result.get("throughput")
    if throughput:
        seconds = throughput["estimated_total_seconds"]
        lines += [f"The first 100 updates measured {throughput['updates_per_second']:.1f} updates/s. Extrapolating those measurements estimates {seconds / 60:.1f} minutes for the complete six-arm training, validation and probe run; this is a planning estimate, not a new runtime measurement. See [duration-estimate.json](duration-estimate.json).", ""]
    lines += ["### Artifacts", "",
              "[Summary](summary.json) · [Per-case metrics](per-case-metrics.json) · [Patch-root effects](direct-backbone-comparison.json) · [Sampler noise sensitivity](noise-sensitivity.json) · [Training probe noise sensitivity](training-probe-noise-sensitivity.json) · [Duration estimate](duration-estimate.json) · [Frozen plan](plan.json) · [Cache receipts](data-receipts.jsonl)", "",
              "## Recommendation", "",
              "Keep direct prediction as the baseline. Decide whether flow matching merits a longer run only after a matched second seed on fresh patch roots and MIDI performances, followed by the reserved test and downstream reconstruction evaluation.", ""]
    (output / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def _effect_text(effect):
    if effect["flow_minus_direct"] is None:
        return "—"
    low, high = effect["ci95"]
    return f"{effect['flow_minus_direct']:+.4f} [{low:+.4f}, {high:+.4f}]"


def _write_svg(output, arms, summaries):
    width, height, panel_height = 900, 660, 190
    colors = {"direct": "#276EF1", "flow": "#D55E00"}
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
             '<rect width="100%" height="100%" fill="white"/>',
             '<text x="24" y="28" font-family="Segoe UI, sans-serif" font-size="18" font-weight="bold">Issue 183 validation loss · 2048-point spectral metric</text>']
    for panel, backbone in enumerate(BACKBONES):
        top = 54 + panel * panel_height
        parts += [f'<text x="24" y="{top + 18}" font-family="Segoe UI, sans-serif" font-size="14" font-weight="bold">{backbone}</text>',
                  f'<path d="M 74 {top + 152} H 860 M 74 {top + 40} V {top + 152}" stroke="#666" fill="none"/>',
                  f'<text x="460" y="{top + 178}" font-family="Segoe UI, sans-serif" font-size="11">Training updates</text>']
        all_values = []
        for mode in MODES:
            trace = summaries[f"{backbone}-{mode}"]["trace"]
            all_values.extend(row["validation_spectral_loss"] for row in trace if row.get("validation_spectral_loss") is not None)
        if not all_values:
            continue
        low, high = min(all_values), max(all_values)
        padding = max(.025, (high - low) * .12)
        low, high = low - padding, high + padding
        parts.append(f'<text x="26" y="{top + 98}" transform="rotate(-90 26 {top + 98})" font-family="Segoe UI, sans-serif" font-size="11">Spectral loss</text>')
        for mode in MODES:
            trace = [row for row in summaries[f"{backbone}-{mode}"]["trace"] if row.get("validation_spectral_loss") is not None]
            points = []
            for row in trace:
                x = 74 + 786 * row["step"] / max(1, summaries[f"{backbone}-{mode}"]["steps"])
                y = top + 152 - 112 * (row["validation_spectral_loss"] - low) / (high - low)
                points.append(f"{x:.1f},{y:.1f}")
            parts.append(f'<polyline points="{" ".join(points)}" fill="none" stroke="{colors[mode]}" stroke-width="2.4"/>')
            for row, point in zip(trace, points, strict=True):
                x, y = map(float, point.split(","))
                parts.append(f'<circle cx="{x}" cy="{y}" r="3.1" fill="{colors[mode]}"/>')
        parts += [f'<text x="690" y="{top + 22}" fill="{colors["direct"]}" font-family="Segoe UI, sans-serif" font-size="12">Direct</text>',
                  f'<text x="760" y="{top + 22}" fill="{colors["flow"]}" font-family="Segoe UI, sans-serif" font-size="12">Flow · Euler 8</text>']
        parts += [f'<text x="40" y="{top + 155}" font-family="Segoe UI, sans-serif" font-size="10">0</text>',
                  f'<text x="835" y="{top + 155}" font-family="Segoe UI, sans-serif" font-size="10">500</text>',
                  f'<text x="40" y="{top + 43}" font-family="Segoe UI, sans-serif" font-size="10">{high:.3f}</text>',
                  f'<text x="40" y="{top + 151}" font-family="Segoe UI, sans-serif" font-size="10">{low:.3f}</text>']
    parts.append("</svg>\n")
    (output / "validation-loss-curves.svg").write_text("\n".join(parts), encoding="utf-8")


def _asset_receipt(output, path, role, case_id, source_path=None):
    return {"role": role, "case_id": case_id, "path": Path(path).relative_to(output).as_posix(),
            "sha256": sha256_file(path), "bytes": Path(path).stat().st_size,
            "source_sha256": sha256_file(source_path) if source_path is not None else None}


def write_listening_previews(output, validation_source, val_rows, per_case, models, codec, pair_dir, device, seed,
                             experiment_plan_id, checkpoint_sha256):
    """Replay the exact ten #173 holdout inputs and export model outputs for grading."""
    from scipy.io import wavfile
    from obruxo_data.midi import Performance
    from obruxo_training.proxy import performance_inputs
    from obruxo_training.spectral import SpectralTargets
    from obruxo_training.temporal_proxy import full_times

    output, validation_source = Path(output), Path(validation_source).resolve()
    listening = output / "listening"
    audio_dir = listening / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    validation_document = _load_json(validation_source / "plan.json")
    source_plan = validation_document.get("plan", validation_document)
    validation_plan_id = validation_document.get("id") or source_plan.get("training", {}).get("pilot_plan_id")
    manifest_path = validation_source.parent / "issue173-listening-test-20261127" / "manifest.json"
    old_listening = manifest_path.parent
    manifest = _load_json(manifest_path)
    planned = {row["case_id"]: row for row in source_plan["evaluation"]["cases"]}
    cases, asset_receipts, sample_rate = [], [], SAMPLE_RATE
    names = [f"{backbone}-direct" for backbone in BACKBONES] + [f"{backbone}-flow-16" for backbone in BACKBONES]
    if len(manifest["cases"]) != 10:
        raise ValueError("the frozen #173 listening packet must contain ten examples")
    for example, entry in enumerate(manifest["cases"], 1):
        case = planned[entry["case_id"]]
        source_audio_path = old_listening / "audio" / f"sample-{example}-target.wav"
        rate, source_audio = wavfile.read(source_audio_path)
        if rate != SAMPLE_RATE or source_audio.dtype != np.float32 or source_audio.ndim != 2 or source_audio.shape[1] != 2:
            raise ValueError(f"unexpected source audio for #173 listening example {example}")
        measured = SpectralTargets()(torch.from_numpy(source_audio.T.copy())[None])
        times = tuple(value.to(device) for value in full_times(measured))
        controls = control_tensor(codec, [PilotTarget.from_dict(case["target"])], device)
        performance = performance_inputs(Performance.from_dict(case["performance"]))[None].to(device)
        lengths = measured["lengths"].to(device)
        frame_mask = measured["masks"][1].to(device)
        length = int(measured["lengths"][0])
        predicted = {}
        for model_index, backbone in enumerate(BACKBONES):
            direct = models[f"{backbone}-direct"]
            with torch.no_grad():
                direct_output = as_spectrogram(direct(controls, performance, times[1], lengths))[0].cpu()
            predicted[f"{backbone}-direct"] = direct_output
            flow = models[f"{backbone}-flow"]
            noise = _deterministic_noise((1, times[1].shape[1], 2 * FREQUENCY_BINS),
                                         seed + example, device)
            with torch.no_grad():
                flow_output = as_spectrogram(flow.sample(controls, performance, times[1], lengths,
                                                         frame_mask, noise, 16))[0].cpu()
            predicted[f"{backbone}-flow-16"] = flow_output
        exact_files, audio_files = {}, {}
        target_path = audio_dir / f"example-{example:02d}-target.wav"
        target_audio = source_audio.astype(np.float32, copy=True)
        target_audio *= np.float32((10 ** (-1 / 20)) / max(1e-8, float(np.abs(target_audio).max())))
        wavfile.write(target_path, sample_rate, target_audio)
        audio_files["target"] = f"audio/{target_path.name}"
        asset_receipts.append(_asset_receipt(output, target_path, "target_audition", entry["case_id"], source_audio_path))
        for model_index, arm in enumerate(names):
            magnitude = predicted[arm].numpy()
            npz_path = audio_dir / f"example-{example:02d}-{arm}.npz"
            np.savez_compressed(npz_path, magnitude_2048=magnitude, sample_rate=SAMPLE_RATE,
                                fft_size=FFT_SIZE, hop=HOP, patch_root=entry["root"], case_id=entry["case_id"])
            exact_files[arm] = f"audio/{npz_path.name}"
            asset_receipts.append(_asset_receipt(output, npz_path, "predicted_spectrum", entry["case_id"]))
            channels = [griffin_lim(torch.from_numpy(magnitude[channel]), length,
                                    173000 + 100 * example + model_index + channel, iterations=48)
                        for channel in range(2)]
            preview = np.stack(channels, axis=1).astype(np.float32, copy=False)
            preview *= np.float32((10 ** (-1 / 20)) / max(1e-8, float(np.abs(preview).max())))
            path = audio_dir / f"example-{example:02d}-{arm}.wav"
            wavfile.write(path, sample_rate, preview)
            audio_files[arm] = f"audio/{path.name}"
            asset_receipts.append(_asset_receipt(output, path, "listening_preview", entry["case_id"]))
        cases.append({"example": example, "case_id": entry["case_id"], "patch_root": entry["root"],
                      "audio": audio_files, "exact_outputs": exact_files})
    payload = {"title": "Issue 183 · Flow matching previews", "models": names, "cases": cases}
    payload["source_plan_id"] = validation_plan_id
    payload["source_manifest_sha256"] = sha256_file(manifest_path)
    payload["asset_receipts"] = asset_receipts
    write_json(listening / "packet.json", payload)
    (listening / "index.html").write_text(listening_html(payload), encoding="utf-8")
    return {"examples": len(cases), "available": True,
            "path": repo_reference(listening / "index.html", "issue183_listening_previews"),
            "experiment_plan_id": experiment_plan_id, "source_plan_id": validation_plan_id,
            "source_manifest_sha256": payload["source_manifest_sha256"],
            "source_case_ids": [case["case_id"] for case in cases],
            "source_case_ids_sha256": canonical_sha256([case["case_id"] for case in cases]),
            "source_checkpoint_sha256": checkpoint_sha256, "asset_receipts": asset_receipts,
            "griffin_lim_iterations": 48,
            "normalization": "each stereo clip peak-normalized to -1 dBFS for audition only"}


def griffin_lim(magnitude, length, seed, iterations=48):
    magnitude = magnitude.cpu().float()
    magnitude = magnitude * math.sqrt(FFT_SIZE)
    window = torch.hann_window(FFT_SIZE)
    generator = cpu_generator(seed)
    spectrum = magnitude * torch.exp(2j * torch.pi * torch.rand(magnitude.shape, generator=generator))
    audio = torch.istft(spectrum, FFT_SIZE, hop_length=HOP, window=window, center=True, length=length)
    for _ in range(iterations):
        estimate = torch.stft(audio, FFT_SIZE, hop_length=HOP, window=window, center=True,
                              pad_mode="constant", return_complex=True)
        spectrum = magnitude * estimate / estimate.abs().clamp_min(1e-8)
        audio = torch.istft(spectrum, FFT_SIZE, hop_length=HOP, window=window, center=True, length=length)
    return audio.numpy()


def export_listening_from_run(source, validation_source, device, threads):
    """Export listening previews from a completed run without retraining or reevaluating."""
    source, validation_source = Path(source).resolve(), Path(validation_source).resolve()
    verified = verify_completed_run(source, validation_source)
    summary = _load_json(source / "summary.json")
    if summary.get("plan_id", summary.get("source_plan_id")) != verified["plan_id"]:
        raise ValueError("run summary belongs to a different frozen training plan")
    saved_hashes = summary.get("source_checkpoint_sha256")
    if saved_hashes and saved_hashes != verified["checkpoint_sha256"]:
        raise ValueError("run summary contains checkpoint hashes from another experiment")
    summary["source_checkpoint_sha256"] = verified["checkpoint_sha256"]
    torch.set_num_threads(threads)
    if device.type == "xpu" and not torch.xpu.is_available():
        raise RuntimeError("configured XPU unavailable")
    codec = SurrogateCodec(PilotCodec(), expanded=True)
    models = {}
    for spec in verified["plan"]["arms"]:
        arm = f"{spec['backbone']}-{spec['mode']}"
        saved = torch.load(source / f"{arm}.pt", map_location="cpu", weights_only=True)
        model = load_model(spec["backbone"], spec["mode"], codec.width).to(device)
        model.load_state_dict(saved["model"])
        models[arm] = model.eval()
    progress = Progress(source, verified["plan"]["training"]["updates_per_arm"])
    progress.update(phase="Writing listening previews", completed=0, total=10, force=True)
    listening = write_listening_previews(source, validation_source, verified["validation_rows"], None,
                                         models, codec, validation_source / "validation-pairs", device,
                                         verified["plan"]["training"]["seed"], verified["plan_id"],
                                         verified["checkpoint_sha256"])
    summary["listening"] = listening
    write_json(source / "summary.json", summary)
    progress.update(phase="Complete", arm="", completed=verified["plan"]["training"]["updates_per_arm"],
                    total=verified["plan"]["training"]["updates_per_arm"],
                    arm_index=len(verified["plan"]["arms"]), state="complete",
                    message="Metrics, report, curves and listening previews saved", force=True)
    return listening


def listening_html(payload):
    rows = []
    for case in payload["cases"]:
        cells = [f'<td><b>Target</b><audio controls preload="none" src="{case["audio"]["target"]}"></audio></td>']
        for name in payload["models"]:
            key = f'{case["example"]}-{name}'
            cells.append(f'<td><b>{name}</b> · <a href="{case["exact_outputs"][name]}">exact predicted spectrum</a>'
                         f'<audio controls preload="none" src="{case["audio"][name]}"></audio><label>Grade '
                         f'<select data-rating="{key}"><option value="">—</option>'
                         + "".join(f'<option value="{score}">{score}</option>' for score in range(1, 6))
                         + "</select></label></td>")
        cells.append(f'<td><textarea data-comment="{case["example"]}-comment" placeholder="Overall feedback for this example"></textarea></td>')
        rows.append(f'<tr><th>Example {case["example"]}<br><small>{case["case_id"]}</small></th>{"".join(cells)}</tr>')
    headers = "".join(f"<th>{name}</th>" for name in payload["models"])
    return """<!doctype html><meta charset="utf-8"><title>Issue 183 listening table</title>
<style>body{font:14px Segoe UI,Arial;margin:24px;color:#222}table{border-collapse:collapse;width:100%;min-width:1500px}th,td{border:1px solid #bbb;padding:8px;vertical-align:top}thead{position:sticky;top:0;background:#f3f5f7}audio{width:190px;display:block;margin:6px 0}textarea{min-width:190px;min-height:58px}select{min-width:72px}button{font:inherit;padding:7px}</style>
<h1>Issue 183 · Direct prediction vs flow matching</h1>
<p>Ten exact #173 held-out examples. All seven clips in each row are normalized to the same −1 dBFS peak for listening only. Model spectra are the direct outputs before deterministic 48-iteration Griffin–Lim reconstruction; download the exact spectra in NPZ form.</p>
<p>Rate each model 1 (worst) to 5 (best) and add one overall comment per example. Ratings save in this browser. <button id="export">Export ratings</button> <span id="status"></span></p>
<div style="overflow:auto"><table><thead><tr><th>Example</th><th>Target</th>""" + headers + """<th>Overall feedback</th></tr></thead><tbody>""" + "".join(rows) + """</tbody></table></div>
<script>const key="issue183-flow-listening-v1";let saved={};try{saved=JSON.parse(localStorage.getItem(key)||"{}")}catch{}function persist(){const value={};document.querySelectorAll("[data-rating],[data-comment]").forEach(e=>{if(e.value)value[e.dataset.rating||e.dataset.comment]=e.value});saved=value;localStorage.setItem(key,JSON.stringify(value));document.getElementById("status").textContent=" Saved"}document.querySelectorAll("[data-rating],[data-comment]").forEach(e=>{e.value=saved[e.dataset.rating||e.dataset.comment]||"";e.addEventListener("change",persist);e.addEventListener("input",persist)});document.getElementById("export").onclick=()=>{const value={format:"issue183_listening_ratings_v1",ratings_and_comments:saved};const a=document.createElement("a");a.href=URL.createObjectURL(new Blob([JSON.stringify(value,null,2)],{type:"application/json"}));a.download="issue183-listening-ratings.json";a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000)};</script>"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-source", type=Path, default=TRAINING_SOURCE)
    parser.add_argument("--validation-source", type=Path, default=VALIDATION_SOURCE)
    parser.add_argument("--output", type=Path, required=True)
    run_mode = parser.add_mutually_exclusive_group()
    run_mode.add_argument("--evaluate-only-from", type=Path, help="Completed run directory; no fitting")
    run_mode.add_argument("--export-listening-from", type=Path,
                          help="Completed run directory; export previews without fitting or reevaluating")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--validation-every", type=int, default=VALIDATION_EVERY)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--device", choices=("xpu", "cpu"), default="xpu")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if min(args.steps, args.validation_every, args.threads) < 1:
        parser.error("steps, validation cadence and threads must be positive")
    if args.export_listening_from is not None:
        listening = export_listening_from_run(args.export_listening_from, args.validation_source,
                                              torch.device(args.device), args.threads)
        print(json.dumps({"listening": listening}, indent=2), flush=True)
        return
    if args.evaluate_only_from is not None:
        reevaluate(args)
        return
    try:
        run(args)
    except BaseException as error:
        if args.output.exists():
            Progress(args.output, args.steps).update(phase="Failed", state="failed",
                message=f"{type(error).__name__}: {error}", force=True)
        raise


if __name__ == "__main__":
    main()
