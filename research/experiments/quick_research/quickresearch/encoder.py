from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import time

from .common import BudgetExpired, Context, digest, file_hash, finish_report, read_json, write_csv, write_json
from .fixtures import load_fixture, performance
from .renders import render_key, render_one, renderer_for, usable, validated_cache


def posterior_diagnostics(reference, posteriors) -> list[dict]:
    import numpy as np
    from obruxo_basic_pitch.postprocess import model_frames_to_time
    times = model_frames_to_time(len(posteriors["note"]))
    rows = []
    for index, note in enumerate(reference):
        held = (times >= note.onset_s) & (times < note.offset_s)
        onset = np.abs(times - note.onset_s) <= .05
        pitch_bin = note.pitch_midi - 21
        def average(bin_index):
            if not 0 <= bin_index < 88 or not held.any():
                return None
            return float(posteriors["note"][held, bin_index].mean())
        rows.append({"reference_note": index, **note.as_dict(), "held_frames": int(held.sum()),
                     "correct_pitch_mean": average(pitch_bin), "lower_octave_mean": average(pitch_bin - 12),
                     "upper_octave_mean": average(pitch_bin + 12),
                     "onset_peak_within_50ms": float(posteriors["onset"][onset, pitch_bin].max()) if onset.any() else None})
    return rows


class FrozenFeatures:
    def __init__(self, args, directory: Path):
        import torch
        from obruxo_basic_pitch.postprocess import StockDecoderSettings
        self.torch, self.directory, self.model = torch, directory, None
        self.checkpoint = Path(args.checkpoint).resolve(strict=True)
        available = hasattr(torch, "xpu") and torch.xpu.is_available()
        if args.device == "xpu" and not available:
            raise ValueError("explicit XPU requested but unavailable")
        self.device = "xpu" if args.device != "cpu" and available else "cpu"
        self.config = {"schema": "quick_basic_pitch_features_v1", "checkpoint_sha256": file_hash(self.checkpoint),
                       "backend": f"pytorch_{self.device}", "precision": "float32", "torch_version": torch.__version__,
                       "implementation_hash": args.implementation_hash, "decoder": asdict(StockDecoderSettings()),
                       "preparation": "prepare_wav + unwrap_window_outputs; pinned stock frontend",
                       "device_request": args.device, "cpu_fallback": args.device == "auto" and not available}

    def get(self, row: dict):
        import numpy as np
        from obruxo_basic_pitch.inference import prepare_wav, unwrap_window_outputs
        from obruxo_basic_pitch.model import BasicPitchICASSP2022
        identity = {"audio_sha256": row["audio_sha256"], **self.config}
        key = digest(identity)
        self.directory.mkdir(parents=True, exist_ok=True)
        arrays_path, metadata_path = self.directory / f"{key}.npz", self.directory / f"{key}.json"
        if arrays_path.exists() and metadata_path.exists():
            saved = read_json(metadata_path)
            if saved.get("identity") == identity and saved.get("arrays_sha256") == file_hash(arrays_path):
                with np.load(arrays_path, allow_pickle=False) as stored:
                    arrays = {name: stored[name] for name in ("note", "onset", "contour")}
                self.validate(arrays)
                return arrays, {"key": key, "identity": identity, "arrays_path": str(arrays_path.resolve()), "cache_hit": True}
        torch = self.torch
        if self.model is None:
            torch.set_num_threads(1)
            self.model = BasicPitchICASSP2022()
            self.model.load_state_dict(torch.load(self.checkpoint, map_location="cpu", weights_only=True))
            self.model.requires_grad_(False).eval().to(self.device)
        prepared = prepare_wav(Path(row["wav"]))
        chunks = {name: [] for name in ("note", "onset", "contour")}
        with torch.inference_mode():
            for start in range(0, len(prepared.windows), 4):
                batch = torch.from_numpy(prepared.windows[start:start + 4]).to(self.device)
                for name, values in self.model(batch).items():
                    chunks[name].append(values.cpu().numpy())
        arrays = unwrap_window_outputs({name: np.concatenate(values) for name, values in chunks.items()},
                                       original_sample_count=prepared.original_sample_count)
        self.validate(arrays)
        if any(parameter.requires_grad or parameter.grad is not None for parameter in self.model.parameters()):
            raise RuntimeError("frozen-model contract violated")
        temporary = arrays_path.with_suffix(".partial.npz")
        np.savez_compressed(temporary, **arrays)
        temporary.replace(arrays_path)
        write_json(metadata_path, {"identity": identity, "arrays_sha256": file_hash(arrays_path)})
        return arrays, {"key": key, "identity": identity, "arrays_path": str(arrays_path.resolve()), "cache_hit": False}

    @staticmethod
    def validate(arrays):
        import numpy as np
        frames = len(arrays["note"])
        if not frames:
            raise ValueError("empty posterior sequence")
        for name, width in (("note", 88), ("onset", 88), ("contour", 264)):
            if arrays[name].shape != (frames, width) or not np.isfinite(arrays[name]).all():
                raise ValueError("invalid posterior shape or nonfinite values")


def plot_fixture(output: Path, fixture_id: str, cases: list[dict]) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from obruxo_basic_pitch.postprocess import model_frames_to_time
    fig, axes = plt.subplots(2, len(cases), squeeze=False, figsize=(5 * len(cases), 6))
    for column, case in enumerate(cases):
        with np.load(case["features"]["arrays_path"], allow_pickle=False) as stored:
            times = model_frames_to_time(len(stored["note"]))
            for index, name in enumerate(("note", "onset")):
                axis = axes[index, column]
                axis.imshow(stored[name].T, origin="lower", aspect="auto", extent=(times[0], times[-1], 20.5, 108.5), vmin=0, vmax=1)
                for note in case["reference"]:
                    axis.plot([note["onset_s"], note["offset_s"]], [note["pitch_midi"]] * 2, color="orange", linewidth=2)
                for note in case["events"]:
                    axis.plot([note["start_time_s"], note["end_time_s"]], [note["pitch_midi"]] * 2, color="white", linewidth=.8)
                axis.set(title=f"{case['performance']} / {name}", xlabel="seconds", ylabel="MIDI pitch", ylim=(33, 90))
    fig.suptitle(f"{fixture_id}: orange = MIDI-held interval; white = decoded event; color = posterior 0–1")
    fig.tight_layout()
    image = output / f"images/{fixture_id}.png"
    image.parent.mkdir(exist_ok=True)
    fig.savefig(image, dpi=120)
    plt.close(fig)
    return f"images/{fixture_id}.png"


def run(context: Context, args) -> dict:
    from obruxo_basic_pitch.evaluation.labels import performance_labels
    from obruxo_basic_pitch.evaluation.metrics import evaluate_notes_and_frames
    from obruxo_basic_pitch.postprocess import decode_notes
    measurement = Path(args.measurement).resolve(strict=True)
    if read_json(measurement / "status.json").get("state") not in ("complete", "complete_with_failures"):
        raise ValueError("B's latest invocation is not complete; do not consume stale artifacts")
    bank = read_json(measurement / "bank.json")
    summary = read_json(measurement / "measurement_summary.json")
    if summary.get("bank_sha256") != file_hash(measurement / "bank.json"):
        raise ValueError("B bank hash mismatch")
    if summary.get("render_manifest_sha256") != file_hash(measurement / "render_manifest.json"):
        raise ValueError("B render-manifest hash mismatch")
    if not summary.get("complete") or bank["implementation_hash"] != args.implementation_hash:
        raise ValueError("C requires completed B coverage under unchanged implementation; failures may remain explicitly scored as unavailable")
    if digest(bank["fixtures"]) != bank["fixtures_hash"]:
        raise ValueError("measurement fixture manifest mismatch")
    features = FrozenFeatures(args, context.output / "features")
    renderer = renderer_for(args)
    if renderer.renderer_id != bank["renderer_id"] or file_hash(Path(args.renderer_config)) != bank["renderer_config_sha256"]:
        raise ValueError("C must reuse B's renderer identity")
    originals = {row["key"]: row for row in read_json(measurement / "render_manifest.json")}
    cases, diagnostics, costs = [], [], []
    reason = None
    performance_names = ("held_c4", "held_c3", "staccato_c4")
    references = {}
    for name in performance_names:
        midi = context.output / f"midi/{name}.mid"
        performance(name).save_midi(midi)
        reference, _ = performance_labels(midi)
        references[name] = (midi, reference)
    plan = [(fixture, name) for fixture in bank["fixtures"] for name in performance_names]
    try:
        for index, (fixture, name) in enumerate(plan):
            context.check(20)
            if costs and 1.25 * max(costs) * (len(plan) - index) + 20 > context.remaining:
                reason = "projected_budget_exceeded"
                break
            started = time.monotonic()
            load_fixture(fixture)
            key = render_key(fixture, name, "baseline", 0)
            if name == "held_c4":
                original = originals.get(key)
                row = validated_cache(measurement / "renders", key, original["identity"]) if original else None
                if row is None:
                    raise ValueError("B baseline cache is missing/corrupt; repair B before C")
            else:
                row = render_one(context, renderer, fixture, name, "baseline", 0, context.output / "renders", args.implementation_hash)
            case = {"fixture_id": fixture["fixture_id"], "category": fixture["category"], "performance": name,
                    "render_key": key, "state": row["state"], "render": row}
            if usable(row):
                try:
                    posteriors, provenance = features.get(row)
                    midi, reference = references[name]
                    events = decode_notes(posteriors)
                    scores = evaluate_notes_and_frames(reference, events, posteriors["note"])
                    case.update(features=provenance, events=[asdict(note) for note in events], reference=[note.as_dict() for note in reference],
                                metrics=scores, midi_sha256=file_hash(midi))
                    for diagnostic in posterior_diagnostics(reference, posteriors):
                        diagnostics.append({"fixture_id": fixture["fixture_id"], "category": fixture["category"], "performance": name, **diagnostic})
                except Exception as error:
                    case.update(state="inference_error", error_type=type(error).__name__, error=str(error))
            case["case_seconds"] = time.monotonic() - started
            costs.append(case["case_seconds"])
            cases.append(case)
            write_json(context.output / "case_results.json", cases)
            write_csv(context.output / "posterior_diagnostics.csv", diagnostics)
            context.progress(completed_cases=len(cases), expected_cases=len(plan), backend=features.config["backend"])
    except BudgetExpired:
        reason = "budget_exhausted"
    complete = len(cases) == len(plan) and reason is None
    good = [case for case in cases if case["state"] == "ok"]
    state = reason or ("complete" if len(good) == len(plan) else "complete_with_failures")
    report = (f"# Controlled Basic Pitch failures\n\nStatus: {state}. Scored {len(good)}/{len(plan)} planned cases. "
              f"Backend: {features.config['backend']}, float32.\n\n"
              "Compare each performance across the same eight fixtures. Dense note/onset posteriors and stock-decoded events localize observed pitch, "
              "onset, or segmentation failures. Reference spans are MIDI-held intervals; post-note-off acoustic release is distinct and remains in the audio.\n\n"
              "Orange overlays show known MIDI, white shows decoded events, and color shows posterior strength. "
              "Missing/late onset support versus fragmented events over strong note support are diagnostic patterns to inspect, not automatic causal labels.\n\n"
              "These fixture-level observations do not establish category-wide rankings, disentanglement, or downstream conditioning benefit. "
              "No inference receives MIDI. No model is trained.\n\n")
    try:
        for fixture in bank["fixtures"]:
            subset = [case for case in good if case["fixture_id"] == fixture["fixture_id"]]
            if not subset:
                continue
            context.check(3)
            image = plot_fixture(context.output, fixture["fixture_id"], subset)
            report += f"![{fixture['fixture_id']} posterior and event evidence]({image})\n\n"
    except BudgetExpired:
        state, complete = "budget_exhausted", False
        report += "\nPlot generation reached the deadline; raw scored evidence is retained.\n"
    write_json(context.output / "encoder_summary.json", {"complete": complete, "state": state, "expected": len(plan),
               "attempted": len(cases), "scored": len(good), "feature_config": features.config})
    write_json(context.output / "case_results.json", cases)
    return finish_report(context, "CONTROLLED_BASIC_PITCH_REPORT.md", report, state, expected=len(plan), scored=len(good))
