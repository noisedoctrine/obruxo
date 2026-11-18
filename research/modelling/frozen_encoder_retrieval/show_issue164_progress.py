from __future__ import annotations

import json
from pathlib import Path
import re
import tkinter as tk
from tkinter import ttk


ROOT = Path(__file__).resolve().parent
DATASET = ROOT / "outputs" / "issue152-validation-train-midi-v3-50"
PRIVATE = ROOT / "outputs" / "issue164-active-rms"
STATUS = PRIVATE / "background-status.private.json"
MODEL_ORDER = (
    "cir_ast", "beats_iter3_plus", "clap_music", "dac_44khz",
    "encodec_24khz", "muq_mulan", "basic_pitch_trunk",
)
MODEL_LABELS = {
    "cir_ast": "CIR AST", "beats_iter3_plus": "BEATs Iter3+", "clap_music": "CLAP Music",
    "dac_44khz": "DAC 44 kHz", "encodec_24khz": "EnCodec 24 kHz",
    "muq_mulan": "MuQ Mulan", "basic_pitch_trunk": "Basic Pitch",
}
EXPECTED_PER_MODEL = 6_450
REFRESH_MS = 3_000


def _read_status() -> dict:
    try:
        return json.loads(STATUS.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return {}


def _read_target() -> float | None:
    try:
        record = json.loads((PRIVATE / "preflight.private.json").read_text(encoding="utf-8-sig"))
        return float(record["normalization"]["target_active_rms"])
    except (OSError, KeyError, ValueError, json.JSONDecodeError):
        return None


def _feature_count(model: str) -> int:
    folder = DATASET / "features_active_rms" / model
    try:
        summary = json.loads((folder / "extraction.latest.json").read_text(encoding="utf-8-sig"))
        target = _read_target()
        if (target is not None and summary.get("model") == model and
                summary.get("normalization", {}).get("target_active_rms") == target):
            return min(int(summary.get("feature_count", 0)), EXPECTED_PER_MODEL)
    except (OSError, json.JSONDecodeError, ValueError, TypeError):
        pass
    try:
        return sum(1 for _ in folder.glob("*.npz"))
    except OSError:
        return 0


def _cache_failure_summary() -> str | None:
    counts = []
    for model in MODEL_ORDER:
        path = DATASET / "features_active_rms" / model / "extraction.latest.json"
        try:
            summary = json.loads(path.read_text(encoding="utf-8-sig"))
            counts.append(int(summary["failure_count"]))
        except (OSError, KeyError, ValueError, json.JSONDecodeError):
            return None
    return f"{sum(counts)} across {len(counts)} caches"


def _latest_log() -> Path | None:
    logs = list(PRIVATE.glob("background-*.stdout.log"))
    return max(logs, key=lambda path: path.stat().st_mtime, default=None)


def _log_details(path: Path | None) -> tuple[list[str], dict[str, int]]:
    if path is None:
        return [], {}
    try:
        with path.open("rb") as log:
            log.seek(max(0, path.stat().st_size - 64_000))
            content = log.read().decode("utf-8", errors="replace")
    except OSError:
        return [], {}

    events = [line.strip() for line in content.splitlines()
              if ("completed " in line and "active-RMS features" in line)
              or re.match(r"^\[.*\] \S+ (running|complete|failed)", line)
              or line.startswith(("Reusing ", "Completed ", "Checkpointed ", "Scored active-RMS ",
                                  "Triplet ", "Verified ", "Original-cache reuse check:"))]
    failures = {}
    for match in re.finditer(r"(?ms)^\{\s*\n.*?^\}", content):
        try:
            summary = json.loads(match.group())
        except json.JSONDecodeError:
            continue
        if "model" in summary and "failure_count" in summary:
            failures[summary["model"]] = int(summary["failure_count"])
    return events[-4:], failures


class ProgressWindow:
    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("Issue #164 · Active-RMS progress")
        self.root.geometry("650x455")
        self.root.minsize(600, 420)
        self.root.columnconfigure(0, weight=1)

        main = ttk.Frame(self.root, padding=12)
        main.grid(row=0, column=0, sticky="nsew")
        self.root.rowconfigure(0, weight=1)
        main.columnconfigure(0, weight=1)

        header = ttk.Frame(main)
        header.grid(row=0, column=0, sticky="ew")
        header.columnconfigure(0, weight=1)
        self.stage = ttk.Label(header, text="Reading run status…", font=("Segoe UI", 11, "bold"))
        self.stage.grid(row=0, column=0, sticky="w")
        controls = ttk.Frame(header)
        controls.grid(row=0, column=1, sticky="e", padx=(8, 0))
        self.play_button = ttk.Button(controls, text="Pause", command=self.toggle_play)
        self.play_button.grid(row=0, column=0, padx=(0, 5))
        ttk.Button(controls, text="Stop", command=self.stop_monitoring).grid(row=0, column=1)
        self.paused = False
        self.stopped = False
        self.after_id = None
        self.overall_text = ttk.Label(main, text="Embedding caches: —")
        self.overall_text.grid(row=1, column=0, sticky="w", pady=(7, 2))
        self.overall = ttk.Progressbar(main, maximum=len(MODEL_ORDER) * EXPECTED_PER_MODEL, mode="determinate")
        self.overall.grid(row=2, column=0, sticky="ew", pady=(0, 10))

        table = ttk.LabelFrame(main, text="Encoder caches", padding=(8, 5))
        table.grid(row=3, column=0, sticky="ew")
        table.columnconfigure(1, weight=1)
        self.rows = {}
        for index, model in enumerate(MODEL_ORDER):
            ttk.Label(table, text=MODEL_LABELS[model], width=19).grid(row=index, column=0, sticky="w", padx=(0, 8), pady=2)
            bar = ttk.Progressbar(table, maximum=EXPECTED_PER_MODEL, mode="determinate")
            bar.grid(row=index, column=1, sticky="ew", pady=2)
            count = ttk.Label(table, text=f"0 / {EXPECTED_PER_MODEL:,}", width=16, anchor="e")
            count.grid(row=index, column=2, sticky="e", padx=(8, 0), pady=2)
            self.rows[model] = (bar, count)

        self.info = ttk.Label(main, text="", justify="left")
        self.info.grid(row=4, column=0, sticky="w", pady=(8, 6))
        events = ttk.LabelFrame(main, text="Recent activity", padding=6)
        events.grid(row=5, column=0, sticky="nsew")
        main.rowconfigure(5, weight=1)
        events.columnconfigure(0, weight=1)
        self.event_text = tk.Text(events, height=4, wrap="word", state="disabled", font=("Consolas", 9))
        self.event_text.grid(row=0, column=0, sticky="nsew")
        ttk.Label(main, text="Play/Pause and Stop control monitoring only; the background extraction keeps running.")\
            .grid(row=6, column=0, sticky="w", pady=(6, 0))
        self.refresh()

    def toggle_play(self) -> None:
        if self.paused or self.stopped:
            self.paused = False
            self.stopped = False
            self.play_button.configure(text="Pause")
            self.refresh()
            return
        self.paused = True
        self.play_button.configure(text="Play")
        if self.after_id is not None:
            self.root.after_cancel(self.after_id)
            self.after_id = None

    def stop_monitoring(self) -> None:
        self.stopped = True
        self.paused = False
        self.play_button.configure(text="Play")
        if self.after_id is not None:
            self.root.after_cancel(self.after_id)
            self.after_id = None
        self.stage.configure(text="Monitoring stopped — background extraction continues")

    def refresh(self) -> None:
        if self.stopped or self.paused:
            return
        status = _read_status()
        stage = status.get("stage", "waiting")
        state = status.get("state", "no status file")
        display_stage = MODEL_LABELS.get(stage, stage.replace("_", " ").title())
        self.stage.configure(text=f"Current phase: {display_stage} — {state}")

        total = 0
        for model, (bar, label) in self.rows.items():
            count = min(_feature_count(model), EXPECTED_PER_MODEL)
            total += count
            bar.configure(value=count)
            label.configure(text=f"{count:,} / {EXPECTED_PER_MODEL:,}")
        all_features = len(MODEL_ORDER) * EXPECTED_PER_MODEL
        self.overall.configure(value=total)
        cache_status = f"Embedding caches: {total:,} / {all_features:,} ({100 * total / all_features:.1f}%)"
        if state == "running" and stage in {"evaluate", "analyze"}:
            cache_status += f" · extraction complete; {stage} still running"
        self.overall_text.configure(text=cache_status)

        log = _latest_log()
        events, failures = _log_details(log)
        failures_text = _cache_failure_summary() or ", ".join(
            f"{MODEL_LABELS.get(model, model)}: {count}" for model, count in failures.items())
        if not failures_text:
            failures_text = "not reported yet"
        phase_changed = status.get("updated_at", "—")
        target = _read_target()
        target_text = f"{target:.8f}" if target is not None else "unavailable"
        self.info.configure(text=f"Target active RMS: {target_text} · reported failures: {failures_text}\nPhase last changed: {phase_changed}")
        self.event_text.configure(state="normal")
        self.event_text.delete("1.0", "end")
        self.event_text.insert("1.0", "\n".join(events) if events else "No recent log events.")
        self.event_text.configure(state="disabled")
        self.after_id = self.root.after(REFRESH_MS, self.refresh)

    def run(self) -> None:
        self.root.mainloop()


if __name__ == "__main__":
    ProgressWindow().run()
