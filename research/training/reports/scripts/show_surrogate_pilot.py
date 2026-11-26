"""Show live progress and training/validation loss curves for a Vital surrogate run."""

import argparse
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys
import tkinter as tk
from tkinter import ttk

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

BACKBONES = ("residual_mlp", "transformer", "conformer")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    python = Path(sys.executable).with_name("python.exe")
    exporter_script = Path(__file__).with_name("export_surrogate_loss_traces.py")
    exporter_log = (args.output / "curve-exporter.log").open("a", encoding="utf-8")
    exporter = subprocess.Popen([str(python), str(exporter_script), str(args.output)],
                                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                                stdout=exporter_log, stderr=subprocess.STDOUT)
    root = tk.Tk()
    root.title("OBRUXO · Live surrogate loss curves")
    root.geometry("1120x820")
    frame = ttk.Frame(root, padding=12)
    frame.pack(fill="both", expand=True)
    phase = ttk.Label(frame, text="Waiting for model checkpoints", font=("Segoe UI", 13, "bold"))
    phase.pack(anchor="w")
    stage_progress = ttk.Progressbar(frame)
    stage_progress.pack(fill="x", pady=(8, 4))
    details = ttk.Label(frame)
    details.pack(anchor="w")
    overall = ttk.Progressbar(frame)
    overall.pack(fill="x", pady=(8, 4))
    overall_label = ttk.Label(frame)
    overall_label.pack(anchor="w")
    figure = Figure(figsize=(10.5, 6.1), dpi=90, layout="constrained")
    axes = figure.subplots(len(BACKBONES), 1, sharex=True)
    lines = {}
    for axis, backbone in zip(axes, BACKBONES, strict=True):
        axis.set_title(backbone.replace("_", " ").title())
        axis.set_ylabel("Loss")
        axis.grid(True, alpha=.25)
        lines[backbone] = (
            axis.plot([], [], "--", label="Training objective (100-update mean)")[0],
            axis.plot([], [], label="Validation objective")[0],
            axis.plot([], [], ":", label="Validation spectral/envelope")[0],
            axis.plot([], [], "-.", label="Validation paired change")[0],
        )
        axis.legend(loc="upper right")
    axes[-1].set_xlabel("Optimizer updates")
    canvas = FigureCanvasTkAgg(figure, master=frame)
    canvas.get_tk_widget().pack(fill="both", expand=True, pady=(4, 8))
    curve_note = ttk.Label(frame, text="Training and full held-out validation objective; lower is better.")
    curve_note.pack(anchor="w")
    events = ttk.Label(frame, justify="left", font=("Consolas", 9), wraplength=1080)
    events.pack(anchor="w", pady=6, fill="x")
    controls = ttk.Frame(frame)
    controls.pack(fill="x")
    paused = False
    curve_stamp = None

    def toggle():
        nonlocal paused
        paused = not paused
        play.config(text="Play monitor" if paused else "Pause monitor")
        if not paused:
            read_progress()

    def read_progress():
        try:
            status = json.loads((args.output / "status.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        phase.config(text=f"{status['phase']} · {status['state']}")
        complete, total = status["completed"], max(1, status["total"])
        stage_progress.config(maximum=total, value=complete)
        updated = datetime.fromisoformat(status["updated_at"]).astimezone().strftime("%H:%M:%S")
        details.config(text=f"{complete:,} / {total:,} · {status['failures']:,} failures · updated {updated}")
        index, count = status["stage_index"], status["stage_count"]
        overall.config(maximum=count, value=max(0, index - 1 + complete / total))
        overall_label.config(text=f"Stage {index} of {count}")
        try:
            with (args.output / "events.log").open("rb") as stream:
                stream.seek(max(0, stream.seek(0, 2) - 8192))
                recent = stream.read().decode("utf-8", errors="replace").splitlines()[-5:]
            events.config(text="\n".join(line[:130] for line in recent))
        except OSError:
            pass

    def read_curves():
        nonlocal curve_stamp
        path = args.output / "live-validation-loss.json"
        try:
            stat = path.stat()
            stamp = (stat.st_mtime_ns, stat.st_size)
            if stamp == curve_stamp:
                return
            data = json.loads(path.read_text(encoding="utf-8"))
            curve_stamp = stamp
            points = data.get("curves", {})
            problems = data.get("errors", {})
        except FileNotFoundError:
            points, problems = {}, {}
        except (OSError, ValueError) as error:
            points, problems = {}, {"exporter": str(error)}
        count = sum(bool(trace) for trace in points.values())
        if count:
            for backbone, trace in points.items():
                train, validation, spectral, change = lines[backbone]
                train_rows = [row for row in trace if row.get("train_loss") is not None]
                train.set_data([row["step"] for row in train_rows], [row["train_loss"] for row in train_rows])
                validation.set_data([row["step"] for row in trace if row.get("validation_loss") is not None],
                                    [row["validation_loss"] for row in trace if row.get("validation_loss") is not None])
                spectral.set_data([row["step"] for row in trace if row.get("validation_spectral_envelope_error") is not None],
                                  [row["validation_spectral_envelope_error"] for row in trace
                                   if row.get("validation_spectral_envelope_error") is not None])
                change.set_data([row["step"] for row in trace if row.get("validation_control_change_error") is not None],
                                [row["validation_control_change_error"] for row in trace
                                 if row.get("validation_control_change_error") is not None])
                axis = train.axes
                if trace:
                    axis.relim()
                    axis.autoscale_view()
            figure.canvas.draw_idle()
            curve_note.config(text=f"Live checkpoints available for {count}/{len(BACKBONES)} models · training and validation loss (lower is better).")
        elif problems:
            curve_note.config(text="Checkpoint reader: " + "; ".join(f"{key}: {value}" for key, value in problems.items())[:220])
        elif exporter.poll() is not None:
            curve_note.config(text=f"Checkpoint reader exited with code {exporter.returncode}; see curve-exporter.log.")
        else:
            curve_note.config(text="Waiting for the first model checkpoint (step 100).")

    def refresh():
        if not paused:
            read_progress()
            read_curves()
        root.after(3000, refresh)

    play = ttk.Button(controls, text="Pause monitor", command=toggle)
    play.pack(side="left")
    stop = ttk.Button(controls, text="Stop monitor")
    stop.pack(side="left", padx=8)
    ttk.Label(controls, text="Play/Pause and Stop affect monitoring only. Training continues independently.").pack(side="left")
    def close_monitor():
        if exporter.poll() is None:
            exporter.terminate()
        exporter_log.close()
        root.destroy()

    stop.configure(command=close_monitor)
    root.protocol("WM_DELETE_WINDOW", close_monitor)
    refresh()
    root.mainloop()


if __name__ == "__main__":
    main()
