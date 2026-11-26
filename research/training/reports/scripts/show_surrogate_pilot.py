"""Show progress from the independently running Vital surrogate pilot."""

import argparse
from datetime import datetime
import json
from pathlib import Path
import tkinter as tk
from tkinter import ttk


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    root = tk.Tk()
    root.title("OBRUXO · Vital surrogate pilot")
    root.geometry("860x320")
    frame = ttk.Frame(root, padding=16)
    frame.pack(fill="both", expand=True)
    phase = ttk.Label(frame, text="Waiting for the worker", font=("Segoe UI", 13, "bold"))
    phase.pack(anchor="w")
    stage_progress = ttk.Progressbar(frame)
    stage_progress.pack(fill="x", pady=(8, 4))
    details = ttk.Label(frame)
    details.pack(anchor="w")
    overall = ttk.Progressbar(frame)
    overall.pack(fill="x", pady=(8, 4))
    overall_label = ttk.Label(frame)
    overall_label.pack(anchor="w")
    events = ttk.Label(frame, justify="left", font=("Consolas", 9), wraplength=805)
    events.pack(anchor="w", pady=8, fill="both", expand=True)
    controls = ttk.Frame(frame)
    controls.pack(fill="x")
    paused = False

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

    def refresh():
        if not paused:
            read_progress()
        root.after(3000, refresh)

    play = ttk.Button(controls, text="Pause monitor", command=toggle)
    play.pack(side="left")
    ttk.Button(controls, text="Stop monitor", command=root.destroy).pack(side="left", padx=8)
    ttk.Label(controls, text="Controls affect the display only. The worker runs independently.").pack(side="left")
    refresh()
    root.mainloop()


if __name__ == "__main__":
    main()
