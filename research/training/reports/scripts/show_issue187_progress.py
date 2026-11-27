"""Display the Issue 187 worker status without controlling the worker."""

from datetime import datetime
import json
from pathlib import Path
import sys
import tkinter as tk
from tkinter import ttk


def main(output):
    root = tk.Tk()
    root.title("OBRUXO · Issue 187 Vital surrogates")
    root.geometry("830x310")
    frame = ttk.Frame(root, padding=14)
    frame.pack(fill="both", expand=True)
    phase = ttk.Label(frame, text="Waiting for the worker", font=("Segoe UI", 13, "bold"))
    phase.pack(anchor="w")
    detail = ttk.Label(frame)
    detail.pack(anchor="w")
    phase_bar = ttk.Progressbar(frame)
    phase_bar.pack(fill="x", pady=(6, 8))
    overall = ttk.Label(frame)
    overall.pack(anchor="w")
    stage_bar = ttk.Progressbar(frame, maximum=3)
    stage_bar.pack(fill="x", pady=(6, 8))
    events = ttk.Label(frame, justify="left", wraplength=790)
    events.pack(anchor="w", fill="both", expand=True)
    controls = ttk.Frame(frame)
    controls.pack(fill="x")
    paused = False

    def refresh():
        if not paused:
            try:
                state = json.loads((output / "status.json").read_text(encoding="utf-8"))
                done, total = state["completed"], max(1, state["total"])
                stage, count = state["stage_index"], state["stage_count"]
                updated = datetime.fromisoformat(state["updated_at"]).astimezone().strftime("%H:%M:%S")
                phase.config(text=f"{state['phase']} · {state['state']}")
                detail.config(text=f"{done:,} / {total:,} · {state['failures']:,} failures · updated {updated}")
                phase_bar.config(maximum=total, value=done)
                overall.config(text=f"Stage {stage} of {count}")
                stage_bar.config(maximum=count, value=(count if state["state"] == "complete" else stage - 1))
                with (output / "events.log").open("rb") as stream:
                    stream.seek(max(0, stream.seek(0, 2) - 8192))
                    tail = stream.read().decode("utf-8", errors="replace").splitlines()[-4:]
                events.config(text="\n".join(tail))
            except (OSError, ValueError, KeyError):
                pass
        root.after(2500, refresh)

    def toggle():
        nonlocal paused
        paused = not paused
        play.config(text="Play monitor" if paused else "Pause monitor")

    play = ttk.Button(controls, text="Pause monitor", command=toggle)
    play.pack(side="left")
    ttk.Button(controls, text="Stop monitor", command=root.destroy).pack(side="left", padx=6)
    ttk.Label(controls, text="Controls affect the monitor only.").pack(side="left")
    refresh()
    root.mainloop()


if __name__ == "__main__":
    main(Path(sys.argv[1]))
