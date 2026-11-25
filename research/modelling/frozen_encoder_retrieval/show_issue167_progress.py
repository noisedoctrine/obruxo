"""Display the V7 comparison's durable progress without controlling its workers."""

from datetime import datetime
import tkinter as tk
from tkinter import ttk

from run_issue167_v7 import LOG, RUN, STATUS, read_json


def main():
    root = tk.Tk()
    root.title("Issue #167 · Vital JEPA V7 comparison")
    root.geometry("800x275")
    panel = ttk.Frame(root, padding=14)
    panel.pack(fill="both", expand=True)
    phase = ttk.Label(panel, text="Waiting for status", font=("Segoe UI", 11, "bold"))
    phase.pack(anchor="w")
    progress = ttk.Progressbar(panel)
    progress.pack(fill="x", pady=8)
    details = ttk.Label(panel)
    details.pack(anchor="w")
    events = ttk.Label(panel, font=("Consolas", 9), justify="left")
    events.pack(anchor="w", pady=8, fill="x", expand=True)
    controls = ttk.Frame(panel)
    controls.pack(fill="x")
    paused = False

    def refresh():
        if not paused:
            try:
                state = read_json(STATUS)
                phase.config(text=f"{state['phase']} · {state['state']}")
                progress.config(maximum=max(1, state['total']), value=state['completed'])
                updated = datetime.fromisoformat(state['updated_at']).astimezone().strftime("%H:%M:%S")
                details.config(text=f"{state['completed']:,} / {state['total']:,} · {state['failures']:,} failures · updated {updated}")
                log = RUN / "analysis.log" if (RUN / "analysis.log").exists() and state['phase'] != 'Render' else LOG
                with log.open("rb") as stream:
                    stream.seek(max(0, log.stat().st_size - 8192))
                    tail = stream.read().decode("utf-8", errors="replace").splitlines()
                events.config(text="\n".join(line[:115] for line in tail[-5:]))
            except (OSError, ValueError, KeyError):
                pass
        root.after(3000, refresh)

    def toggle():
        nonlocal paused
        paused = not paused
        play.config(text="Play monitor" if paused else "Pause monitor")

    play = ttk.Button(controls, text="Pause monitor", command=toggle)
    play.pack(side="left")
    ttk.Button(controls, text="Stop monitor", command=root.destroy).pack(side="left", padx=6)
    ttk.Label(controls, text="Controls affect the display only.").pack(side="left", padx=6)
    refresh()
    root.mainloop()


if __name__ == "__main__":
    main()
