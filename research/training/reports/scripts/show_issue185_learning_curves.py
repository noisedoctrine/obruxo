"""Live, read-only monitor for the Issue 185 continuation worker."""

import argparse
import ctypes
import json
import os
from pathlib import Path
import tkinter as tk
import tkinter.font as tkfont
from tkinter import ttk

ARMS = ("residual_mlp-direct", "residual_mlp-flow")
COLORS = {"residual_mlp-direct": "#276EF1", "residual_mlp-flow": "#D55E00"}
MILESTONES = (500, 1_000, 2_000, 4_000, 8_000)


def read_json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def set_dpi_awareness():
    if os.name != "nt":
        return
    try:
        ctypes.windll.user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))
    except (AttributeError, OSError):
        pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    set_dpi_awareness()
    root = tk.Tk()
    root.title("OBRUXO · Issue 185 learning curves")
    root.geometry("1080x740")
    root.minsize(820, 560)
    tkfont.nametofont("TkDefaultFont", root=root).configure(family="Segoe UI", size=11)
    frame = ttk.Frame(root, padding=14)
    frame.pack(fill="both", expand=True)

    phase = ttk.Label(frame, text="Waiting for the worker", font=("Segoe UI", 15, "bold"))
    phase.pack(anchor="w")
    detail = ttk.Label(frame, text="Training progress appears here.")
    detail.pack(anchor="w", pady=(3, 0))
    progress = ttk.Progressbar(frame, mode="determinate", maximum=15_000)
    progress.pack(fill="x", pady=(10, 4))
    overall = ttk.Label(frame, text="Overall updates 0 / 15,000 continuation updates")
    overall.pack(anchor="w")

    chart = tk.Canvas(frame, background="white", highlightthickness=1, highlightbackground="#cbd2d9")
    chart.pack(fill="both", expand=True, pady=(10, 8))
    recent = ttk.Label(frame, text="Recent events", justify="left", font=("Consolas", 10), wraplength=1000)
    recent.pack(fill="x", pady=(0, 8))
    controls = ttk.Frame(frame)
    controls.pack(fill="x")
    paused = False
    ttk.Label(controls, text="Pause and Stop affect this monitor only. The training worker keeps running.").pack(side="right")

    def draw(checkpoints):
        chart.delete("all")
        width, height = max(500, chart.winfo_width()), max(280, chart.winfo_height())
        left, right, top, bottom = 80, width - 28, 42, height - 44
        series = {}
        for arm in ARMS:
            series[arm] = []
            for step, value in sorted((checkpoints or {}).get(arm, {}).items(), key=lambda item: int(item[0])):
                metric = value.get("metrics", {}).get("spectral_loss")
                if metric is not None:
                    series[arm].append((int(step), metric))
        values = [value for points in series.values() for _, value in points]
        chart.create_text(left, 18, anchor="w", text="Validation spectral loss · lower is better",
                          font=("Segoe UI", 12, "bold"), fill="#202831")
        chart.create_text(right, height - 12, anchor="e", text="Cumulative optimizer updates · 8,000 per model",
                          font=("Segoe UI", 10), fill="#374151")
        if not values:
            chart.create_text(width / 2, height / 2, text="Waiting for the first validation result",
                              font=("Segoe UI", 12), fill="#4b5563")
            return
        low, high = min(values), max(values)
        padding = max(.01, .1 * (high - low))
        low, high = low - padding, high + padding
        x = lambda step: left + (step - 500) / 7_500 * (right - left)
        y = lambda value: bottom - (value - low) / (high - low) * (bottom - top)
        for index in range(5):
            tick = high - (high - low) * index / 4
            tick_y = y(tick)
            chart.create_line(left, tick_y, right, tick_y, fill="#e3e8ee")
            chart.create_text(left - 10, tick_y, text=f"{tick:.3f}", anchor="e", fill="#374151")
        chart.create_line(left, top, left, bottom, fill="#4b5563")
        chart.create_line(left, bottom, right, bottom, fill="#4b5563")
        for step in MILESTONES:
            tick_x = x(step)
            chart.create_text(tick_x, bottom + 16, text=f"{step:,}", font=("Segoe UI", 9), fill="#374151")
        for index, arm in enumerate(ARMS):
            points = series[arm]
            color = COLORS[arm]
            label = "Direct prediction" if arm.endswith("direct") else "Flow matching · Euler 8"
            legend_y = 18 + index * 20
            chart.create_line(right - 230, legend_y, right - 208, legend_y, fill=color, width=3)
            chart.create_text(right - 202, legend_y, anchor="w", text=label, fill=color,
                              font=("Segoe UI", 9, "bold"))
            if len(points) > 1:
                coordinates = [coordinate for step, value in points for coordinate in (x(step), y(value))]
                chart.create_line(*coordinates, fill=color, width=3, smooth=False)
            for step, value in points:
                chart.create_oval(x(step) - 4, y(value) - 4, x(step) + 4, y(value) + 4,
                                  fill=color, outline="white")
            if points:
                step, value = points[-1]
                chart.create_text(right - 7, top + 44 + index * 22, anchor="e",
                                  text=f"{label}: {value:.4f} at {step:,}", fill=color,
                                  font=("Segoe UI", 10, "bold"))

    def refresh():
        if not paused:
            status = read_json(args.output / "status.json")
            checkpoints = read_json(args.output / "checkpoints.json")
            draw(checkpoints)
            if status:
                phase.config(text=f"{status.get('phase', 'Unknown')} · {status.get('state', 'unknown')}")
                arm = status.get("arm") or "Preparing"
                done, total = status.get("completed", 0), status.get("total", 8_000)
                detail.config(text=f"{arm} · {done:,} / {total:,} updates"
                                  + (f" · {status['message']}" if status.get("message") else ""))
                arm_index = status.get("arm_index", 1)
                progress.configure(maximum=15_000,
                                   value=min(15_000, max(0, (arm_index - 1) * 7_500 + max(0, done - 500))))
                overall.config(text=f"Overall updates {int(progress['value']):,} / 15,000 continuation updates")
            try:
                rows = (args.output / "events.jsonl").read_text(encoding="utf-8").splitlines()[-4:]
                events = [json.loads(row) for row in rows]
                recent.config(text="\n".join(
                    f"{event.get('updated_at', '')[11:19]} · {event.get('phase', 'Unknown')} · "
                    f"{event.get('completed', 0):,} / {event.get('total', 0):,} · {event.get('state', 'unknown')}"
                    + (f" · {event['message']}" if event.get("message") else "")
                    for event in events) if events else "No worker events yet")
            except OSError:
                recent.config(text="Waiting for status and checkpoint files")
        root.after(2500, refresh)

    def toggle():
        nonlocal paused
        paused = not paused
        pause.config(text="Play monitor" if paused else "Pause monitor")
        if not paused:
            refresh()

    pause = ttk.Button(controls, text="Pause monitor", command=toggle)
    pause.pack(side="left")
    ttk.Button(controls, text="Stop monitor", command=root.destroy).pack(side="left", padx=(8, 0))
    root.after(250, refresh)
    root.mainloop()


if __name__ == "__main__":
    main()
