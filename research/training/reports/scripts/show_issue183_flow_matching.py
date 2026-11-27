"""Live Tkinter monitor for the independently running issue 183 worker."""

import argparse
import ctypes
from datetime import datetime
import json
import os
from pathlib import Path
import tkinter as tk
import tkinter.font as tkfont
from tkinter import ttk

BACKBONES = ("residual_mlp", "transformer", "conformer")
MODES = ("direct", "flow")
COLORS = {"direct": "#276EF1", "flow": "#D55E00"}


def enable_windows_dpi_awareness():
    if os.name != "nt":
        return
    try:
        set_context = ctypes.windll.user32.SetProcessDpiAwarenessContext
        set_context.argtypes = [ctypes.c_void_p]
        set_context.restype = ctypes.c_bool
        if set_context(ctypes.c_void_p(-4)):
            return
    except (AttributeError, OSError):
        pass
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except (AttributeError, OSError):
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except (AttributeError, OSError):
            pass


def configure_fonts(root):
    tkfont.nametofont("TkDefaultFont", root=root).configure(family="Segoe UI", size=11)
    tkfont.nametofont("TkTextFont", root=root).configure(family="Segoe UI", size=11)
    tkfont.nametofont("TkFixedFont", root=root).configure(family="Consolas", size=10)


def read_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def read_traces(directory):
    traces = {}
    for backbone in BACKBONES:
        for mode in MODES:
            arm = f"{backbone}-{mode}"
            path = directory / "traces" / f"{arm}.jsonl"
            try:
                traces[arm] = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
            except (OSError, ValueError):
                traces[arm] = []
    return traces


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    enable_windows_dpi_awareness()
    root = tk.Tk()
    root.title("OBRUXO · Issue 183 flow matching")
    configure_fonts(root)
    root.geometry("1600x1050")
    root.minsize(1180, 780)
    frame = ttk.Frame(root, padding=12)
    frame.pack(fill="both", expand=True)

    phase = ttk.Label(frame, text="Waiting for the worker", font=("Segoe UI", 15, "bold"))
    phase.pack(anchor="w")
    detail = ttk.Label(frame, text="The training worker runs independently.")
    detail.pack(anchor="w", pady=(3, 0))
    arm_progress = ttk.Progressbar(frame, mode="determinate")
    arm_progress.pack(fill="x", pady=(8, 3))
    overall_progress = ttk.Progressbar(frame, mode="determinate", maximum=6)
    overall_progress.pack(fill="x", pady=3)
    overall_label = ttk.Label(frame, text="Overall: 0 of 6 arms")
    overall_label.pack(anchor="w")

    tabs = ttk.Notebook(frame)
    tabs.pack(fill="both", expand=True, pady=(8, 4))
    validation_tab = ttk.Frame(tabs, padding=4)
    objective_tab = ttk.Frame(tabs, padding=4)
    tabs.add(validation_tab, text="Validation spectral loss")
    tabs.add(objective_tab, text="Training objective")
    validation_canvas = tk.Canvas(validation_tab, background="white", highlightthickness=0)
    validation_canvas.pack(fill="both", expand=True)
    objective_canvas = tk.Canvas(objective_tab, background="white", highlightthickness=0)
    objective_canvas.pack(fill="both", expand=True)

    events = ttk.Label(frame, text="Recent worker events", justify="left", font=("Consolas", 10), wraplength=1380)
    events.pack(anchor="w", fill="x", pady=(4, 8))
    controls = ttk.Frame(frame)
    controls.pack(fill="x")
    paused = False
    status_text = ttk.Label(controls, text="Buttons affect the monitor window only. The worker keeps running.")
    status_text.pack(side="right", padx=8)

    def draw_panel(canvas, bounds, title, series, x_max, shared_range=None):
        left, top, right, bottom = bounds
        canvas.create_rectangle(left, top, right, bottom, outline="#cbd2d9")
        canvas.create_text(left + 12, top + 15, anchor="w", text=title, fill="#202831",
                           font=("Segoe UI", 12, "bold"))
        plot_left, plot_right = left + 76, right - 246
        plot_top, plot_bottom = top + 40, bottom - 34
        canvas.create_text(plot_left, bottom - 10, anchor="w", text="0", fill="#374151", font=("Segoe UI", 11))
        canvas.create_text(plot_right, bottom - 10, anchor="e", text=str(x_max), fill="#374151", font=("Segoe UI", 11))
        values = [point[1] for _, points, _ in series for point in points]
        if not values:
            canvas.create_text((plot_left + plot_right) / 2, (plot_top + plot_bottom) / 2,
                               text="Waiting for validation updates", fill="#4b5563", font=("Segoe UI", 11))
            return
        if shared_range is None:
            low, high = min(values), max(values)
            pad = max(.025, .08 * (high - low))
            low, high = low - pad, high + pad
        else:
            low, high = shared_range
        if high - low < 1e-9:
            low, high = low - .05, high + .05
        for tick in range(5):
            fraction = tick / 4
            y = plot_top + (plot_bottom - plot_top) * fraction
            value = high - (high - low) * fraction
            canvas.create_line(plot_left, y, plot_right, y, fill="#e3e8ee", width=1.5)
            canvas.create_text(plot_left - 10, y, anchor="e", text=f"{value:.3f}",
                               fill="#374151", font=("Segoe UI", 11))
        canvas.create_line(plot_left, plot_top, plot_left, plot_bottom, fill="#4b5563", width=2)
        canvas.create_line(plot_left, plot_bottom, plot_right, plot_bottom, fill="#4b5563", width=2)

        def position(point):
            step, value = point
            x = plot_left + (plot_right - plot_left) * step / max(1, x_max)
            y = plot_bottom - (plot_bottom - plot_top) * (value - low) / (high - low)
            return x, y

        for series_index, (label, points, color) in enumerate(series):
            if len(points) > 1:
                coords = []
                for point in points:
                    coords.extend(position(point))
                canvas.create_line(*coords, fill=color, width=4, smooth=False)
                for point in points:
                    x, y = position(point)
                    canvas.create_oval(x - 4.5, y - 4.5, x + 4.5, y + 4.5,
                                       fill=color, outline="white", width=1.5)
            legend_x = right - 236 + series_index * 112
            canvas.create_line(legend_x, top + 15, legend_x + 19, top + 15, fill=color, width=3)
            canvas.create_text(legend_x + 25, top + 15, anchor="w", text=label, fill=color,
                               font=("Segoe UI", 10, "bold"))
            if points:
                newest_index = len(points) - 1
                minimum_index = min(range(len(points)), key=lambda index: points[index][1])
                short = "Direct" if label.startswith("Direct") else "Flow"
                annotations = [(newest_index, f"{short} latest {points[newest_index][1]:.4f} @ {points[newest_index][0]}")]
                if minimum_index == newest_index:
                    annotations[0] = (newest_index, f"{short} latest + min {points[newest_index][1]:.4f} @ {points[newest_index][0]}")
                else:
                    annotations.append((minimum_index, f"{short} min {points[minimum_index][1]:.4f} @ {points[minimum_index][0]}"))
                for annotation_index, (point_index, text) in enumerate(annotations):
                    x, y = position(points[point_index])
                    label_y = top + 46 + series_index * 50 + annotation_index * 19
                    canvas.create_rectangle(x - 5, y - 5, x + 5, y + 5,
                                            fill="white", outline=color, width=2)
                    canvas.create_line(x + 6, y, plot_right + 9, label_y,
                                       fill=color, dash=(3, 2), width=1.5)
                    canvas.create_text(plot_right + 15, label_y, anchor="w", text=text,
                                       fill=color, font=("Segoe UI", 10, "bold"))

    def redraw(traces):
        validation_canvas.delete("all")
        objective_canvas.delete("all")
        width = max(600, validation_canvas.winfo_width())
        height = max(560, validation_canvas.winfo_height())
        row_height = max(170, (height - 18) // 3)
        validation_values = [point.get("validation_spectral_loss")
                             for trace in traces.values() for point in trace
                             if point.get("validation_spectral_loss") is not None]
        validation_range = None
        if validation_values:
            low, high = min(validation_values), max(validation_values)
            pad = max(.025, .08 * (high - low))
            validation_range = (low - pad, high + pad)
        for row, backbone in enumerate(BACKBONES):
            direct = [p for p in traces[f"{backbone}-direct"] if p.get("validation_spectral_loss") is not None]
            flow = [p for p in traces[f"{backbone}-flow"] if p.get("validation_spectral_loss") is not None]
            series = [
                ("Direct", [(p["step"], p["validation_spectral_loss"]) for p in direct], COLORS["direct"]),
                ("Flow · Euler 8", [(p["step"], p["validation_spectral_loss"]) for p in flow], COLORS["flow"]),
            ]
            draw_panel(validation_canvas, (8, 6 + row * row_height, width - 8, 6 + (row + 1) * row_height - 4),
                       backbone, series, 500, shared_range=validation_range)
        obj_width, obj_height = max(600, objective_canvas.winfo_width()), max(560, objective_canvas.winfo_height())
        panel_width, panel_height = (obj_width - 28) / 2, (obj_height - 28) / 3
        for index, backbone in enumerate(BACKBONES):
            for column, mode in enumerate(MODES):
                arm = f"{backbone}-{mode}"
                points = [(p["step"], p["train_objective"]) for p in traces[arm]
                          if p.get("train_objective") is not None]
                x, y = 8 + column * (panel_width + 8), 6 + index * (panel_height + 8)
                label = "Direct spectral objective" if mode == "direct" else "Flow velocity MSE"
                draw_panel(objective_canvas, (x, y, x + panel_width, y + panel_height),
                           f"{backbone} · {label}", [(label, points, COLORS[mode])], 500)

    def refresh():
        if not paused:
            status = read_json(args.output / "status.json")
            traces = read_traces(args.output)
            redraw(traces)
            if status:
                phase.config(text=f"{status.get('phase', 'Unknown phase')} · {status.get('state', 'unknown')}")
                done, total = status.get("completed", 0), max(1, status.get("total", 1))
                arm_progress.configure(maximum=total, value=min(done, total))
                updated = status.get("updated_at")
                if updated:
                    try:
                        updated = datetime.fromisoformat(updated).astimezone().strftime("%H:%M:%S")
                    except ValueError:
                        pass
                detail.config(text=f"{status.get('arm') or 'Preparing'} · {done:,}/{total:,} updates or validation pairs"
                                  + (f" · updated {updated}" if updated else "")
                                  + (f" · {status['message']}" if status.get("message") else ""))
                index, count = status.get("arm_index", 0), status.get("arm_count", 6)
                overall_progress.configure(maximum=count, value=max(0, index - 1 + done / total))
                overall_label.config(text=f"Overall: arm {index} of {count}")
            try:
                rows = (args.output / "events.jsonl").read_text(encoding="utf-8").splitlines()[-4:]
                events.config(text="\n".join(rows) if rows else "No events yet")
            except OSError:
                events.config(text="Waiting for status and trace files")
        root.after(2500, refresh)

    def toggle():
        nonlocal paused
        paused = not paused
        play.config(text="Play monitor" if paused else "Pause monitor")
        if not paused:
            refresh()

    play = ttk.Button(controls, text="Pause monitor", command=toggle)
    play.pack(side="left")
    ttk.Button(controls, text="Stop monitor", command=root.destroy).pack(side="left", padx=(8, 0))
    root.after(250, refresh)
    root.mainloop()


if __name__ == "__main__":
    main()
