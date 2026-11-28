"""Display the Issue 187 worker status without controlling the worker."""

from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import sys
import tkinter as tk
from tkinter import font as tkfont
from tkinter import ttk


def set_dpi_awareness():
    if sys.platform == "win32":
        import ctypes

        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(2)
        except (AttributeError, OSError):
            pass


def read_timeline(output, now):
    pattern = re.compile(r"^(\S+) (.*?): (\d+)/(\d+) \(([^)]+)\)")
    events = []
    paths = (output / "events.log", output.parent / "issue187-rescore" / "events.log")
    for path in paths:
        if not path.exists():
            continue
        source_events = []
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            match = pattern.match(line)
            if match:
                timestamp, phase, done, total, state = match.groups()
                phase = display_text(phase)
                source_events.append((datetime.fromisoformat(timestamp), phase, int(done), int(total), state))
        receipt_starts = [index for index, item in enumerate(source_events)
                          if item[1] == "Stage 1 · Verify input receipts" and item[2] == 0]
        latest_receipt = receipt_starts[-1] if receipt_starts else None
        events.extend(item for index, item in enumerate(source_events)
                      if item[1] != "Stage 1 · Verify input receipts" or latest_receipt is None
                      or index >= latest_receipt)
    if not events:
        return {}
    events.sort(key=lambda item: item[0])
    records, active = {}, None

    def close(end, completed=False):
        nonlocal active
        if active is None:
            return
        record = records.setdefault(active["phase"], {"seconds": 0., "complete": False,
                                                        "started": active["start"], "segments": 0})
        record["seconds"] += max(0., (end - active["start"]).total_seconds())
        record["complete"] |= completed
        record["started"] = min(record["started"], active["start"])
        record["segments"] += 1
        active = None

    for timestamp, phase, done, total, state in events:
        if active and active["phase"] != phase:
            close(timestamp)
        if active is None:
            active = {"phase": phase, "start": timestamp}
        if done >= total or state != "running":
            close(timestamp, completed=state == "complete" or (state == "running" and total > 0 and done >= total))
    if active:
        close(now)
    return records


def plan_tree():
    def leaf(key, title, phase=None, artifact=None, decision=None, training_report=None):
        return {"id": key, "title": title, "phase": phase, "artifact": artifact,
                "decision": decision, "training_report": training_report}

    stage1 = [leaf("receipts", "Verify input receipts", "Verify input receipts"),
              leaf("waves", "Restore validation audio", artifact="validation-waveforms")]
    for model in ("open-small", "same-s", "same-l", "same-l-embedded", "dac"):
        stage1.append({"id": model, "title": f"{model} codec", "children": [
            leaf(f"{model}-recon", "Recover missing reconstructions", f"Reconstruct with {model}",
                 artifact=f"../issue187-rescore/waveforms-{model}.json"),
            leaf(f"{model}-score", "Score waveform reconstruction", f"Score reconstruction with {model}",
                 artifact=f"../issue187-rescore/rescore-{model}.json"),
            leaf(f"{model}-cir", "Score CIR similarity", artifact=f"../issue187-rescore/cir-{model}.json"),
        ]})
    stage1.extend([
        leaf("stage1-results", "Save results; apply Stage 1 gate", artifact="stage1-published-receipt.json"),
    ])
    stage2 = [leaf("qualify-generator", "Select codec; qualify generator",
                   phase="-embedded", decision="stage1_selected"),
              leaf("latent-cache", "Cache decoder latents", "Cache decoder input latents")]
    for mode in ADAPTERS:
        name = CONFIGURATION_NAMES[mode]
        report = f"arm-{mode}.json"
        arm = [leaf(f"{mode}-load", "Initialize arm", f"Load {name} arm", artifact=report),
               leaf(f"{mode}-updates", f"Train adapter (up to {MAX_UPDATES:,} updates)", f"{name} updates",
                    training_report=report)]
        arm.extend(leaf(f"{mode}-val-{step}", f"Validation checkpoint · {step:,} updates",
                        f"{name} validation at {step:,}", artifact=f"validation-{mode}-{step}.json")
                    for step in MILESTONES)
        stage2.append({"id": f"{mode}-arm", "title": name, "children": arm})
    stage2.append(
        leaf("stage2-gate", "Compare to direct baseline; apply Stage 2 gate", artifact="stage2.json",
             decision="stage2_selected"),
    )
    stage3 = [leaf(f"stage3-{mode}", f"Optimize and render 192 cases · {CONFIGURATION_NAMES[mode]}",
                   f"Native Vital inversion with {CONFIGURATION_NAMES[mode]}", artifact=f"stage3-{mode}.json")
              for mode in ADAPTERS]
    stage3.append(leaf("stage3-summary", "Summarize inversion and diagnostics", artifact="stage3.json",
                       decision="stage3"))
    return [
        {"id": "stage1", "title": "Stage 1 · Codec and retrieval screening", "stage": 1,
         "children": stage1},
        {"id": "stage2", "title": "Stage 2 · Train and select adapters", "stage": 2,
         "children": stage2},
        {"id": "stage3", "title": "Stage 3 · Native Vital inversion", "stage": 3,
         "children": stage3},
    ]


def format_elapsed(seconds):
    if seconds is None:
        return "—"
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes:02}m {seconds:02}s"
    if minutes:
        return f"{minutes}m {seconds:02}s"
    return f"{seconds}s"


def read_learning_curves(output):
    curves = {"training": {}, "validation": {}}
    for mode in ADAPTERS:
        arm_path = output / f"arm-{mode}.json"
        try:
            arm = json.loads(arm_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            arm = {}
        training = [(int(row["update"]), float(row["loss"])) for row in arm.get("logs", [])
                    if int(row["update"]) <= MAX_UPDATES and math.isfinite(float(row["loss"]))]
        validation = []
        for path in output.glob(f"validation-{mode}-*.json"):
            try:
                step = int(path.stem.rsplit("-", 1)[1])
                if step > MAX_UPDATES:
                    continue
                metrics = json.loads(path.read_text(encoding="utf-8"))["metrics"]
                loss = float(metrics.get("mel_log1p", metrics["spectral_loss"]))
            except (OSError, ValueError, KeyError, TypeError):
                continue
            if math.isfinite(loss):
                validation.append((step, loss))
        if training:
            curves["training"][mode] = training
        if validation:
            validation.sort()
            curves["validation"][mode] = validation
    return curves


ADAPTERS = tuple(f"{arm}-{backbone}" for arm in ("latent", "generator")
                 for backbone in ("residual_mlp", "transformer"))
CONFIGURATION_NAMES = {
    "latent-residual_mlp": "Residual MLP + SAME-L",
    "latent-transformer": "Transformer + SAME-L",
    "generator-residual_mlp": "Residual MLP + Stable Audio 3 Medium",
    "generator-transformer": "Transformer + Stable Audio 3 Medium",
}
MAX_UPDATES = 6_000
MILESTONES = tuple(range(0, MAX_UPDATES + 1, 500))


def display_text(value):
    for internal, display in CONFIGURATION_NAMES.items():
        value = value.replace(internal, display)
    return value


def training_status(report, active):
    if report is None:
        return None
    if active:
        return "In progress"
    if report.get("updates", 0) == MAX_UPDATES:
        return "Complete"
    if report.get("updates", 0) > MAX_UPDATES:
        return "Checkpoint"
    if report.get("stopped"):
        return "Stopped"
    return "Checkpoint"


def main(output):
    set_dpi_awareness()
    root = tk.Tk()
    root.title("OBRUXO · Issue 187 Vital surrogates")
    screen_width, screen_height = root.winfo_screenwidth(), root.winfo_screenheight()
    window_width, window_height = int(screen_width * .94), int(screen_height * .86)
    panel_width = int(window_width * .52)
    dashboard_width = window_width - panel_width
    root.geometry(f"{window_width}x{window_height}+0+0")
    style = ttk.Style(root)
    body_font = tkfont.Font(root, family="Segoe UI", size=10)
    heading_font = tkfont.Font(root, family="Segoe UI", size=10, weight="bold")
    style.configure("Treeview", font=body_font, rowheight=body_font.metrics("linespace") + 10)
    style.configure("Treeview.Heading", font=heading_font)
    style.configure("TLabel", font=body_font)
    legend_font = tkfont.Font(root, family="Segoe UI", size=9)
    root.columnconfigure(0, weight=1)
    root.rowconfigure(0, weight=1)
    split = tk.PanedWindow(root, orient=tk.HORIZONTAL, sashwidth=14, sashpad=3, sashrelief="raised",
                           sashcursor="sb_h_double_arrow", showhandle=True, handlesize=16, handlepad=4,
                           opaqueresize=True, borderwidth=0, relief="flat", background="#b5b5b5")
    split.grid(row=0, column=0, sticky="nsew")

    side = ttk.Frame(split, padding=10, width=panel_width)
    side.rowconfigure(2, weight=1)
    side.columnconfigure(0, weight=1)
    ttk.Label(side, text="Study plan", font=tkfont.Font(root, family="Segoe UI", size=13, weight="bold")).grid(
        row=0, column=0, sticky="w", pady=(0, 2))
    ttk.Label(side, text="Drag the divider or column edges to resize.").grid(row=1, column=0, sticky="w", pady=(0, 6))
    tree = ttk.Treeview(side, columns=("state", "elapsed"), show="tree headings", selectmode="browse")
    tree.heading("#0", text="Task / subtask", anchor="w")
    tree.heading("state", text="State", anchor="w")
    tree.heading("elapsed", text="Elapsed", anchor="e")
    status_tags = {"Complete": "complete", "In progress": "in_progress", "Checkpoint": "checkpoint",
                   "Stopped": "stopped", "Failed": "stopped", "Skipped": "muted", "Not run": "muted",
                   "Upcoming": "upcoming"}
    tree.tag_configure("complete", background="#E8F5E9", foreground="#1B5E20")
    tree.tag_configure("in_progress", background="#E3F2FD", foreground="#0D47A1")
    tree.tag_configure("checkpoint", background="#FFF8E1", foreground="#8A5700")
    tree.tag_configure("stopped", background="#FFEBEE", foreground="#B71C1C")
    tree.tag_configure("muted", background="#F1F3F4", foreground="#616161")
    tree.tag_configure("upcoming", background="#FFFFFF", foreground="#333333")

    def set_row_status(iid, status):
        tree.set(iid, "state", status)
        tree.item(iid, tags=(status_tags.get(status, "upcoming"),))

    tree_width = max(400, panel_width - 48)
    task_width = int(tree_width * .54)
    state_width = int(tree_width * .23)
    tree.column("#0", width=task_width, minwidth=200, stretch=True)
    tree.column("state", width=state_width, minwidth=90, stretch=True)
    tree.column("elapsed", width=tree_width - task_width - state_width, minwidth=95, stretch=True, anchor="e")
    scrollbar = ttk.Scrollbar(side, orient="vertical", command=tree.yview)
    tree.configure(yscrollcommand=scrollbar.set)
    tree.grid(row=2, column=0, sticky="nsew")
    scrollbar.grid(row=2, column=1, sticky="ns", padx=(5, 14))

    dashboard = ttk.Frame(split, padding=16, width=dashboard_width)
    dashboard.columnconfigure(0, weight=1)
    dashboard.rowconfigure(7, weight=1)
    split.add(side, minsize=max(360, int(window_width * .3)), stretch="always", padx=3, pady=3)
    split.add(dashboard, minsize=max(360, int(window_width * .3)), stretch="always", padx=3, pady=3)
    phase = ttk.Label(dashboard, text="Waiting for the worker",
                      font=tkfont.Font(root, family="Segoe UI", size=14, weight="bold"),
                      wraplength=int(dashboard_width * .88))
    phase.grid(row=0, column=0, sticky="w")
    detail = ttk.Label(dashboard)
    detail.grid(row=1, column=0, sticky="w", pady=(6, 0))
    phase_bar = ttk.Progressbar(dashboard)
    phase_bar.grid(row=2, column=0, sticky="ew", pady=(10, 8))
    overall = ttk.Label(dashboard)
    overall.grid(row=3, column=0, sticky="w")
    stage_bar = ttk.Progressbar(dashboard, maximum=3)
    stage_bar.grid(row=4, column=0, sticky="ew", pady=(6, 18))
    events = ttk.Label(dashboard, justify="left", wraplength=int(dashboard_width * .88))
    events.grid(row=5, column=0, sticky="nw")
    ttk.Label(dashboard, text="Training and validation curves", font=heading_font).grid(
        row=6, column=0, sticky="w", pady=(10, 2))
    curve_canvas = tk.Canvas(dashboard, height=430, background="white", highlightthickness=1,
                             highlightbackground="#b5b5b5")
    curve_canvas.grid(row=7, column=0, sticky="nsew")
    controls = ttk.Frame(dashboard)
    controls.grid(row=8, column=0, sticky="w", pady=(10, 0))
    paused = False
    learning_curves = {"training": {}, "validation": {}}

    def draw_plot(title, series, y, height):
        width = max(200, curve_canvas.winfo_width())
        curve_canvas.create_text(10, y + 2, text=title, anchor="nw", font=heading_font)
        points = [(x, value) for values in series.values() for x, value in values]
        if not points:
            curve_canvas.create_text(10, y + height / 2, text="Waiting for Stage 2 data", anchor="w",
                                     fill="#555555", font=body_font)
            return
        x_max = max(1, max(x for x, _ in points))
        first_update = min(x for x, _ in points)
        later = {mode: [(x, value) for x, value in values if x > first_update]
                 for mode, values in series.items()}
        detail_points = [(x, value) for values in later.values() for x, value in values]
        colors = ("#1464a5", "#7741a7", "#c45820", "#268062")
        names = CONFIGURATION_NAMES
        slot = (width - 20) / 2
        legend_width = max(90, int(slot - 30))
        wrapped = any(legend_font.measure(names.get(mode, mode)) > legend_width for mode in series)
        legend_row_height = 32 if wrapped else 21
        legend_y = y + 38
        for index, mode in enumerate(series):
            color = colors[index % len(colors)]
            legend_x = 10 + (index % 2) * slot
            row_y = legend_y + (index // 2) * legend_row_height
            curve_canvas.create_line(legend_x, row_y, legend_x + 18, row_y, fill=color, width=2)
            curve_canvas.create_text(legend_x + 23, row_y, text=names.get(mode, mode), anchor="w",
                                     justify="left", width=legend_width, fill=color, font=legend_font)

        def draw_panel(panel_x, panel_width, label, panel_series):
            panel_points = [(x, value) for values in panel_series.values() for x, value in values]
            if not panel_points:
                curve_canvas.create_text(panel_x + 8, y + 68, text=f"{label}: no later points",
                                         anchor="nw", fill="#555555", font=body_font)
                return
            y_min, y_max = min(value for _, value in panel_points), max(value for _, value in panel_points)
            padding = max((y_max - y_min) * .12, abs(y_max) * .02, 1e-8)
            y_min, y_max = y_min - padding, y_max + padding
            y_ticks = [y_min + fraction * (y_max - y_min) for fraction in (0., .5, 1.)]
            y_labels = [f"{value:.3g}" for value in y_ticks]
            left = panel_x + max(48, max(body_font.measure(tick) for tick in y_labels) + 12)
            heading_y = y + (82 if wrapped else 72)
            right, top, bottom = panel_x + panel_width - 8, heading_y + 23, y + height - 32
            curve_canvas.create_text(panel_x + 8, heading_y, text=label, anchor="nw", fill="#444444",
                                     font=body_font)
            for fraction, tick in zip((0., .5, 1.), y_labels, strict=True):
                grid_y = bottom - fraction * (bottom - top)
                curve_canvas.create_line(left, grid_y, right, grid_y, fill="#e1e1e1")
                curve_canvas.create_text(left - 6, grid_y, text=tick, anchor="e", fill="#555555",
                                         font=body_font)
            curve_canvas.create_line(left, top, left, bottom, fill="#777777")
            curve_canvas.create_line(left, bottom, right, bottom, fill="#777777")
            for index, (mode, values) in enumerate(panel_series.items()):
                coords = []
                for step, value in values:
                    px = left + step / x_max * (right - left)
                    py = bottom - (value - y_min) / (y_max - y_min) * (bottom - top)
                    coords.extend((px, py))
                color = colors[index % len(colors)]
                if len(coords) >= 4:
                    curve_canvas.create_line(*coords, fill=color, width=2)
                for px, py in zip(coords[::2], coords[1::2], strict=True):
                    curve_canvas.create_oval(px - 3, py - 3, px + 3, py + 3, fill=color, outline=color)
            for fraction in (0., .5, 1.):
                tick_x = left + fraction * (right - left)
                curve_canvas.create_line(tick_x, bottom, tick_x, bottom + 5, fill="#777777")
                curve_canvas.create_text(tick_x, bottom + 7, text=f"{fraction * x_max:,.0f}", anchor="n",
                                         fill="#555555", font=body_font)

        gap = 12
        panel_width = (width - 20 - gap) / 2
        draw_panel(10, panel_width, "Full y range", series)
        draw_panel(10 + panel_width + gap, panel_width, "Detail · after first update", later if detail_points else series)

    def redraw_curves(_event=None):
        curve_canvas.delete("all")
        height = max(160, curve_canvas.winfo_height() // 2)
        draw_plot("Training objective", learning_curves["training"], 0, height)
        draw_plot("Validation MEL-log1p", learning_curves["validation"], height, height)
    curve_canvas.bind("<Configure>", redraw_curves)
    specs, items = plan_tree(), {}

    def insert(nodes, parent=""):
        for node in nodes:
            iid = tree.insert(parent, "end", iid=node["id"], text=node["title"], values=("Upcoming", "—"),
                              tags=("upcoming",))
            items[iid] = node
            insert(node.get("children", []), iid)
    insert(specs)
    def open_tree(nodes):
        for node in nodes:
            tree.item(node["id"], open=True)
            open_tree(node.get("children", []))
    open_tree(specs)

    def refresh_tree(state, records, decisions, now):
        current_stage = state.get("stage_index", 0)
        elapsed_values = {}
        for iid, node in items.items():
            if node.get("stage"):
                stage = node["stage"]
                stage_events = [row for phase_name, row in records.items()
                                if phase_name.startswith(f"Stage {stage} ·")]
                if state.get("state") in ("complete", "failed") and stage <= current_stage:
                    status = "Complete" if state.get("state") == "complete" else "Stopped"
                elif stage < current_stage:
                    status = "Complete"
                elif stage == current_stage and state.get("state") == "running":
                    status = "In progress"
                elif stage == current_stage and state.get("state") == "failed":
                    status = "Stopped"
                else:
                    status = "Checkpoint" if stage_events else "Upcoming"
                seconds = sum(row["seconds"] for row in stage_events)
                stage_start = min((row["started"] for row in stage_events if row["started"]), default=None)
                if status == "In progress" and stage_start:
                    seconds = max(seconds, (now - stage_start).total_seconds())
                if status == "Complete" and stage_start:
                    next_events = [row["started"] for name, row in records.items()
                                   if name.startswith(f"Stage {stage + 1} ·") and row["started"]]
                    if next_events:
                        seconds = (min(next_events) - stage_start).total_seconds()
                set_row_status(iid, status)
                tree.set(iid, "elapsed", format_elapsed(seconds) if stage_events else "—")
                elapsed_values[iid] = seconds
                continue
            children = node.get("children")
            if children:
                statuses = [tree.set(child["id"], "state") for child in children]
                if statuses and all(value == "Complete" for value in statuses):
                    status = "Complete"
                elif statuses and all(value in ("Skipped", "Not run") for value in statuses):
                    status = "Skipped"
                elif "In progress" in statuses:
                    status = "In progress"
                elif "Stopped" in statuses:
                    status = "Stopped"
                elif "Checkpoint" in statuses:
                    status = "Checkpoint"
                elif any(value == "Complete" for value in statuses):
                    status = "In progress"
                elif node.get("id") == "generator-arm" and decisions.get("generator"):
                    status = "Not run"
                elif any(value not in ("Upcoming", "Not run", "Skipped") for value in statuses):
                    status = "In progress"
                else:
                    status = "Upcoming"
                elapsed = sum(elapsed_values.get(child["id"], 0.) for child in children)
                set_row_status(iid, status)
                tree.set(iid, "elapsed", format_elapsed(elapsed) if elapsed else "—")
                elapsed_values[iid] = elapsed
                continue

            status, seconds = "Upcoming", None
            phase_match = node.get("phase")
            matching = [(name, record) for name, record in records.items()
                        if phase_match and phase_match in name]
            artifact_complete = bool(node.get("artifact") and (output / node["artifact"]).exists())
            if node.get("artifact"):
                artifact = output / node["artifact"]
                if artifact_complete:
                    status = "Complete"
                    seconds = 0.
                if node.get("decision") and decisions:
                    decision = decisions.get(node["decision"])
                    if node["id"] == "qualify-generator" and decision:
                        status = "Complete"
                    elif decision:
                        status = "Complete"
            if matching:
                record = max(matching, key=lambda pair: pair[1]["started"] or now)[1]
                seconds = sum(item["seconds"] for _, item in matching)
                if not artifact_complete:
                    status = "Complete" if record["complete"] else "Checkpoint"
                if phase_match and phase_match in state.get("phase", ""):
                    if not artifact_complete:
                        status = "In progress"
                        seconds = max(seconds, (now - (record["started"] or now)).total_seconds())
            if node.get("training_report"):
                report_path = output / node["training_report"]
                try:
                    report = json.loads(report_path.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    report = None
                active = (node.get("phase") in state.get("phase", "") and state.get("state") == "running")
                status = training_status(report, active) or status
                if report:
                    seconds = max(seconds, float(report.get("elapsed_seconds", 0.) or 0.))
            if node["id"] == "stage2-gate":
                stage2_state = decisions.get("stage2")
                if stage2_state in ("analysis_required", "complete"):
                    status = "Complete"
                elif stage2_state in ("incomplete", "blocked_missing_verified_audio"):
                    status = "Stopped"
                else:
                    status = "In progress" if stage2_state == "running" else "Upcoming"
            elif node["id"] == "stage3-summary":
                stage3_state = decisions.get("stage3")
                status = {"complete": "Complete", "skipped": "Skipped", "running": "In progress"}.get(
                    stage3_state, "Upcoming")
            elif node["id"].startswith("stage3-"):
                completed = set(decisions.get("completed_configurations", []))
                if decisions.get("stage2") in ("analysis_required", "complete", "incomplete") \
                        and node["id"].removeprefix("stage3-") not in completed:
                    status = "Skipped"
            set_row_status(iid, status)
            tree.set(iid, "elapsed", format_elapsed(seconds))
            elapsed_values[iid] = seconds or 0.

    def refresh():
        nonlocal learning_curves
        if not paused:
            try:
                now = datetime.now(timezone.utc)
                state = json.loads((output / "status.json").read_text(encoding="utf-8"))
                done, total = state["completed"], max(1, state["total"])
                stage, count = state["stage_index"], state["stage_count"]
                updated = datetime.fromisoformat(state["updated_at"]).astimezone().strftime("%H:%M:%S")
                phase.config(text=f"{display_text(state['phase'])} · {state['state']}")
                detail.config(text=f"{done:,} / {total:,} · {state['failures']:,} failures · updated {updated}")
                phase_bar.config(maximum=total, value=done)
                overall.config(text=f"Stage {stage} of {count}")
                stage_bar.config(maximum=count, value=(count if state["state"] == "complete" else stage - 1))
                with (output / "events.log").open("rb") as stream:
                    stream.seek(max(0, stream.seek(0, 2) - 8192))
                    tail = stream.read().decode("utf-8", errors="replace").splitlines()[-5:]
                events.config(text="Recent events\n" + "\n".join(display_text(line) for line in tail))
                decisions_path = output / "decisions.json"
                decisions = json.loads(decisions_path.read_text(encoding="utf-8")) if decisions_path.exists() else {}
                refresh_tree(state, read_timeline(output, now), decisions, now)
                learning_curves = read_learning_curves(output)
                redraw_curves()
            except (OSError, ValueError, KeyError, TypeError):
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
