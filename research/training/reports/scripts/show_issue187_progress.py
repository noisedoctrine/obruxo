"""Display the Issue 187 worker status without controlling the worker."""

from datetime import datetime, timezone
import json
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
    path = output / "events.log"
    if not path.exists():
        return {}
    pattern = re.compile(r"^(\S+) (.*?): (\d+)/(\d+) \(([^)]+)\)")
    events = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = pattern.match(line)
        if match:
            timestamp, phase, done, total, state = match.groups()
            events.append((datetime.fromisoformat(timestamp), phase, int(done), int(total), state))
    starts = [index for index, item in enumerate(events)
              if item[1] == "Stage 1 · Verify input receipts" and item[2] == 0]
    events = events[starts[-1]:] if starts else events
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
            close(timestamp, completed=state == "complete" or (total > 0 and done >= total))
    if active:
        close(now)
    return records


def plan_tree():
    def leaf(key, title, phase=None, artifact=None, decision=None):
        return {"id": key, "title": title, "phase": phase, "artifact": artifact, "decision": decision}

    stage1 = [leaf("receipts", "Verify input receipts", "Verify input receipts"),
              leaf("waves", "Restore validation audio", "Recover frozen validation waveforms")]
    for model in ("open-small", "same-s", "same-l"):
        stage1.append({"id": model, "title": f"{model} codec", "children": [
            leaf(f"{model}-recon", "Reconstruct validation pairs", f"Reconstruct with {model}"),
            leaf(f"{model}-retrieval", "Encode features; score retrieval", f"Cross-MIDI features with {model}"),
        ]})
    stage1.extend([
        leaf("stage1-results", "Save results; apply Stage 1 gate", artifact="stage1.json"),
    ])
    stage2 = [leaf("qualify-generator", "Select codec; qualify generator",
                   phase="-embedded", decision="stage1_selected"),
              leaf("latent-cache", "Cache decoder latents", "Cache decoder input latents")]
    for mode in ("latent", "generator"):
        arm = [leaf(f"{mode}-load", "Initialize arm", f"Load {mode} arm"),
               leaf(f"{mode}-updates", "Train adapter (up to 8,000 updates)", f"{mode} updates")]
        arm.extend(leaf(f"{mode}-val-{step}", f"Validation checkpoint · {step:,} updates",
                        f"{mode} validation at {step:,}") for step in (0, 500, 1000, 2000, 4000, 8000))
        stage2.append({"id": f"{mode}-arm", "title": f"{mode.title()} adapter arm", "children": arm})
    stage2.append(
        leaf("stage2-gate", "Compare to direct baseline; apply Stage 2 gate", artifact="stage2.json",
             decision="stage2_selected"),
    )
    stage3 = [
        leaf("inversion", "Optimize and render 192 inversion cases", "Native Vital inversion"),
        leaf("stage3-summary", "Summarize inversion and diagnostics", artifact="stage3.json",
             decision="stage3"),
    ]
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
    dashboard.rowconfigure(5, weight=1)
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
    controls = ttk.Frame(dashboard)
    controls.grid(row=6, column=0, sticky="w", pady=(10, 0))
    paused = False
    specs, items = plan_tree(), {}

    def insert(nodes, parent=""):
        for node in nodes:
            iid = tree.insert(parent, "end", iid=node["id"], text=node["title"], values=("Upcoming", "—"))
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
                if state.get("state") in ("complete", "failed") and stage <= current_stage:
                    status = "Complete" if state.get("state") == "complete" else "Stopped"
                elif stage < current_stage:
                    status = "Complete"
                elif stage == current_stage and state.get("state") == "running":
                    status = "In progress"
                elif stage == current_stage and state.get("state") == "failed":
                    status = "Stopped"
                else:
                    status = "Upcoming"
                stage_events = [row for phase_name, row in records.items() if phase_name.startswith(f"Stage {stage} ·")]
                seconds = sum(row["seconds"] for row in stage_events)
                stage_start = min((row["started"] for row in stage_events if row["started"]), default=None)
                if status == "In progress" and stage_start:
                    seconds = max(seconds, (now - stage_start).total_seconds())
                if status == "Complete" and stage_start:
                    next_events = [row["started"] for name, row in records.items()
                                   if name.startswith(f"Stage {stage + 1} ·") and row["started"]]
                    if next_events:
                        seconds = (min(next_events) - stage_start).total_seconds()
                tree.set(iid, "state", status)
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
                elif any(value in ("In progress", "Checkpoint") for value in statuses):
                    status = "In progress"
                elif any(value == "Complete" for value in statuses):
                    status = "In progress"
                elif node.get("id") == "generator-arm" and decisions.get("generator"):
                    status = "Not run"
                elif any(value not in ("Upcoming", "Not run", "Skipped") for value in statuses):
                    status = "In progress"
                else:
                    status = "Upcoming"
                elapsed = sum(elapsed_values.get(child["id"], 0.) for child in children)
                tree.set(iid, "state", status)
                tree.set(iid, "elapsed", format_elapsed(elapsed) if elapsed else "—")
                elapsed_values[iid] = elapsed
                continue

            status, seconds = "Upcoming", None
            phase_match = node.get("phase")
            matching = [(name, record) for name, record in records.items()
                        if phase_match and phase_match in name]
            if node.get("artifact"):
                artifact = output / node["artifact"]
                if artifact.exists():
                    status = "Complete"
                    seconds = 0.
                if node.get("decision") and decisions:
                    decision = decisions.get(node["decision"])
                    if node["id"] == "qualify-generator" and decision:
                        status = "Complete"
                    elif node["id"] == "stage2-gate" and decisions.get("stage3") == "skipped":
                        status = "Skipped"
                    elif node["id"] == "stage3-summary" and decision == "skipped":
                        status = "Skipped"
                    elif decision:
                        status = "Complete"
            if matching:
                record = max(matching, key=lambda pair: pair[1]["started"] or now)[1]
                seconds = sum(item["seconds"] for _, item in matching)
                status = "Complete" if record["complete"] else "Checkpoint"
                if phase_match and phase_match in state.get("phase", ""):
                    status = "In progress"
                    seconds = max(seconds, (now - (record["started"] or now)).total_seconds())
            if node["id"].startswith("generator-") and decisions.get("generator"):
                status = "Not run"
            if node["id"] == "inversion" and decisions.get("stage3") == "skipped":
                status = "Skipped"
            tree.set(iid, "state", status)
            tree.set(iid, "elapsed", format_elapsed(seconds))
            elapsed_values[iid] = seconds or 0.

    def refresh():
        if not paused:
            try:
                now = datetime.now(timezone.utc)
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
                    tail = stream.read().decode("utf-8", errors="replace").splitlines()[-5:]
                events.config(text="Recent events\n" + "\n".join(tail))
                decisions_path = output / "decisions.json"
                decisions = json.loads(decisions_path.read_text(encoding="utf-8")) if decisions_path.exists() else {}
                refresh_tree(state, read_timeline(output, now), decisions, now)
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
