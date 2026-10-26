"""Read-only diagnostics of the already verified basic training mechanism."""

from collections import Counter, defaultdict
import json
from statistics import mean


def pooled_pitch(pitch):
    if tuple(pitch.shape) != (3521,):
        raise ValueError("pooled pitch audit requires the saved basic feature layout")
    notes = pitch[:704].reshape(88, 8).mean(1)
    contour = pitch[1408:3520].reshape(264, 8).mean(1)
    return {"note_top_midi": 21 + int(notes.argmax()) if float(notes.max()) > 1e-8 else None,
            "contour_peak_midi": 21 + int(contour.argmax()) / 3 if float(contour.max()) > 1e-8 else None,
            "note_top_probability": float(notes.max())}


def exposures(plan):
    counts = Counter((example["state"], example["input_view"]) for update in plan["updates"] for example in update["examples"])
    return [{"training_index": state, "input_view": view, "scheduled_primary_observations": count}
            for (state, view), count in sorted(counts.items())]


def joint_probes(plan, probes, finals):
    measured = defaultdict(dict)
    for row in probes:
        query = row["query"]
        cell = plan["fit_audit_cells"][query["index"]]
        if cell["training_index"] is None or cell["input_view"] not in (1, 2):
            continue
        candidate = json.dumps(query["candidate"], sort_keys=True, separators=(",", ":"), allow_nan=False)
        key = cell["training_index"], cell["input_view"], candidate
        playback = cell["playback_view"]
        if playback in measured[key] and measured[key][playback] != row["loss"]:
            raise ValueError("the same fixed-phase candidate/playback has inconsistent saved costs")
        measured[key][playback] = row["loss"]
    final = {}
    for row in finals:
        cell = plan["fit_audit_cells"][row["index"]]
        if cell["training_index"] is not None:
            key = cell["training_index"], cell["input_view"], cell["playback_view"], row["arm"]
            if key in final:
                raise ValueError("duplicate final playback cost")
            final[key] = row["scores"]["spectral"]
    result = {}
    for view in (1, 2):
        states = sorted({state for state, input_view, _candidate in measured if input_view == view})
        rows = []
        for state in states:
            available = [costs for (source, input_view, _candidate), costs in measured.items() if source == state and input_view == view]
            complete = [mean(costs.values()) for costs in available if set(costs) == {0, view}]
            if not complete:
                raise ValueError("every input patch needs a candidate measured under both performances")
            best = min(complete)
            control = mean(final[state, view, playback, "parameter"] for playback in (0, view))
            model = mean(final[state, view, playback, "spectral"] for playback in (0, view))
            rows.append({"training_index": state, "complete_measured_candidates": len(complete),
                         "incomplete_candidates_not_used": len(available) - len(complete), "best_probed_joint_loss": best,
                         "best_probed_or_final_sound_loss": min(best, model),
                         "parameter_final_joint_loss": control, "spectral_final_joint_loss": model,
                         "probe_beats_parameter": best < control, "final_worse_than_probe": model > best + 1e-6})
        if not rows:
            raise ValueError("both wider input types are required")
        result[str(view)] = {"patches": len(rows), "rows": rows,
                             "best_probed_joint_mean": mean(row["best_probed_joint_loss"] for row in rows),
                             "best_probed_or_final_sound_mean": mean(row["best_probed_or_final_sound_loss"] for row in rows),
                             "parameter_final_joint_mean": mean(row["parameter_final_joint_loss"] for row in rows),
                             "spectral_final_joint_mean": mean(row["spectral_final_joint_loss"] for row in rows),
                             "probe_beats_parameter": sum(row["probe_beats_parameter"] for row in rows),
                             "final_worse_than_probe": sum(row["final_worse_than_probe"] for row in rows)}
    return result
