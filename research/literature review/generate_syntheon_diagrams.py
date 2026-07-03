#!/usr/bin/env python3
"""Generate diagrams for syntheon_notes.md.

Policy: only render an image where ASCII cannot carry the content (complex
branching or signal-domain shapes that must be drawn). Everything else stays as
fenced-```text``` in the notes. Currently that means just the modulation gap,
which draws actual LFO / envelope waveforms.

Outputs PNGs to images/ next to this script, matching the house style of the
lfo_representation report images.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).parent / "images"
OUT.mkdir(exist_ok=True)

INK = "#1f2328"
INK_SOFT = "#3c434a"
ACCENT = "#2b6cb0"      # inherit / static / envelope
REJECT = "#b42318"      # gap
EXTEND = "#7c3aed"      # ours / LFO
LINE = "#6b7280"


def diagram_modulation_gap():
    """Static knob vector (left) vs static + modulation curves (right)."""
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12, 5.2))

    # ---- left: static knob vector (Syntheon) ----
    axL.set_title("What Syntheon emits\n(static knob vector)", fontsize=11,
                  color=INK, weight="bold")
    axL.set_xlim(0, 10); axL.set_ylim(0, 10)
    axL.set_axis_off()
    labels = ["osc1", "osc2", "flt", "atk", "dec", "sus", "rel", "fx1", "fx2"]
    vals = [0.7, 0.4, 0.6, 0.3, 0.5, 0.8, 0.35, 0.55, 0.25]
    for i, (lab, v) in enumerate(zip(labels, vals)):
        x = 1 + i
        axL.plot([x, x], [1, 1 + v * 6], color=ACCENT, lw=8,
                 solid_capstyle="round", alpha=0.85)
        axL.plot([x - 0.35, x + 0.35], [1 + v * 6, 1 + v * 6],
                 color=INK, lw=1.5)
        axL.text(x, 0.4, lab, ha="center", va="top", fontsize=8, color=INK_SOFT)
    axL.plot([0.3, 9.7], [1, 1], color=LINE, lw=1)
    axL.text(5, 9.2, "every value is a fixed scalar", ha="center",
             fontsize=9, color=INK_SOFT, style="italic")
    axL.text(5, 8.4, "no time axis, no curves", ha="center",
             fontsize=9, color=REJECT, weight="bold")

    # ---- right: full Vital sound (modulation) ----
    axR.set_title("What an evolving Vital sound needs\n(static params + modulation)",
                  fontsize=11, color=INK, weight="bold")
    axR.set_xlim(0, 10); axR.set_ylim(0, 10)
    axR.set_axis_off()

    # static sliders, dimmed (still present)
    for i, (lab, v) in enumerate(zip(labels, vals)):
        x = 1 + i
        axR.plot([x, x], [1, 1 + v * 2.2], color=INK_SOFT, lw=6,
                 solid_capstyle="round", alpha=0.25)
        axR.text(x, 0.4, lab, ha="center", va="top", fontsize=7,
                 color="#b0b6bc")
    axR.text(5, 3.9, "static params (Syntheon's ceiling)",
             ha="center", fontsize=8, color=INK_SOFT, style="italic")

    # modulation curves drawn for real
    t = np.linspace(0, 2 * np.pi, 200)
    xlfo = np.linspace(1, 9, 200)
    lfo = 6.8 + 1.0 * np.sin(2 * t)
    axR.plot(xlfo, lfo, color=EXTEND, lw=2.2)
    axR.text(1.1, 8.3, "LFO curve", color=EXTEND, fontsize=8.5, weight="bold")

    # envelope: ADSR shape
    xe = np.linspace(1, 9, 200)
    seg = np.array([0.0, 0.6, 0.4, 0.4, 0.0])
    xe_seg = np.linspace(1, 9, len(seg))
    env = 5.0 + np.interp(xe, xe_seg, seg) * 1.4
    axR.plot(xe, env, color=ACCENT, lw=2.2)
    axR.text(1.1, 7.0, "envelope / automation", color=ACCENT,
             fontsize=8.5, weight="bold")

    # routing annotation
    axR.annotate("", xy=(8.6, 5.2), xytext=(8.6, 6.8),
                 arrowprops=dict(arrowstyle="<->", color=LINE, lw=1.2))
    axR.text(8.95, 6.0, "routing\n+ amount", fontsize=8, color=INK_SOFT,
             ha="left", va="center")

    fig.suptitle("The modulation gap", fontsize=13, weight="bold",
                 color=INK, y=1.0)
    fig.tight_layout()
    fig.savefig(OUT / "syntheon_modulation_gap.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    diagram_modulation_gap()
    print(f"wrote {len(list(OUT.glob('*.png')))} PNG(s) to {OUT}")
