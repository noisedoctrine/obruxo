#!/usr/bin/env python3
"""Generate diagrams for syntheon_notes.md.

Renders flow/lineage diagrams with matplotlib (no graphviz dependency).
Outputs PNGs to images/ next to this script, matching the house style of the
lfo_representation report images.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

# --- shared style -----------------------------------------------------------
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.edgecolor": "#9aa0a6",
})

OUT = Path(__file__).parent / "images"
OUT.mkdir(exist_ok=True)

INK = "#1f2328"        # primary text / borders
INK_SOFT = "#3c434a"
ACCENT = "#2b6cb0"     # inherit / forward
REJECT = "#b42318"     # reject / gap
EXTEND = "#7c3aed"     # extend / ours
PANEL = "#f3f4f6"      # light fill
PANEL_2 = "#e7ecf3"
LINE = "#6b7280"


def box(ax, x, y, w, h, text, *, fc=PANEL, ec=LINE, fontsize=9.5,
        fontcolor=INK, weight="normal", rounding=0.02, lw=1.2):
    """Rounded box centered at (x, y)."""
    bb = FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle=f"round,pad=0.005,rounding_size={rounding}",
        fc=fc, ec=ec, lw=lw, mutation_aspect=1,
    )
    ax.add_patch(bb)
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize,
            color=fontcolor, weight=weight, zorder=5)
    return bb


def arrow(ax, x0, y0, x1, y1, *, color=LINE, lw=1.4, style="-|>",
          connectionstyle="arc3,rad=0", ls="-"):
    a = FancyArrowPatch(
        (x0, y0), (x1, y1),
        arrowstyle=style, mutation_scale=14,
        color=color, lw=lw, connectionstyle=connectionstyle,
        linestyle=ls, zorder=3,
    )
    ax.add_patch(a)
    return a


def fig_base(width=10, height=6):
    fig, ax = plt.subplots(figsize=(width, height))
    ax.set_axis_off()
    ax.set_aspect("equal")
    return fig, ax


# --- Diagram 1: end-to-end pipeline -----------------------------------------
def diagram_pipeline():
    fig, ax = fig_base(12.5, 5.4)
    ax.set_xlim(0, 12.5)
    ax.set_ylim(0, 5.4)

    # title
    ax.text(0.2, 5.05, "Syntheon: audio \u2192 Vital preset",
            fontsize=13, weight="bold", color=INK)

    # input
    box(ax, 1.1, 2.4, 1.6, 0.8, "audio.wav\n(16 kHz, 4 s)", fc="#fff7e6", ec="#d9a441")

    # preprocessing (two stacked)
    box(ax, 3.5, 3.1, 2.0, 0.7, "CREPE (large)\n\u2192 f\u2080", fc=PANEL)
    box(ax, 3.5, 1.7, 2.0, 0.7, "A-weighted\nloudness", fc=PANEL)
    arrow(ax, 1.9, 2.55, 2.5, 3.05)
    arrow(ax, 1.9, 2.25, 2.5, 1.75)

    # inferencer
    box(ax, 6.6, 2.4, 2.5, 2.0,
        "Inferencer\n(WTSv2 / Diff-WTS)\n\n"
        "differentiable wavetable\nsynth + attention\n+ ADSR + reverb",
        fc=PANEL_2, ec=ACCENT, lw=1.6)
    arrow(ax, 4.5, 3.05, 5.35, 2.7, color=ACCENT)
    arrow(ax, 4.5, 1.75, 5.35, 2.1, color=ACCENT)

    # converter
    box(ax, 9.7, 2.4, 2.3, 2.0,
        "Converter\n(Vital-specific)\n\n"
        "attack/decay = quartic\nsustain = linear\nwavetables \u2192 Base64",
        fc=PANEL, ec=LINE)
    arrow(ax, 7.85, 2.4, 8.55, 2.4, color=ACCENT)
    ax.text(8.2, 2.62, "param dict", fontsize=8, color=INK_SOFT, ha="center")

    # output
    box(ax, 11.9, 2.4, 1.0, 0.8, "preset\n.vital", fc="#eaf6ec", ec="#2e7d32")
    arrow(ax, 10.85, 2.4, 11.4, 2.4)

    # training annotation: multiscale STFT loss
    box(ax, 6.6, 0.45, 4.6, 0.6,
        "training: multiscale STFT loss  [4096 \u2026 128], Hann, 75% overlap",
        fc="#eef2ff", ec=ACCENT, fontsize=8.5)
    arrow(ax, 6.6, 0.78, 6.6, 1.38, color=ACCENT, style="-|>", ls="--")
    ax.text(6.78, 1.08, "differentiable\nspectral loss", fontsize=7.5,
            color=ACCENT, ha="left", va="center")

    fig.tight_layout()
    fig.savefig(OUT / "syntheon_pipeline.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


# --- Diagram 2: conceptual lineage ------------------------------------------
def diagram_lineage():
    fig, ax = fig_base(12, 6.4)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6.4)

    ax.text(0.2, 6.05, "Where Syntheon's ideas come from",
            fontsize=13, weight="bold", color=INK)

    # four sources across the top, weighted by contribution (arrow thickness)
    sources = [
        # (x, label, weight 1-5, color)
        (1.6, "DDSP\nEngel et al. 2020", 3, INK),
        (4.6, "Diff. Wavetable Synthesis\nShan et al. ICASSP 2022", 5, ACCENT),
        (7.6, "CREPE\nKim et al. 2018", 2, INK),
        (10.4, "DiffSynth\nMasuda & Shimamura 2021", 2, INK),
    ]
    syn_y = 3.4
    for x, label, _, _ in sources:
        box(ax, x, 5.2, 2.4, 0.95, label, fc=PANEL, fontsize=8.5)

    # weighted arrows into Syntheon
    for x, _, w, col in sources:
        arrow(ax, x, 4.72, 6.0, syn_y + 0.45, color=col,
              lw=1.0 + w * 0.55, connectionstyle=f"arc3,rad={0.08 if x < 6 else -0.08}")

    box(ax, 6.0, syn_y, 3.0, 0.9,
        "Syntheon / WTSv2\n(gudgud96 \u2014 ADC22 talk)",
        fc=PANEL_2, ec=ACCENT, lw=1.6, weight="bold")

    # the gap
    arrow(ax, 6.0, syn_y - 0.45, 6.0, 2.15, color=REJECT, lw=2.2, ls="--")
    ax.text(6.5, 2.75, '"static-only" gap', fontsize=9, color=REJECT,
            weight="bold", ha="left")

    # WASPAA 2025 closes the gap
    box(ax, 6.0, 1.7, 4.2, 0.85,
        "Modulation Discovery\nLFO-net + B\u00e9zier / LPF curves\n(Mitcheltree, Tan, Reiss \u2014 WASPAA 2025)",
        fc="#fdf0f0", ec=REJECT, fontsize=8.5)
    ax.text(8.5, 1.7, "same author (Tan),\nsame DDSP foundation",
            fontsize=7.5, color=INK_SOFT, ha="left", va="center")

    # OBRUXO
    box(ax, 6.0, 0.5, 4.6, 0.7,
        "OBRUXO \u2014 playable .vital patches with full modulation",
        fc="#f3eefe", ec=EXTEND, lw=1.6, weight="bold", fontsize=9)
    arrow(ax, 6.0, 1.28, 6.0, 0.88, color=EXTEND, lw=1.8)

    fig.tight_layout()
    fig.savefig(OUT / "syntheon_lineage.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


# --- Diagram 3: the modulation gap ------------------------------------------
def diagram_modulation_gap():
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
    # LFO: sine, plotted in the modulation region
    xlfo = np.linspace(1, 9, 200)
    lfo = 6.8 + 1.0 * np.sin(2 * t)
    axR.plot(xlfo, lfo, color=EXTEND, lw=2.2, label="LFO (sine)")
    axR.text(1.1, 8.3, "LFO curve", color=EXTEND, fontsize=8.5, weight="bold")

    # envelope: ADSR shape
    xe = np.linspace(1, 9, 200)
    # quick attack, decay, sustain, release-ish normalized to 4.8-6.0 band
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

    fig.suptitle("The modulation gap", fontsize=13, weight="bold", color=INK, y=1.0)
    fig.tight_layout()
    fig.savefig(OUT / "syntheon_modulation_gap.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


# --- Diagram 4: inherit / reject / extend scope -----------------------------
def diagram_scope():
    fig, ax = fig_base(10.5, 6.4)
    ax.set_xlim(0, 10.5)
    ax.set_ylim(0, 6.4)

    ax.text(0.2, 6.05, "Where Syntheon sits in the Vital sound space",
            fontsize=13, weight="bold", color=INK)

    # outer = full vital sound space
    outer = FancyBboxPatch((0.6, 0.7), 9.3, 4.8,
                           boxstyle="round,pad=0.01,rounding_size=0.12",
                           fc="#fafafa", ec=LINE, lw=1.4)
    ax.add_patch(outer)
    ax.text(0.9, 5.2, "full Vital sound space", fontsize=10,
            color=INK_SOFT, weight="bold")

    # inner = Syntheon (static params)
    inner = FancyBboxPatch((1.6, 2.2), 3.2, 2.4,
                           boxstyle="round,pad=0.01,rounding_size=0.1",
                           fc=PANEL_2, ec=ACCENT, lw=1.6)
    ax.add_patch(inner)
    ax.text(3.2, 3.8, "Syntheon", ha="center", fontsize=11,
            color=ACCENT, weight="bold")
    ax.text(3.2, 3.25, "static params only", ha="center", fontsize=8.5,
            color=INK_SOFT)

    # OBRUXO extension region
    ext = FancyBboxPatch((5.4, 1.5), 4.0, 3.6,
                         boxstyle="round,pad=0.01,rounding_size=0.1",
                         fc="#f3eefe", ec=EXTEND, lw=1.6, ls="--")
    ax.add_patch(ext)
    ax.text(7.4, 4.7, "OBRUXO extends here", ha="center", fontsize=10,
            color=EXTEND, weight="bold")
    for i, txt in enumerate(["modulation curves", "routing + amounts",
                             "variable-length / evolving sounds"]):
        ax.text(7.4, 3.9 - i * 0.55, "\u2022 " + txt, ha="center",
                fontsize=9, color=INK)

    # arrow from inner to ext
    arrow(ax, 4.8, 3.4, 5.35, 3.4, color=EXTEND, lw=1.8, ls="--")

    fig.tight_layout()
    fig.savefig(OUT / "syntheon_scope.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    diagram_pipeline()
    diagram_lineage()
    diagram_modulation_gap()
    diagram_scope()
    print(f"wrote {len(list(OUT.glob('*.png')))} PNGs to {OUT}")
