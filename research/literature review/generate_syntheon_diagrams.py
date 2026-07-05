#!/usr/bin/env python3
"""Generate diagrams for syntheon_notes.md.

Policy: only render an image where ASCII cannot carry the content (complex
branching or signal-domain shapes that must be drawn). Two diagrams qualify:

  1. modulation gap  - draws actual LFO / envelope waveforms (ASCII can't).
  2. WTSv2 architecture - multiple input streams fan out and converge through
     wavetable / noise / ADSR paths (the branching is the point).

Outputs PNGs to images/ next to this script, matching the house style of the
lfo_representation report images.

Architecture traced from the syntheon source (gudgud96/syntheon,
syntheon/inferencer/vital/models/), not reconstructed from memory.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).parent / "images"
OUT.mkdir(exist_ok=True)

# palette
INK = "#1f2328"
INK_SOFT = "#3c434a"
ACCENT = "#2b6cb0"      # wavetable / harmonic path
NOISE_C = "#6b7280"     # noise path
ADSR_C = "#0e7c7b"      # ADSR path (teal)
EXTEND = "#7c3aed"      # final / ours
INPUT_C = "#b8860b"     # inputs (amber)
LINE = "#9aa0a6"
LANE = "#f6f7f8"


def box(ax, x, y, w, h, text, *, fc="#ffffff", ec=LINE, fontsize=8.5,
        fontcolor=INK, weight="normal", rounding=0.08, lw=1.2, zorder=4):
    bb = FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle=f"round,pad=0.005,rounding_size={rounding}",
        fc=fc, ec=ec, lw=lw, zorder=zorder,
    )
    ax.add_patch(bb)
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize,
            color=fontcolor, weight=weight, zorder=zorder + 1)
    return (x, y, w, h)


def arrow(ax, x0, y0, x1, y1, *, color=LINE, lw=1.3, style="-|>",
          connectionstyle="arc3,rad=0", ls="-", zorder=2):
    a = FancyArrowPatch(
        (x0, y0), (x1, y1),
        arrowstyle=style, mutation_scale=12,
        color=color, lw=lw, connectionstyle=connectionstyle,
        linestyle=ls, zorder=zorder,
    )
    ax.add_patch(a)
    return a


def lane(ax, x0, y0, x1, y1, label, color):
    rect = FancyBboxPatch(
        (x0, y0), x1 - x0, y1 - y0,
        boxstyle="round,pad=0.01,rounding_size=0.15",
        fc=LANE, ec=color, lw=1.0, linestyle=(0, (4, 3)),
        alpha=0.7, zorder=1,
    )
    ax.add_patch(rect)
    ax.text(x0 + 0.2, y1 - 0.25, label, fontsize=8, color=color,
            weight="bold", zorder=2, va="top")


def bottom(n):
    """Bottom-center point of a node tuple (x, y, w, h)."""
    x, y, w, h = n
    return (x, y - h / 2)


def top(n):
    x, y, w, h = n
    return (x, y + h / 2)


def right(n):
    x, y, w, h = n
    return (x + w / 2, y)


def left(n):
    x, y, w, h = n
    return (x - w / 2, y)


def connect(ax, n_from, n_to, *, side_from="bottom", side_to="top",
            color=LINE, lw=1.3, rad=0.0, ls="-", shrink=0.0):
    """Draw an arrow between two node tuples using the named sides."""
    sidef = {"bottom": bottom, "top": top, "left": left, "right": right}[side_from]
    sidet = {"bottom": bottom, "top": top, "left": left, "right": right}[side_to]
    x0, y0 = sidef(n_from)
    x1, y1 = sidet(n_to)
    arrow(ax, x0, y0, x1, y1, color=color, lw=lw,
          connectionstyle=f"arc3,rad={rad}", ls=ls)


# --- Diagram: modulation gap -------------------------------------------------
def diagram_modulation_gap():
    """Left: static knob vector. Right: same knobs + modulation curves above."""
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5.6))

    labels = ["osc1", "osc2", "flt", "atk", "dec", "sus", "rel", "fx1", "fx2"]
    vals = [0.7, 0.4, 0.6, 0.3, 0.5, 0.8, 0.35, 0.55, 0.25]

    # ---------------- left panel: static knob vector -----------------
    axL.set_xlim(0, 10); axL.set_ylim(0, 10)
    axL.set_axis_off()
    axL.set_title("What Syntheon emits\na static knob vector",
                  fontsize=11.5, color=INK, weight="bold", pad=10)
    base = 1.5
    top = 9.0
    span = top - base
    for i, (lab, v) in enumerate(zip(labels, vals)):
        x = 1 + i
        axL.plot([x, x], [base, base + v * span], color=ACCENT, lw=9,
                 solid_capstyle="round", alpha=0.9)
        axL.plot([x - 0.38, x + 0.38], [base + v * span, base + v * span],
                 color=INK, lw=1.6)
        axL.text(x, base - 0.5, lab, ha="center", va="top",
                 fontsize=8.5, color=INK_SOFT)
    axL.plot([0.3, 9.7], [base, base], color=LINE, lw=1.2)
    axL.text(5, 9.65, "every value is a fixed scalar", ha="center",
             fontsize=9.5, color=INK_SOFT, style="italic")
    axL.text(5, 0.15, "no time axis \u00b7 no curves \u00b7 no routing",
             ha="center", fontsize=9.5, color="#b42318", weight="bold")

    # ---------------- right panel: static + modulation ---------------
    axR.set_xlim(0, 10); axR.set_ylim(0, 10)
    axR.set_axis_off()
    axR.set_title("What an evolving Vital sound needs\nstatic knobs + modulation",
                  fontsize=11.5, color=INK, weight="bold", pad=10)

    # band layout: knobs low, curves high, clearly separated
    knob_base = 1.5
    knob_top = 3.7
    knob_span = knob_top - knob_base
    curve_base = 4.6      # shared baseline for modulation curves
    curve_top = 9.0

    # divider between the two bands
    axR.plot([0.3, 9.7], [4.15, 4.15], color=LINE, lw=0.8, ls=(0, (4, 3)))
    axR.text(0.35, 3.95, "static params (Syntheon's ceiling)",
             fontsize=8, color=INK_SOFT, style="italic", va="top")
    axR.text(0.35, 9.25, "modulation \u2014 what Syntheon can't emit",
             fontsize=8, color=EXTEND, style="italic", weight="bold", va="top")

    # static knobs (dimmed, in lower band)
    for i, (lab, v) in enumerate(zip(labels, vals)):
        x = 1 + i
        axR.plot([x, x], [knob_base, knob_base + v * knob_span],
                 color=INK_SOFT, lw=7, solid_capstyle="round", alpha=0.3)
        axR.text(x, knob_base - 0.5, lab, ha="center", va="top",
                 fontsize=7.5, color="#9aa0a6")
    axR.plot([0.3, 9.7], [knob_base, knob_base], color=LINE, lw=1)

    # LFO curve (sine) in upper band
    t = np.linspace(0, 4 * np.pi, 400)
    xlfo = np.linspace(0.8, 9.2, 400)
    lfo_mid = (curve_base + curve_top) / 2
    lfo_amp = (curve_top - curve_base) / 2 * 0.55
    lfo = lfo_mid + lfo_amp * np.sin(2 * t)
    axR.plot(xlfo, lfo, color=EXTEND, lw=2.4)
    axR.text(0.9, curve_top - 0.1, "LFO", color=EXTEND, fontsize=9,
             weight="bold", va="top")

    # envelope / automation curve (ADSR-ish) in upper band, below the LFO
    xe = np.linspace(0.8, 9.2, 400)
    # normalized ADSR: quick attack, decay, sustain, release
    env_norm = np.zeros_like(xe)
    a, d, s_lvl, r = 0.05, 0.25, 0.6, 0.9
    for j, xv in enumerate((xe - 0.8) / 8.4):
        if xv < a:
            env_norm[j] = xv / a
        elif xv < a + d:
            env_norm[j] = 1.0 - (1.0 - s_lvl) * (xv - a) / d
        elif xv < r:
            env_norm[j] = s_lvl
        else:
            env_norm[j] = s_lvl * max(0.0, 1.0 - (xv - r) / (1.0 - r))
    env = curve_base + 0.05 + env_norm * (curve_top - curve_base - 0.1) * 0.45
    axR.plot(xe, env, color=ACCENT, lw=2.4)
    axR.text(9.1, curve_base + 0.15, "envelope", color=ACCENT, fontsize=9,
             weight="bold", va="bottom", ha="right")

    fig.suptitle("The modulation gap", fontsize=14, weight="bold",
                 color=INK, y=1.0)
    fig.tight_layout()
    fig.savefig(OUT / "syntheon_modulation_gap.png", dpi=180,
                bbox_inches="tight")
    plt.close(fig)


# --- Diagram: WTSv2 architecture (NN-style slabs) ---------------------------
# PlotNeuralNet / AlexNet visual language: extruded 3D layer slabs, thin
# connectors, muted palette, side labels. Keeps the real branching topology
# (wavetable / noise / ADSR lanes converging), just in the NN-diagram idiom.

# slab palette (muted, PlotNeuralNet-ish)
SLAB_INPUT = "#edc949"   # amber  - inputs
SLAB_ENC = "#76b7b2"     # teal   - encoder / shared
SLAB_WT = "#4e79a7"      # blue   - wavetable path
SLAB_NOISE = "#bab0ac"   # warm gray - noise path
SLAB_ADSR = "#59a14f"    # green  - ADSR path
SLAB_OUT = "#af7aa1"     # mauve  - outputs
SLAB_EDGE = "#3d3d3d"
SLAB_LINE = "#7a7a7a"


def slab(ax, x, y, w, h, depth=0.45, *, fc, label=None, sublabel=None,
         fontsize=8, ec=SLAB_EDGE, textcolor="white", lw=0.9, zorder=2):
    """Draw an extruded 'neural net layer' slab and return its front face rect.

    Front face is centered at (x, y). Depth extends up-right for the 3D effect.
    zorder layout: slab faces (2/3) < connectors (6) < slab labels (8), so
    cross-lane arrows stay visible above faces but text stays on top.
    """
    # top face (parallelogram toward upper-right)
    top = plt.Polygon([
        (x - w / 2, y + h / 2),
        (x - w / 2 + depth, y + h / 2 + depth),
        (x + w / 2 + depth, y + h / 2 + depth),
        (x + w / 2, y + h / 2),
    ], closed=True, fc=_shade(fc, 1.25), ec=ec, lw=lw, zorder=zorder)
    # right face
    right = plt.Polygon([
        (x + w / 2, y + h / 2),
        (x + w / 2 + depth, y + h / 2 + depth),
        (x + w / 2 + depth, y - h / 2 + depth),
        (x + w / 2, y - h / 2),
    ], closed=True, fc=_shade(fc, 0.75), ec=ec, lw=lw, zorder=zorder)
    # front face
    front = FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle="round,pad=0.002,rounding_size=0.06",
        fc=fc, ec=ec, lw=lw, zorder=zorder + 1,
    )
    for p in (top, right):
        ax.add_patch(p)
    ax.add_patch(front)
    txt = label or ""
    if sublabel:
        txt = f"{label}\n{sublabel}" if label else sublabel
    ax.text(x, y, txt, ha="center", va="center", fontsize=fontsize,
            color=textcolor, weight="bold", zorder=8)
    return (x, y, w, h)


def _shade(hexcolor, factor):
    """Lighten (>1) or darken (<1) a hex color."""
    h = hexcolor.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    r = int(min(max(r * factor, 0), 255))
    g = int(min(max(g * factor, 0), 255))
    b = int(min(max(b * factor, 0), 255))
    return f"#{r:02x}{g:02x}{b:02x}"


def slab_connect(ax, n_from, n_to, *, side_from="right", side_to="left",
                 color=SLAB_LINE, lw=1.1, rad=0.0, zorder=6):
    """Connector zorder sits above slab faces (zorder 5) but below labels (7+),
    so cross-lane arrows aren't buried behind intermediate slabs."""
    sf = {"bottom": bottom, "top": top, "left": left, "right": right}[side_from]
    st = {"bottom": bottom, "top": top, "left": left, "right": right}[side_to]
    x0, y0 = sf(n_from)
    x1, y1 = st(n_to)
    arrow(ax, x0, y0, x1, y1, color=color, lw=lw,
          connectionstyle=f"arc3,rad={rad}", zorder=zorder)


def diagram_architecture_nn():
    fig, ax = plt.subplots(figsize=(18, 10))
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 10)
    ax.set_axis_off()

    ax.text(0.1, 9.65, "WTSv2 \u2014 Syntheon's Vital model",
            fontsize=16, weight="bold", color=INK)
    ax.text(0.1, 9.2,
            "forward pass, traced from syntheon/inferencer/vital/models/*.py",
            fontsize=9.5, color=INK_SOFT, style="italic")

    # lane y-centers, well separated
    wt_y = 7.2       # wavetable lane (top)
    mid_y = 4.6      # encoder / shared lane (middle)
    adsr_y = 2.0     # ADSR lane (bottom)

    # ---- inputs (left column) ----
    xi = 1.5
    pitch = slab(ax, xi, wt_y, 1.4, 0.7, fc=SLAB_INPUT, label="pitch",
                 fontsize=8.5)
    loud = slab(ax, xi, mid_y, 1.4, 0.7, fc=SLAB_INPUT, label="loudness",
                fontsize=8.5)
    audio = slab(ax, xi, adsr_y + 0.9, 1.4, 0.7, fc=SLAB_INPUT,
                 label="audio y", fontsize=8.5)
    mfcc = slab(ax, xi, adsr_y - 0.9, 1.4, 0.7, fc=SLAB_INPUT,
                label="mfcc", fontsize=8.5)

    # ---- encoder (shared) ----
    x = 4.0
    enc = slab(ax, x, mid_y, 1.9, 1.8, fc=SLAB_ENC, label="encoder",
               sublabel="in_mlps\u2192GRU\n\u2192out_mlp", fontsize=8.5)
    slab_connect(ax, pitch, enc, side_from="right", side_to="left", rad=0.20)
    slab_connect(ax, loud, enc, color=SLAB_LINE)
    slab_connect(ax, mfcc, enc, side_from="right", side_to="left", rad=-0.20)

    # ---- wavetable lane (top) ----
    x = 6.8
    extract = slab(ax, x, wt_y, 2.3, 1.1, fc=SLAB_WT, label="wavetable",
                   sublabel="extraction", fontsize=8.5)
    slab_connect(ax, audio, extract, side_from="right", side_to="left",
                 rad=0.22)
    slab_connect(ax, pitch, extract, side_from="right", side_to="left",
                 rad=0.08, color=SLAB_LINE, lw=0.9)

    x = 9.6
    attn = slab(ax, x, wt_y, 1.9, 1.1, fc=SLAB_WT, label="attention",
                sublabel="softmax", fontsize=8.5)
    slab_connect(ax, extract, attn)

    x = 12.3
    wtsynth = slab(ax, x, wt_y, 2.2, 1.4, fc=SLAB_WT, label="wavetable",
                   sublabel="osc V2", fontsize=8.5)
    slab_connect(ax, attn, wtsynth)

    x = 15.2
    harm = slab(ax, x, wt_y, 1.6, 1.1, fc=SLAB_WT, label="harmonic",
                fontsize=8.5)
    slab_connect(ax, wtsynth, harm)

    # pitch & loudness also feed the oscillator/amplitude; annotate compactly
    # rather than drawing two long crossing arrows
    ax.annotate("pitch + loudness\nfeed osc / amplitude",
                xy=(wtsynth[0] - wtsynth[2] / 2, wtsynth[1] + 0.1),
                xytext=(6.6, wt_y + 1.55),
                fontsize=7.5, color=SLAB_LINE, style="italic",
                ha="left", va="center",
                arrowprops=dict(arrowstyle="-", color=SLAB_LINE,
                                lw=0.8, ls=(0, (3, 2))))

    # ---- noise lane (middle, off encoder) ----
    x = 6.8
    noise_proj = slab(ax, x, mid_y, 2.3, 1.15, fc=SLAB_NOISE,
                      label="noise proj", sublabel="scale_fn\u2192IR",
                      fontsize=8.5, textcolor=INK)
    slab_connect(ax, enc, noise_proj, color=SLAB_LINE)
    x = 9.6
    noise_filt = slab(ax, x, mid_y, 1.9, 1.15, fc=SLAB_NOISE,
                      label="filtered", sublabel="fft_conv",
                      fontsize=8.5, textcolor=INK)
    slab_connect(ax, noise_proj, noise_filt, color=SLAB_LINE)

    # ---- ADSR lane (bottom) ----
    x = 6.8
    adsr_gru = slab(ax, x, adsr_y, 2.3, 1.15, fc=SLAB_ADSR,
                    label="ADSR GRUs", sublabel="3\u00d7bidir\u2192sig",
                    fontsize=8.5)
    slab_connect(ax, loud, adsr_gru, side_from="right", side_to="left",
                 rad=-0.22, color=SLAB_LINE)
    x = 9.6
    adsr_env = slab(ax, x, adsr_y, 1.9, 1.15, fc=SLAB_ADSR,
                    label="ADSR env", sublabel="power-fn",
                    fontsize=8.5)
    slab_connect(ax, adsr_gru, adsr_env, color=SLAB_LINE)

    # ---- convergence (right) ----
    x = 15.2
    summ = slab(ax, x, mid_y, 2.0, 1.15, fc=SLAB_OUT, label="sum",
                sublabel="harm+noise", fontsize=8.5)
    slab_connect(ax, harm, summ, side_from="bottom", side_to="top", rad=0.18)
    slab_connect(ax, noise_filt, summ, side_from="right", side_to="left",
                 rad=0.22, color=SLAB_LINE)

    final = slab(ax, x, adsr_y, 2.0, 1.15, fc=SLAB_OUT, label="output",
                 sublabel="sig\u00d7ADSR", fontsize=8.5)
    slab_connect(ax, summ, final, side_from="bottom", side_to="top", rad=0.18)
    slab_connect(ax, adsr_env, final, side_from="right", side_to="left",
                 color=SLAB_LINE)

    # legend across the bottom (no title collision)
    legend_items = [
        ("input", SLAB_INPUT), ("encoder / shared", SLAB_ENC),
        ("wavetable path", SLAB_WT), ("noise path", SLAB_NOISE),
        ("ADSR path", SLAB_ADSR), ("output", SLAB_OUT),
    ]
    lx, ly = 0.4, 0.35
    for i, (name, col) in enumerate(legend_items):
        rx = lx + i * 2.7
        ax.add_patch(plt.Rectangle((rx, ly), 0.28, 0.20,
                                   fc=col, ec=SLAB_EDGE, lw=0.7))
        ax.text(rx + 0.36, ly + 0.10, name, fontsize=8, color=INK,
                va="center")

    # reverb note, bottom right
    ax.text(17.9, 0.45,
            "Reverb module exists in code\nbut is commented out in forward()\n"
            "\u2014 shipped model applies no reverb",
            fontsize=7.8, color="#b42318", style="italic",
            va="center", ha="right")

    fig.tight_layout()
    fig.savefig(OUT / "syntheon_architecture_nn.png", dpi=180,
                bbox_inches="tight")
    plt.close(fig)


# --- Diagram: WTSv2 architecture (swim-lane flowchart) ----------------------
def diagram_architecture():
    """Horizontal left-to-right flow. Three lanes (wavetable top, encoder/noise
    middle, ADSR bottom) converge at the right. Same topology as the NN-style
    diagram, plain-box idiom."""
    fig, ax = plt.subplots(figsize=(18, 9))
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 9)
    ax.set_axis_off()

    ax.text(0.1, 8.6, "WTSv2 \u2014 Syntheon's Vital model (forward pass)",
            fontsize=15, weight="bold", color=INK)
    ax.text(0.1, 8.2,
            "traced from syntheon/inferencer/vital/models/*.py",
            fontsize=9, color=INK_SOFT, style="italic")

    # lane bands (subtle)
    lane(ax, 4.9, 5.9, 16.3, 8.0, "wavetable path", ACCENT)
    lane(ax, 4.9, 3.4, 16.3, 5.5, "encoder + noise", INK_SOFT)
    lane(ax, 4.9, 0.9, 16.3, 3.0, "ADSR path", ADSR_C)

    wt_y = 6.95
    mid_y = 4.45
    adsr_y = 1.95

    # ---- inputs (left column, stacked to face the lane they feed) ----
    xi = 1.5
    pitch = box(ax, xi, 7.2, 1.5, 0.7, "pitch", fc="#fff7e6", ec=INPUT_C,
                fontsize=9, weight="bold")
    audio = box(ax, xi, 5.8, 1.5, 0.7, "audio y", fc="#fff7e6", ec=INPUT_C,
                fontsize=9, weight="bold")
    loud = box(ax, xi, 4.4, 1.5, 0.7, "loudness", fc="#fff7e6", ec=INPUT_C,
               fontsize=9, weight="bold")
    mfcc = box(ax, xi, 3.0, 1.5, 0.7, "mfcc", fc="#fff7e6", ec=INPUT_C,
               fontsize=9, weight="bold")

    # ---- encoder (shared) ----
    x = 4.0
    enc = box(ax, x, mid_y, 1.9, 1.7,
              "encoder\nin_mlps\u2192GRU\n\u2192out_mlp",
              fc="#eef0f2", ec=INK_SOFT, fontsize=8.5)
    connect(ax, pitch, enc, side_from="right", side_to="left", rad=0.18)
    connect(ax, loud, enc, color=INK_SOFT)
    connect(ax, mfcc, enc, side_from="right", side_to="left", rad=-0.18)

    # ---- wavetable lane (top, left to right) ----
    x = 6.8
    extract = box(ax, x, wt_y, 2.3, 1.1,
                  "wavetable extraction\ninfer_wavetables(y, pitch)\n"
                  "[or wt1_conv1d(y) in training]",
                  fc="#eaf1f8", ec=ACCENT, fontsize=7.8)
    connect(ax, audio, extract, side_from="right", side_to="left", rad=0.22)
    connect(ax, pitch, extract, side_from="right", side_to="left",
            rad=0.08, color=INPUT_C, lw=1.0)

    x = 9.6
    attn = box(ax, x, wt_y, 1.9, 1.1,
               "attention\nsoftmax mix",
               fc="#eaf1f8", ec=ACCENT, fontsize=8.3)
    connect(ax, extract, attn, color=ACCENT)

    x = 12.3
    synth = box(ax, x, wt_y, 2.2, 1.4,
                "WavetableSynthV2\nattention-weighted osc\n"
                "+ linear interp\n+ phase accum.",
                fc="#dce8f5", ec=ACCENT, fontsize=7.8, lw=1.4)
    connect(ax, attn, synth, color=ACCENT)

    x = 15.2
    harm = box(ax, x, wt_y, 1.7, 1.1, "harmonic",
               fc="#dce8f5", ec=ACCENT, fontsize=9, weight="bold")
    connect(ax, synth, harm, color=ACCENT)

    # pitch & loudness also feed the oscillator/amplitude; annotate compactly
    ax.annotate("pitch + loudness\nfeed osc / amplitude",
                xy=(synth[0] - synth[2] / 2, synth[1] + 0.15),
                xytext=(6.6, wt_y + 1.35),
                fontsize=7.8, color=INPUT_C, style="italic",
                ha="left", va="center",
                arrowprops=dict(arrowstyle="-", color=INPUT_C,
                                lw=0.8, ls=(0, (3, 2))))

    # ---- noise lane (middle, off encoder) ----
    x = 6.8
    noise = box(ax, x, mid_y, 2.3, 1.15,
                "noise path\nproj_matrices\u2192scale_fn\n"
                "\u2192 amp_to_IR",
                fc="#f0f0f2", ec=NOISE_C, fontsize=7.8)
    connect(ax, enc, noise, color=NOISE_C)
    x = 9.6
    noise_out = box(ax, x, mid_y, 1.9, 1.15,
                    "filtered noise\nfft_convolve(rand)",
                    fc="#f4f4f5", ec=NOISE_C, fontsize=7.8)
    connect(ax, noise, noise_out, color=NOISE_C)

    # ---- ADSR lane (bottom) ----
    x = 6.8
    adsr_gru = box(ax, x, adsr_y, 2.3, 1.15,
                   "ADSR: 3\u00d7 bidir GRU\nattack/decay/sustain\n"
                   "\u2192 sigmoid heads",
                   fc="#e6f3f2", ec=ADSR_C, fontsize=7.7)
    connect(ax, loud, adsr_gru, side_from="right", side_to="left",
            rad=-0.22, color=INPUT_C, lw=1.0)
    x = 9.6
    adsr_env = box(ax, x, adsr_y, 1.9, 1.15,
                   "ADSR envelope\npower-fn shaping",
                   fc="#d6eceb", ec=ADSR_C, fontsize=7.8)
    connect(ax, adsr_gru, adsr_env, color=ADSR_C)

    # ---- convergence (right) ----
    x = 15.2
    summ = box(ax, x, mid_y, 2.1, 1.0,
               "signal =\nharmonic + noise",
               fc="#f3eefe", ec=EXTEND, fontsize=8.5, weight="bold")
    connect(ax, harm, summ, side_from="bottom", side_to="top", rad=0.18,
            color=ACCENT)
    connect(ax, noise_out, summ, side_from="right", side_to="left",
            rad=0.22, color=NOISE_C)

    final = box(ax, x, adsr_y, 2.1, 1.0,
                "final signal =\nsignal \u00d7 ADSR",
                fc="#ece3fb", ec=EXTEND, fontsize=8.5, weight="bold", lw=1.5)
    connect(ax, summ, final, side_from="bottom", side_to="top",
            rad=0.18, color=EXTEND)
    connect(ax, adsr_env, final, side_from="right", side_to="left",
            color=ADSR_C)

    # reverb note
    ax.text(0.2, 0.3,
            "note: the Reverb module exists in the code but is commented out\n"
            "in forward() \u2014 the shipped model does not apply reverb.",
            fontsize=8, color="#b42318", style="italic", va="bottom")

    fig.tight_layout()
    fig.savefig(OUT / "syntheon_architecture.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    diagram_modulation_gap()
    diagram_architecture()
    diagram_architecture_nn()
    print(f"wrote {len(list(OUT.glob('*.png')))} PNG(s) to {OUT}")
