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
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12, 5.2))

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
             fontsize=9, color="#b42318", weight="bold")

    axR.set_title("What an evolving Vital sound needs\n(static params + modulation)",
                  fontsize=11, color=INK, weight="bold")
    axR.set_xlim(0, 10); axR.set_ylim(0, 10)
    axR.set_axis_off()
    for i, (lab, v) in enumerate(zip(labels, vals)):
        x = 1 + i
        axR.plot([x, x], [1, 1 + v * 2.2], color=INK_SOFT, lw=6,
                 solid_capstyle="round", alpha=0.25)
        axR.text(x, 0.4, lab, ha="center", va="top", fontsize=7,
                 color="#b0b6bc")
    axR.text(5, 3.9, "static params (Syntheon's ceiling)",
             ha="center", fontsize=8, color=INK_SOFT, style="italic")

    t = np.linspace(0, 2 * np.pi, 200)
    xlfo = np.linspace(1, 9, 200)
    lfo = 6.8 + 1.0 * np.sin(2 * t)
    axR.plot(xlfo, lfo, color=EXTEND, lw=2.2)
    axR.text(1.1, 8.3, "LFO curve", color=EXTEND, fontsize=8.5, weight="bold")

    xe = np.linspace(1, 9, 200)
    seg = np.array([0.0, 0.6, 0.4, 0.4, 0.0])
    xe_seg = np.linspace(1, 9, len(seg))
    env = 5.0 + np.interp(xe, xe_seg, seg) * 1.4
    axR.plot(xe, env, color=ACCENT, lw=2.2)
    axR.text(1.1, 7.0, "envelope / automation", color=ACCENT,
             fontsize=8.5, weight="bold")

    axR.annotate("", xy=(8.6, 5.2), xytext=(8.6, 6.8),
                 arrowprops=dict(arrowstyle="<->", color=LINE, lw=1.2))
    axR.text(8.95, 6.0, "routing\n+ amount", fontsize=8, color=INK_SOFT,
             ha="left", va="center")

    fig.suptitle("The modulation gap", fontsize=13, weight="bold",
                 color=INK, y=1.0)
    fig.tight_layout()
    fig.savefig(OUT / "syntheon_modulation_gap.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


# --- Diagram: WTSv2 architecture --------------------------------------------
def diagram_architecture():
    fig, ax = plt.subplots(figsize=(15, 12))
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 12)
    ax.set_axis_off()

    ax.text(0.1, 11.6, "WTSv2 \u2014 Syntheon's Vital model (forward pass)",
            fontsize=14, weight="bold", color=INK)
    ax.text(0.1, 11.2,
            "traced from syntheon/inferencer/vital/models/*.py",
            fontsize=8.5, color=INK_SOFT, style="italic")

    # swim lanes
    lane(ax, 0.3, 1.7, 4.6, 10.4, "ENCODER + NOISE", INK_SOFT)
    lane(ax, 5.0, 1.7, 10.2, 10.4, "WAVETABLE PATH", ACCENT)
    lane(ax, 10.6, 3.0, 14.7, 10.4, "ADSR PATH", ADSR_C)

    # ---- inputs (top) ----
    inp_y = 9.7
    pitch = box(ax, 1.5, inp_y, 1.5, 0.55, "pitch", fc="#fff7e6", ec=INPUT_C,
                fontsize=8.5, weight="bold")
    loud = box(ax, 2.9, inp_y, 1.5, 0.55, "loudness", fc="#fff7e6", ec=INPUT_C,
               fontsize=8.5, weight="bold")
    mfcc = box(ax, 4.3, inp_y, 1.3, 0.55, "mfcc", fc="#fff7e6", ec=INPUT_C,
               fontsize=8.5, weight="bold")
    audio = box(ax, 7.6, inp_y, 2.0, 0.55, "raw audio  y", fc="#fff7e6",
                ec=INPUT_C, fontsize=8.5, weight="bold")

    # ---- encoder lane ----
    enc = box(ax, 2.9, 8.3, 3.6, 0.9,
              "encoder\nin_mlps \u2192 GRU \u2192 out_mlp",
              fc="#eef0f2", ec=INK_SOFT, fontsize=8.3)
    hidden = box(ax, 2.9, 6.9, 2.6, 0.6, "hidden representation",
                 fc="#e7ecf3", ec=INK_SOFT, fontsize=8.3, weight="bold")
    noise = box(ax, 2.9, 5.0, 3.8, 1.15,
                "noise path\nproj_matrices \u2192 scale_function\n"
                "\u2192 amp_to_IR \u2192 fft_convolve(rand noise)",
                fc="#f0f0f2", ec=NOISE_C, fontsize=7.8)
    noise_out = box(ax, 2.9, 3.6, 1.8, 0.5, "filtered noise",
                    fc="#f4f4f5", ec=NOISE_C, fontsize=8)

    arrow(ax, 1.5, inp_y - 0.28, 2.4, enc[1] + enc[3] / 2)
    arrow(ax, 2.9, inp_y - 0.28, 2.9, enc[1] + enc[3] / 2)
    arrow(ax, 4.3, inp_y - 0.28, 3.5, enc[1] + enc[3] / 2)
    arrow(ax, enc[1], enc[1] - enc[3] / 2 - 0.0, hidden[1] + hidden[3] / 2,
          color=INK_SOFT)
    arrow(ax, hidden[1], hidden[1] - hidden[3] / 2, noise[1] + noise[3] / 2,
          color=NOISE_C)
    arrow(ax, noise[1], noise[1] - noise[3] / 2, noise_out[1] + noise_out[3] / 2,
          color=NOISE_C)

    # ---- wavetable lane ----
    extract = box(ax, 7.6, 8.3, 4.4, 1.0,
                  "wavetable extraction\n"
                  "infer_wavetables(y, pitch)  [inference]\n"
                  "wt1_conv1d(y)               [training]",
                  fc="#eaf1f8", ec=ACCENT, fontsize=7.6)
    attn = box(ax, 7.6, 6.8, 3.6, 0.7,
               "attention_wt1 \u2192 softmax\n(mixing weights)",
               fc="#eaf1f8", ec=ACCENT, fontsize=7.8)
    synth = box(ax, 7.6, 5.5, 4.2, 0.95,
                "WavetableSynthV2\nattention-weighted wavetable osc\n"
                "+ linear interpolation + phase accum.",
                fc="#dce8f5", ec=ACCENT, fontsize=7.8, lw=1.4)
    harm = box(ax, 7.6, 4.1, 2.0, 0.55, "harmonic",
               fc="#dce8f5", ec=ACCENT, fontsize=8.5, weight="bold")

    arrow(ax, audio[1], audio[1] - audio[3] / 2,
          extract[1] + extract[3] / 2, color=ACCENT)
    arrow(ax, pitch[1], pitch[1] - pitch[3] / 2, 0, 0)  # placeholder, removed below
    # pitch -> extraction (curve)
    arrow(ax, pitch[1] + pitch[2] / 2, pitch[1], extract[1] - extract[2] / 2,
          extract[1] + 0.2, color=INPUT_C, lw=1.1,
          connectionstyle="arc3,rad=-0.35")
    ax.text(5.0, 9.15, "pitch", fontsize=7, color=INPUT_C, style="italic")

    arrow(ax, extract[1], extract[1] - extract[3] / 2,
          attn[1] + attn[3] / 2, color=ACCENT)
    arrow(ax, attn[1], attn[1] - attn[3] / 2,
          synth[1] + synth[3] / 2, color=ACCENT)
    # loudness -> synth amplitude (curve across)
    arrow(ax, loud[1] + loud[2] / 2, loud[1], synth[1] - synth[2] / 2,
          synth[1] + 0.15, color=INPUT_C, lw=1.1,
          connectionstyle="arc3,rad=-0.25")
    ax.text(5.6, 5.95, "loudness\n(amplitude)", fontsize=7, color=INPUT_C,
            style="italic", ha="center")
    # pitch -> synth oscillator
    arrow(ax, pitch[1] + pitch[2] / 2, pitch[1] - 0.1,
          synth[1] - synth[2] / 2 + 0.2, synth[1] - 0.1, color=INPUT_C, lw=1.0,
          connectionstyle="arc3,rad=-0.45")
    arrow(ax, synth[1], synth[1] - synth[3] / 2,
          harm[1] + harm[3] / 2, color=ACCENT)

    # ---- ADSR lane ----
    adsr_gru = box(ax, 12.6, 7.4, 3.4, 1.1,
                   "ADSR: 3\u00d7 bidirectional GRU\n"
                   "attack_gru / decay_gru / sustain_gru\n"
                   "\u2192 sigmoid heads",
                   fc="#e6f3f2", ec=ADSR_C, fontsize=7.7)
    adsr_env = box(ax, 12.6, 5.4, 2.8, 0.8,
                   "ADSR envelope\n(get_amp_shaper: power-fn shaping)",
                   fc="#d6eceb", ec=ADSR_C, fontsize=7.7)
    # loudness -> ADSR (curve across all lanes)
    arrow(ax, loud[1] + loud[2] / 2, loud[1], adsr_gru[1] - adsr_gru[2] / 2,
          adsr_gru[1] + 0.1, color=INPUT_C, lw=1.1,
          connectionstyle="arc3,rad=-0.2")
    ax.text(8.7, 8.7, "loudness", fontsize=7, color=INPUT_C, style="italic")
    arrow(ax, adsr_gru[1], adsr_gru[1] - adsr_gru[3] / 2,
          adsr_env[1] + adsr_env[3] / 2, color=ADSR_C)

    # ---- convergence ----
    summ = box(ax, 7.6, 2.7, 4.0, 0.7,
               "signal  =  harmonic  +  filtered noise",
               fc="#f3eefe", ec=EXTEND, fontsize=8.6, weight="bold")
    final = box(ax, 7.6, 1.3, 4.6, 0.7,
                "final signal  =  signal  \u00d7  ADSR",
                fc="#ece3fb", ec=EXTEND, fontsize=8.6, weight="bold", lw=1.5)

    arrow(ax, harm[1], harm[1] - harm[3] / 2, summ[1] + summ[2] / 2 - 0.3,
          summ[1] + summ[3] / 2, color=ACCENT)
    arrow(ax, noise_out[1], noise_out[1] - noise_out[3] / 2,
          summ[1] - summ[2] / 2 + 0.3, summ[1] + summ[3] / 2, color=NOISE_C,
          connectionstyle="arc3,rad=0.2")
    arrow(ax, adsr_env[1], adsr_env[1] - adsr_env[3] / 2,
          final[1] + final[2] / 2 - 0.4, final[1] + final[3] / 2,
          color=ADSR_C, connectionstyle="arc3,rad=0.25")
    arrow(ax, summ[1], summ[1] - summ[3] / 2,
          final[1] + final[3] / 2, color=EXTEND)

    # reverb note
    ax.text(0.4, 0.5,
            "note: the Reverb module exists in the code but is commented out\n"
            "in forward() \u2014 the shipped model does not apply reverb.",
            fontsize=7.6, color="#b42318", style="italic", va="bottom")

    fig.tight_layout()
    fig.savefig(OUT / "syntheon_architecture.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    diagram_modulation_gap()
    diagram_architecture()
    print(f"wrote {len(list(OUT.glob('*.png')))} PNG(s) to {OUT}")
