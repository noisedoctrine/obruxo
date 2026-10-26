"""Constant-gain listening loudness matching, separate from reconstruction scores."""

import hashlib
import math
from pathlib import Path
import re
import shutil
import subprocess

import numpy as np

POLICY = {"purpose": "timbre", "loudness": "integrated_bs1770_constant_gain", "target_lufs": -23., "maximum_spread_lu": .1,
          "true_peak_ceiling_dbtp": -1., "absolute_level": "assessed_separately", "envelopes": "preserved_without_dynamic_processing"}


def resolve_ffmpeg(path=None):
    if path is None:
        path = shutil.which("ffmpeg")
    if path is None:
        try:
            from imageio_ffmpeg import get_ffmpeg_exe
            path = get_ffmpeg_exe()
        except ImportError as error:
            raise ValueError("tone listening requires FFmpeg for loudness matching; supply --ffmpeg or install the listening extra") from error
    path = Path(path).resolve()
    if not path.is_file():
        raise ValueError("loudness-meter executable is missing")
    return path


class LoudnessMeter:
    def __init__(self, path=None):
        self.path = resolve_ffmpeg(path)
        self.flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        version = subprocess.run([str(self.path), "-version"], capture_output=True, text=True, check=True, timeout=20, creationflags=self.flags)
        self.identity = {"backend": "ffmpeg_ebur128_true_peak", "version": version.stdout.splitlines()[0],
                         "executable_sha256": hashlib.sha256(self.path.read_bytes()).hexdigest()}

    def measure(self, audio, rate=44100):
        if (rate != 44100 or audio.dtype != np.float32 or audio.ndim != 2 or audio.shape[1] != 2
                or len(audio) < .4 * rate or not np.isfinite(audio).all()):
            raise ValueError("loudness matching needs finite 44.1-kHz stereo float32 audio of at least 400 ms")
        command = [str(self.path), "-hide_banner", "-nostats", "-f", "f32le", "-ar", str(rate), "-ac", "2", "-i", "pipe:0",
                   "-af", "ebur128=peak=true:metadata=1,ametadata=mode=print", "-f", "null", "-"]
        result = subprocess.run(command, input=np.ascontiguousarray(audio, dtype="<f4").tobytes(), capture_output=True, timeout=30, creationflags=self.flags)
        if result.returncode:
            raise ValueError("loudness meter failed: " + result.stderr.decode(errors="replace")[-800:])
        log = result.stderr.decode(errors="replace")
        integrated = re.findall(r"lavfi\.r128\.I=(-?\d+(?:\.\d+)?)", log)
        peaks = re.findall(r"True peak:\s+Peak:\s+(-?\d+(?:\.\d+)?)\s+dBFS", log)
        if not integrated or not peaks:
            raise ValueError("loudness meter did not return integrated loudness and true peak")
        loudness, peak = float(integrated[-1]), float(peaks[-1])
        if not math.isfinite(loudness) or loudness <= -70 or not math.isfinite(peak):
            raise ValueError("clip has no measurable gated loudness")
        # FFmpeg prints the final true peak to 0.1 dB; use its upper rounding bound.
        rms = float(np.sqrt(np.mean(audio.astype(np.float64)**2)))
        sample_peak = float(np.max(np.abs(audio)))
        return {"integrated_lufs": loudness, "true_peak_dbtp": peak, "true_peak_upper_dbtp": peak + .05,
                "rms_dbfs": 20 * math.log10(rms), "sample_peak_dbfs": 20 * math.log10(sample_peak)}


def match_triplet(audio, meter):
    if set(audio) != {"reference", "candidate", "control"} or len({value.shape for value in audio.values()}) != 1:
        raise ValueError("loudness matching needs three aligned complete clips")
    before = {name: meter.measure(value) for name, value in audio.items()}
    gains = {name: POLICY["target_lufs"] - value["integrated_lufs"] for name, value in before.items()}
    highest = max(before[name]["true_peak_upper_dbtp"] + gain for name, gain in gains.items())
    attenuation = min(0., POLICY["true_peak_ceiling_dbtp"] - highest - .05)
    matched = {name: (value.astype(np.float64) * 10**((gains[name] + attenuation) / 20)).astype(np.float32) for name, value in audio.items()}
    after = {name: meter.measure(value) for name, value in matched.items()}
    levels = [value["integrated_lufs"] for value in after.values()]
    if max(levels) - min(levels) > POLICY["maximum_spread_lu"] or any(value["true_peak_upper_dbtp"] > POLICY["true_peak_ceiling_dbtp"] for value in after.values()):
        raise ValueError("exported listening clips miss the frozen loudness/headroom tolerance")
    return matched, {"before": before, "after": after, "individual_gain_db": gains, "common_gain_db": attenuation,
                     "maximum_spread_lu": max(levels) - min(levels), "policy": POLICY}
