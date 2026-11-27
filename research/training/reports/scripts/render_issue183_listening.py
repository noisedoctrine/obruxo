"""Export deterministic listening previews from a completed issue 183 run."""

# ruff: noqa: E402
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from time import sleep

import numpy as np
import torch
from scipy.io import wavfile

TRAINING = Path(__file__).resolve().parents[2]
RESEARCH = TRAINING.parent
WORKTREE = TRAINING.parents[1]
MAIN_ROOT = WORKTREE.parent.parent if WORKTREE.parent.name == "tmp" else WORKTREE
sys.path[:0] = [str(RESEARCH / "data_generation"), str(TRAINING)]

import run_issue183_flow_matching as pilot

from obruxo_data.midi import Performance
from obruxo_training.codec import PilotCodec, PilotTarget
from obruxo_training.flow_matching import (DirectSpectrumModel, FlowSpectrumModel, as_spectrogram)
from obruxo_training.proxy import performance_inputs
from obruxo_training.spectral import SAMPLE_RATE, SpectralTargets
from obruxo_training.surrogate_codec import SurrogateCodec
from obruxo_training.surrogate_pilot import control_tensor
from obruxo_training.temporal_proxy import full_times

BACKBONES = ("residual_mlp", "transformer", "conformer")
FFT_SIZE, HOP = 2048, 512
FREQUENCY_BINS = FFT_SIZE // 2 + 1


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    for attempt in range(20):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if attempt == 19:
                raise
            sleep(.05)


def update_status(output, phase, completed, total, *, state="running", message=None):
    status = {"phase": phase, "arm": "", "completed": completed, "total": total,
              "arm_index": 6, "arm_count": 6, "failures": 0, "state": state,
              "updated_at": __import__("datetime").datetime.now(
                  __import__("datetime").timezone.utc).isoformat(timespec="seconds"),
              "message": message}
    write_json(output / "status.json", status)


def griffin_lim(magnitude, length, seed, iterations=48):
    magnitude = torch.as_tensor(magnitude, dtype=torch.float32) * math.sqrt(FFT_SIZE)
    window = torch.hann_window(FFT_SIZE)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    spectrum = magnitude * torch.exp(2j * torch.pi * torch.rand(magnitude.shape, generator=generator))
    audio = torch.istft(spectrum, FFT_SIZE, hop_length=HOP, window=window, center=True, length=length)
    for _ in range(iterations):
        estimate = torch.stft(audio, FFT_SIZE, hop_length=HOP, window=window, center=True,
                              pad_mode="constant", return_complex=True)
        spectrum = magnitude * estimate / estimate.abs().clamp_min(1e-8)
        audio = torch.istft(spectrum, FFT_SIZE, hop_length=HOP, window=window, center=True, length=length)
    return audio.numpy()


def listening_html(payload):
    rows = []
    for case in payload["cases"]:
        cells = [f'<td><b>Target</b><audio controls preload="none" src="{case["audio"]["target"]}"></audio></td>']
        for name in payload["models"]:
            cells.append(f'<td><b>{name}</b> · <a href="{case["exact_outputs"][name]}">exact predicted spectrum</a>'
                         f'<audio controls preload="none" src="{case["audio"][name]}"></audio><label>Grade '
                         f'<select data-rating="{case["example"]}-{name}"><option value="">—</option>'
                         + "".join(f'<option value="{score}">{score}</option>' for score in range(1, 6))
                         + "</select></label></td>")
        cells.append(f'<td><textarea data-comment="{case["example"]}-comment" placeholder="Overall feedback for this example"></textarea></td>')
        rows.append(f'<tr><th>Example {case["example"]}<br><small>{case["case_id"]}</small></th>{"".join(cells)}</tr>')
    headers = "".join(f"<th>{name}</th>" for name in payload["models"])
    return """<!doctype html><meta charset="utf-8"><title>Issue 183 listening table</title>
<style>body{font:15px Segoe UI,Arial;margin:24px;color:#222}table{border-collapse:collapse;width:100%;min-width:1800px}th,td{border:1px solid #bbb;padding:9px;vertical-align:top}thead{position:sticky;top:0;background:#f3f5f7}audio{width:205px;display:block;margin:7px 0}textarea{min-width:190px;min-height:60px}select{min-width:72px}button{font:inherit;padding:7px}</style>
<h1>Issue 183 · Direct prediction vs flow matching</h1>
<p>Ten exact #173 held-out examples. All seven clips in each row are peak-normalized to −1 dBFS for listening only. The model predicts magnitude spectra; the previews use the same fixed-seed, deterministic 48-iteration Griffin–Lim transform. Download the exact predicted spectra in NPZ form.</p>
<p>Rate each model 1 (worst) to 5 (best) and add one overall comment per example. Ratings save in this browser. <button id="export">Export ratings</button> <span id="status"></span></p>
<div style="overflow:auto"><table><thead><tr><th>Example</th><th>Target</th>""" + headers + """<th>Overall feedback</th></tr></thead><tbody>""" + "".join(rows) + """</tbody></table></div>
<script>const key="issue183-flow-listening-v1";let saved={};try{saved=JSON.parse(localStorage.getItem(key)||"{}")}catch{}function persist(){const value={};document.querySelectorAll("[data-rating],[data-comment]").forEach(e=>{if(e.value)value[e.dataset.rating||e.dataset.comment]=e.value});saved=value;localStorage.setItem(key,JSON.stringify(value));document.getElementById("status").textContent=" Saved"}document.querySelectorAll("[data-rating],[data-comment]").forEach(e=>{e.value=saved[e.dataset.rating||e.dataset.comment]||"";e.addEventListener("change",persist);e.addEventListener("input",persist)});document.getElementById("export").onclick=()=>{const value={format:"issue183_listening_ratings_v1",ratings_and_comments:saved};const a=document.createElement("a");a.href=URL.createObjectURL(new Blob([JSON.stringify(value,null,2)],{type:"application/json"}));a.download="issue183-listening-ratings.json";a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000)};</script>"""


@torch.no_grad()
def render(args):
    run_output, validation_source = args.run_output, args.validation_source
    verified = pilot.verify_completed_run(run_output, validation_source)
    device = torch.device(args.device)
    torch.set_num_threads(args.threads)
    if device.type == "xpu" and not torch.xpu.is_available():
        raise RuntimeError("configured XPU is unavailable")
    codec = SurrogateCodec(PilotCodec(), expanded=True)
    models = {}
    for backbone in BACKBONES:
        for mode, model_type in (("direct", DirectSpectrumModel), ("flow", FlowSpectrumModel)):
            arm = f"{backbone}-{mode}"
            saved = torch.load(run_output / f"{arm}.pt", map_location="cpu", weights_only=True)
            model = model_type(backbone, codec.width).to(device)
            model.load_state_dict(saved["model"])
            model.eval()
            models[arm] = model

    manifest_path = validation_source.parent / "issue173-listening-test-20261127" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validation_frozen = json.loads((validation_source / "plan.json").read_text(encoding="utf-8"))
    source_plan = validation_frozen.get("plan", validation_frozen)
    planned = {row["case_id"]: row for row in source_plan["evaluation"]["cases"]}
    listening = run_output / "listening"
    audio_dir = listening / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    names = [f"{backbone}-direct" for backbone in BACKBONES] + [f"{backbone}-flow-16" for backbone in BACKBONES]
    cases, assets = [], []
    if len(manifest["cases"]) != 10:
        raise ValueError("the frozen #173 listening packet must contain ten examples")
    update_status(run_output, "Writing ten-case listening previews", 0, 10)
    for example, entry in enumerate(manifest["cases"], 1):
        case = planned[entry["case_id"]]
        rate, source_audio = wavfile.read(manifest_path.parent / "audio" / f"sample-{example}-target.wav")
        if rate != SAMPLE_RATE or source_audio.dtype != np.float32 or source_audio.ndim != 2 or source_audio.shape[1] != 2:
            raise ValueError(f"unexpected target audio for #173 listening example {example}")
        measured = SpectralTargets()(torch.from_numpy(source_audio.T.copy())[None])
        times = tuple(value.to(device) for value in full_times(measured))
        controls = control_tensor(codec, [PilotTarget.from_dict(case["target"])], device)
        performance = performance_inputs(Performance.from_dict(case["performance"]))[None].to(device)
        lengths = measured["lengths"].to(device)
        frame_mask = measured["masks"][1].to(device)
        length = int(measured["lengths"][0])
        predictions = {}
        for model_index, backbone in enumerate(BACKBONES):
            direct = models[f"{backbone}-direct"]
            predictions[f"{backbone}-direct"] = as_spectrogram(
                direct(controls, performance, times[1], lengths))[0].float().cpu().numpy()
            flow = models[f"{backbone}-flow"]
            generator = torch.Generator(device="cpu").manual_seed(183000 + example)
            noise = torch.randn((1, times[1].shape[1], 2 * FREQUENCY_BINS), generator=generator).to(device)
            predictions[f"{backbone}-flow-16"] = as_spectrogram(
                flow.sample(controls, performance, times[1], lengths, frame_mask, noise, 16))[0].float().cpu().numpy()

        audio_files, exact_files = {}, {}
        target = source_audio.astype(np.float32, copy=True)
        target *= np.float32(10 ** (-1 / 20) / max(1e-8, float(np.abs(target).max())))
        target_path = audio_dir / f"example-{example:02d}-target.wav"
        wavfile.write(target_path, SAMPLE_RATE, target)
        audio_files["target"] = f"audio/{target_path.name}"
        for model_index, name in enumerate(names):
            magnitude = predictions[name]
            npz_path = audio_dir / f"example-{example:02d}-{name}.npz"
            np.savez_compressed(npz_path, magnitude_2048=magnitude, sample_rate=SAMPLE_RATE,
                                fft_size=FFT_SIZE, hop=HOP, patch_root=entry["root"], case_id=entry["case_id"])
            exact_files[name] = f"audio/{npz_path.name}"
            reconstructed = [griffin_lim(magnitude[channel], length,
                                         173000 + 100 * example + model_index + channel)
                             for channel in range(2)]
            preview = np.stack(reconstructed, axis=1).astype(np.float32)
            preview *= np.float32(10 ** (-1 / 20) / max(1e-8, float(np.abs(preview).max())))
            wav_path = audio_dir / f"example-{example:02d}-{name}.wav"
            wavfile.write(wav_path, SAMPLE_RATE, preview)
            audio_files[name] = f"audio/{wav_path.name}"
        cases.append({"example": example, "case_id": entry["case_id"], "patch_root": entry["root"],
                      "audio": audio_files, "exact_outputs": exact_files})
        for path in [target_path, *(audio_dir / Path(value).name for key, value in audio_files.items() if key != "target"),
                     *(audio_dir / Path(value).name for value in exact_files.values())]:
            assets.append({"path": path.name, "sha256": sha256_file(path), "bytes": path.stat().st_size})
        update_status(run_output, "Writing ten-case listening previews", example, 10)
    payload = {"title": "Issue 183 · Flow matching previews", "models": names, "cases": cases}
    write_json(listening / "packet.json", payload)
    page = listening / "index.html"
    page.write_text(listening_html(payload), encoding="utf-8")
    metadata = {"examples": len(cases), "available": True,
                "path": pilot.repo_reference(page, "issue183_listening_previews"),
                "source_case_ids": [case["case_id"] for case in cases],
                "griffin_lim_iterations": 48, "normalization": "each stereo clip peak-normalized to -1 dBFS for audition only",
                "exporter_script_sha256": sha256_file(__file__), "assets": assets}
    write_json(listening / "asset-receipts.json", {"format": "issue183_listening_assets_v1", "assets": assets})
    summary_path = run_output / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["listening"] = metadata
    write_json(summary_path, summary)
    update_status(run_output, "Complete", 500, 500, state="complete",
                  message="Training, evaluation, report, curves and ten-case listening previews complete")
    with (run_output / "events.jsonl").open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(json.loads((run_output / "status.json").read_text()), sort_keys=True) + "\n")
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-output", type=Path, required=True)
    parser.add_argument("--validation-source", type=Path, default=MAIN_ROOT / "research/training/outputs/issue173-model-selection-active-20261126-v4")
    parser.add_argument("--device", choices=("xpu", "cpu"), default="xpu")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    print(json.dumps(render(args), indent=2), flush=True)


if __name__ == "__main__":
    main()
