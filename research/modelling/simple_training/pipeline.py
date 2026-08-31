"""Small audio -> dense controls -> Vital experiment; no downloaded preset bank."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly
import torch
from torch import nn
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "data_generation"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "basic_pitch"))

from obruxo_data.midi import Performance  # noqa: E402
from obruxo_data.render import RenderRequest, VitalRenderer  # noqa: E402
from obruxo_data.vital import VitalPreset  # noqa: E402
from obruxo_basic_pitch import BasicPitchICASSP2022  # noqa: E402
from obruxo_basic_pitch.constants import AUDIO_N_SAMPLES  # noqa: E402
from preset_contract import PresetContract  # noqa: E402


SAMPLE_RATE = 22_050
SAMPLES = 13_230  # A fixed C4 for 0.5 seconds, then 0.1 seconds of release.
OCTAVES = (-12, 0, 12)
LEVEL_MIN, LEVEL_MAX = 0.2, 0.8
BP_CHECKPOINT = Path(__file__).resolve().parents[1] / "basic_pitch/artifacts/basic_pitch_icassp_2022.pt"
VALID_FRAMES = (SAMPLES + 255) // 256
TRANSFORMER = {"layers": 1, "width": 32, "heads": 2, "feedforward": 64, "tokens": VALID_FRAMES + 1, "dropout": 0.0}
CONTRACT = {"version": 4, "model": "basic_pitch_tiny_transformer", "transformer": TRANSFORMER, "preset_version": "1.5.5",
            "sample_rate": SAMPLE_RATE, "samples": SAMPLES,
            "categorical": {"osc_1_transpose": list(OCTAVES)},
            "continuous": {"osc_1_level": [LEVEL_MIN, LEVEL_MAX]},
            "dense_order": ["octave_logit_0", "octave_logit_1", "octave_logit_2", "level_logit"]}


class PresetCodec:
    """Argmax (first index wins ties), sigmoid, and a fixed canonical template."""

    def __init__(self):
        self.output_contract = PresetContract()
        self.template = self.output_contract.template

    def preset(self, category: int, level: float) -> VitalPreset:
        if type(category) is not int or category not in range(len(OCTAVES)) or type(level) not in (int, float) or not np.isfinite(level) or not 0 <= level <= 1:
            raise ValueError("expected an octave category and a finite normalized level in [0, 1]")
        preset = VitalPreset(self.template.to_dict(), self.template.schema)
        preset.set_raw("osc_1_transpose", OCTAVES[category])
        preset.set_raw("osc_1_level", LEVEL_MIN + (LEVEL_MAX - LEVEL_MIN) * level)
        self.output_contract.validate(preset.to_dict()).require_valid()
        return preset

    def decode(self, dense: torch.Tensor) -> VitalPreset:
        dense = torch.as_tensor(dense).detach().cpu()
        if dense.shape != (4,) or dense.dtype == torch.bool or dense.is_complex() or not torch.isfinite(dense).all():
            raise ValueError("dense prediction must contain four finite logits")
        return self.preset(int(dense[:3].argmax()), float(dense[3].sigmoid()))

    def save(self, preset: VitalPreset, path: Path | str) -> None:
        self.output_contract.save(preset, path)


class PerceptualSpectrum(nn.Module):
    """Two-resolution log-mel magnitude features, retaining 16 time bins each.

    L1 distance in this representation is the perceptual spectral objective.
    This intentionally small loss is not a calibrated listening-quality score.
    """

    fft_sizes = (512, 2048)
    mel_bins, time_bins = 32, 16
    size = len(fft_sizes) * mel_bins * time_bins

    def __init__(self):
        super().__init__()
        mel_max = 2595 * np.log10(1 + (SAMPLE_RATE / 2) / 700)
        edges = 700 * (10 ** (torch.linspace(0, mel_max, self.mel_bins + 2) / 2595) - 1)
        for n_fft in self.fft_sizes:
            frequencies = torch.linspace(0, SAMPLE_RATE / 2, n_fft // 2 + 1)
            rising = (frequencies[None] - edges[:-2, None]) / (edges[1:-1] - edges[:-2])[:, None]
            falling = (edges[2:, None] - frequencies[None]) / (edges[2:] - edges[1:-1])[:, None]
            bank = torch.minimum(rising, falling).clamp_min(0)
            bank = bank / bank.sum(dim=1, keepdim=True).clamp_min(1e-8)
            self.register_buffer(f"mel_{n_fft}", bank)
            self.register_buffer(f"window_{n_fft}", torch.hann_window(n_fft))

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        if audio.ndim != 2 or audio.shape[1] != SAMPLES or not torch.isfinite(audio).all():
            raise ValueError(f"expected finite mono audio [batch, {SAMPLES}]")
        features = []
        for n_fft in self.fft_sizes:
            spectrum = torch.stft(audio, n_fft, hop_length=n_fft // 4,
                                  window=getattr(self, f"window_{n_fft}"), return_complex=True).abs()
            mel = getattr(self, f"mel_{n_fft}") @ (spectrum / n_fft)
            log_mel = torch.log1p(1000 * mel)
            features.append(F.adaptive_avg_pool1d(log_mel, self.time_bins).flatten(1))
        return torch.cat(features, dim=1)

    def loss(self, predicted_audio: torch.Tensor, target_audio: torch.Tensor) -> torch.Tensor:
        return F.l1_loss(self(predicted_audio), self(target_audio))


class TinyFusion(nn.Module):
    """Audio-conditioning tokens reusable by direct or future generative preset heads."""

    def __init__(self):
        super().__init__()
        width = TRANSFORMER["width"]
        self.frames = nn.Linear(880, width)
        self.energy = nn.Linear(1, width)
        self.position = nn.Parameter(torch.randn(1, VALID_FRAMES + 1, width) * 0.02)
        layer = nn.TransformerEncoderLayer(width, TRANSFORMER["heads"], TRANSFORMER["feedforward"],
                                          dropout=0.0, batch_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=1, enable_nested_tensor=False)

    def forward(self, timbre: torch.Tensor, performance: torch.Tensor, level: torch.Tensor) -> torch.Tensor:
        frames = self.frames(torch.cat((timbre, performance), dim=-1))
        tokens = torch.cat((self.energy(level).unsqueeze(1), frames), dim=1)
        return self.transformer(tokens + self.position)


class AudioToDense(nn.Module):
    def __init__(self, checkpoint: Path | None = BP_CHECKPOINT):
        super().__init__()
        self.performance = BasicPitchICASSP2022()
        self.timbre = BasicPitchICASSP2022()
        if checkpoint is not None:
            state = torch.load(checkpoint, map_location="cpu", weights_only=True)
            self.performance.load_state_dict(state)
            self.timbre.load_state_dict(state)
        self.performance.requires_grad_(False).eval()
        self.fusion = TinyFusion()
        self.head = nn.Linear(TRANSFORMER["width"], 4)

    def train(self, mode: bool = True):
        super().train(mode)
        self.performance.eval()  # Freeze BatchNorm buffers as well as parameters.
        return self

    @staticmethod
    def sequence(outputs: dict[str, torch.Tensor]) -> torch.Tensor:
        # Exclude frames belonging to the zero-padding of BasicPitch's fixed input window.
        return torch.cat([outputs[key][:, :VALID_FRAMES] for key in ("note", "onset", "contour")], dim=-1)

    def encode_audio(self, audio: torch.Tensor) -> torch.Tensor:
        """Return [batch, 53, 32] conditioning independently of the preset prediction head."""
        if audio.ndim != 2 or audio.shape[1] != SAMPLES:
            raise ValueError(f"expected audio [batch, {SAMPLES}]")
        padded = F.pad(audio, (0, AUDIO_N_SAMPLES - SAMPLES)).unsqueeze(-1)
        with torch.no_grad():
            performance = self.sequence(self.performance(padded))
        timbre = self.sequence(self.timbre(padded))
        # BasicPitch normalizes loudness; retain one global level feature for the level control.
        level = torch.log(audio.square().mean(dim=1, keepdim=True).clamp_min(1e-10))
        return self.fusion(timbre, performance, level)

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        return self.head(self.encode_audio(audio)[:, 0])


def state_digest(module: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in module.state_dict().items():
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def gradient_norm(module: nn.Module) -> float:
    return sum(float(parameter.grad.detach().square().sum()) for parameter in module.parameters()
               if parameter.grad is not None) ** 0.5


def device_from_name(name: str) -> torch.device:
    if name == "xpu" and not torch.xpu.is_available():
        raise RuntimeError("XPU requested but unavailable; use --device cpu explicitly for CPU checks")
    return torch.device(name)


class SpectralSurrogate(nn.Module):
    """Predict the loss features of Vital; this is not a waveform synthesizer."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(4, 64), nn.SiLU(), nn.Linear(64, 128), nn.SiLU(),
                                 nn.Linear(128, PerceptualSpectrum.size), nn.Softplus())

    def forward(self, controls: torch.Tensor) -> torch.Tensor:
        return self.net(controls)


def surrogate_loss(dense: torch.Tensor, target: torch.Tensor, surrogate: SpectralSurrogate) -> torch.Tensor:
    # Enumerate hard categories. Never feed soft categories outside the proxy's training domain.
    batch = dense.shape[0]
    categories = torch.eye(3, device=dense.device).repeat(batch, 1)
    levels = dense[:, 3:].sigmoid().repeat_interleave(3, dim=0)
    spectra = surrogate(torch.cat((categories, levels), dim=1)).reshape(batch, 3, -1)
    costs = (spectra - target[:, None]).abs().mean(dim=2)
    return (dense[:, :3].softmax(dim=1) * costs).sum(dim=1).mean()


class DynamicAudio:
    def __init__(self, codec: PresetCodec, plugin_path: str | None = None):
        self.codec = codec
        self.renderer = VitalRenderer(plugin_path)
        self.performance = Performance(ticks_per_beat=480, bpm=120)
        self.performance.add_note(pitch=60, velocity=100, start_tick=0, duration_ticks=480)
        self.performance.end_tick = 480
        self.render_count = 0

    def render(self, preset: VitalPreset) -> torch.Tensor:
        result = self.renderer.render(RenderRequest(preset=preset, performance=self.performance,
                                      sample_rate=SAMPLE_RATE, tail_seconds=0.1, renderer_id=self.renderer.renderer_id))
        self.render_count += 1
        audio = torch.from_numpy(result.audio.copy()).mean(dim=1)
        if audio.shape != (SAMPLES,) or not torch.isfinite(audio).all() or audio.square().mean() < 1e-10:
            raise RuntimeError("Vital returned invalid or silent audio")
        return audio

    def batch(self, size: int, generator: torch.Generator, *, balanced: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        categories = torch.arange(size) % 3 if balanced else torch.randint(3, (size,), generator=generator)
        levels = torch.rand(size, 1, generator=generator)
        controls = torch.cat((F.one_hot(categories, 3).float(), levels), dim=1)
        audio = torch.stack([self.render(self.codec.preset(int(c), float(level)))
                             for c, level in zip(categories, levels[:, 0])])
        return controls, audio

    def verify_controls(self) -> dict:
        frequencies = []
        for category, octave in enumerate(OCTAVES):
            audio = self.render(self.codec.preset(category, 0.5)).numpy()
            frequencies.append(float(np.fft.rfftfreq(len(audio), 1 / SAMPLE_RATE)[np.abs(np.fft.rfft(audio)).argmax()]))
            expected = 261.625565 * 2 ** (octave / 12)
            if abs(frequencies[-1] - expected) > 3:
                raise RuntimeError("rendered oscillator pitch does not match the preset; check state loading")
        quiet = self.render(self.codec.preset(1, 0)).square().mean().sqrt()
        loud = self.render(self.codec.preset(1, 1)).square().mean().sqrt()
        ratio = float(loud / quiet)
        if ratio < 2:
            raise RuntimeError("rendered oscillator level does not respond to the preset")
        return {"octave_peak_hz": frequencies, "loud_quiet_rms_ratio": ratio}


def update(loss: torch.Tensor, optimizer: torch.optim.Optimizer, module: nn.Module) -> float:
    if not torch.isfinite(loss):
        raise RuntimeError("non-finite training loss")
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    norm = nn.utils.clip_grad_norm_(module.parameters(), 10, error_if_nonfinite=True)
    if norm <= 0:
        raise RuntimeError("training objective produced no gradient")
    optimizer.step()
    return float(norm)


def train(args: argparse.Namespace) -> None:
    if min(args.proxy_steps, args.steps, args.batch_size) < 1:
        raise ValueError("step counts and batch size must be positive")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    device = device_from_name(args.device)
    train_rng = torch.Generator().manual_seed(args.seed)
    validation_rng = torch.Generator().manual_seed(args.seed + 1)
    codec, spectrum = PresetCodec(), PerceptualSpectrum().to(device)
    source = DynamicAudio(codec, args.plugin_path)
    control_response = source.verify_controls()
    model, proxy = AudioToDense().to(device), SpectralSurrogate().to(device)
    model.train()
    frozen_before, timbre_before = state_digest(model.performance), state_digest(model.timbre)
    fusion_before = state_digest(model.fusion)
    validation_controls, validation_audio = source.batch(6, validation_rng, balanced=True)
    validation_controls, validation_audio = validation_controls.to(device), validation_audio.to(device)
    validation_features = spectrum(validation_audio)
    history = []

    def evaluate_real() -> tuple[float, torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            was_training = model.training
            model.eval()
            dense = model(validation_audio)
            rendered = torch.stack([source.render(codec.decode(row)) for row in dense]).to(device)
            model.train(was_training)
            return float(spectrum.loss(rendered, validation_audio)), dense, rendered

    initial_real_loss, _, _ = evaluate_real()
    optimizer = torch.optim.Adam(proxy.parameters(), lr=3e-3)
    for step in range(args.proxy_steps):
        controls, audio = source.batch(args.batch_size, train_rng)
        loss = F.l1_loss(proxy(controls.to(device)), spectrum(audio.to(device)))
        norm = update(loss, optimizer, proxy)
        history.append({"stage": "surrogate", "step": step + 1, "loss": float(loss.detach()), "gradient_norm": norm})
        if step == 0 or (step + 1) % 10 == 0:
            print(json.dumps(history[-1]), flush=True)

    proxy.eval().requires_grad_(False)
    model.eval()
    with torch.no_grad():
        proxy_validation_loss = float(F.l1_loss(proxy(validation_controls), validation_features))
        initial_proxy_loss = float(surrogate_loss(model(validation_audio), validation_features, proxy))
    model.train()
    optimizer = torch.optim.Adam((parameter for parameter in model.parameters() if parameter.requires_grad), lr=1e-3)
    if device.type == "xpu":
        torch.xpu.reset_peak_memory_stats(device)
    for step in range(args.steps):
        _, audio = source.batch(args.batch_size, train_rng)
        audio = audio.to(device)
        if device.type == "xpu":
            torch.xpu.synchronize(device)
        started = time.perf_counter()
        loss = surrogate_loss(model(audio), spectrum(audio), proxy)
        norm = update(loss, optimizer, model)
        timbre_gradient, head_gradient, fusion_gradient = gradient_norm(model.timbre), gradient_norm(model.head), gradient_norm(model.fusion)
        if min(timbre_gradient, head_gradient, fusion_gradient) <= 0 or any(p.grad is not None for p in model.performance.parameters()):
            raise RuntimeError("frozen/trainable branch gradient contract failed")
        if device.type == "xpu":
            torch.xpu.synchronize(device)
        history.append({"stage": "audio_to_dense", "step": step + 1, "loss": float(loss.detach()), "gradient_norm": norm,
                        "timbre_gradient_norm": timbre_gradient, "head_gradient_norm": head_gradient,
                        "fusion_gradient_norm": fusion_gradient, "model_step_seconds": time.perf_counter() - started})
        if step == 0 or (step + 1) % 10 == 0:
            print(json.dumps(history[-1]), flush=True)

    model.eval()
    if (frozen_before != state_digest(model.performance) or timbre_before == state_digest(model.timbre)
            or fusion_before == state_digest(model.fusion)):
        raise RuntimeError("branch weights or frozen running statistics violated the training contract")
    peak_xpu_bytes = torch.xpu.max_memory_allocated(device) if device.type == "xpu" else None
    final_real_loss, dense, rendered = evaluate_real()
    with torch.no_grad():
        final_proxy_loss = float(surrogate_loss(model(validation_audio), validation_features, proxy))
    codec.save(codec.decode(dense[0]), output / "predicted.vital")
    codec.save(codec.preset(int(validation_controls[0, :3].argmax()), float(validation_controls[0, 3])), output / "target.vital")
    wavfile.write(output / "target.wav", SAMPLE_RATE, validation_audio[0].cpu().numpy())
    wavfile.write(output / "predicted.wav", SAMPLE_RATE, rendered[0].cpu().numpy())
    checkpoint = {"contract": CONTRACT, "template_json": codec.template.to_json(canonical=True),
                  "preset_contract": codec.output_contract.spec,
                  "model": {k: v.cpu() for k, v in model.state_dict().items()},
                  "surrogate": {k: v.cpu() for k, v in proxy.state_dict().items()}}
    torch.save(checkpoint, output / "checkpoint.pt")
    (output / "preset_contract.json").write_text(json.dumps(codec.output_contract.spec, indent=2) + "\n", encoding="utf-8")
    report = {"contract": CONTRACT, "seed": args.seed, "proxy_steps": args.proxy_steps, "steps": args.steps,
              "device": str(device), "torch_version": str(torch.__version__),
              "device_name": torch.xpu.get_device_name(device) if device.type == "xpu" else "CPU",
              "basic_pitch_checkpoint_sha256": hashlib.sha256(BP_CHECKPOINT.read_bytes()).hexdigest(),
              "frozen_branch_unchanged": frozen_before == state_digest(model.performance),
              "trainable_branch_changed": timbre_before != state_digest(model.timbre),
              "transformer_fusion_changed": fusion_before != state_digest(model.fusion),
              "model_parameters": sum(p.numel() for p in model.parameters()),
              "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
              "fusion_parameters": sum(p.numel() for p in model.fusion.parameters()),
              "peak_xpu_allocated_bytes": peak_xpu_bytes,
              "batch_size": args.batch_size, "validation_examples": len(validation_audio),
              "renderer_id": source.renderer.renderer_id, "render_count": source.render_count,
              "control_response": control_response,
              "surrogate_validation_l1": proxy_validation_loss,
              "initial_expected_surrogate_loss": initial_proxy_loss, "final_expected_surrogate_loss": final_proxy_loss,
              "initial_real_vital_spectral_loss": initial_real_loss, "final_real_vital_spectral_loss": final_real_loss,
              "real_vital_loss_improved": final_real_loss < initial_real_loss,
              "predicted_dense": dense.tolist(), "history": history}
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k not in ("contract", "history", "predicted_dense")}), flush=True)


def read_audio(path: str) -> torch.Tensor:
    rate, audio = wavfile.read(path)
    if np.issubdtype(audio.dtype, np.unsignedinteger):
        audio = (audio.astype(np.float32) - 128) / 128
    elif np.issubdtype(audio.dtype, np.signedinteger):
        audio = audio.astype(np.float32) / -float(np.iinfo(audio.dtype).min)
    else:
        audio = audio.astype(np.float32)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if not np.isfinite(audio).all() or len(audio) == 0:
        raise ValueError("input audio must be nonempty and finite")
    if rate != SAMPLE_RATE:
        divisor = np.gcd(rate, SAMPLE_RATE)
        audio = resample_poly(audio, SAMPLE_RATE // divisor, rate // divisor)
    if len(audio) != SAMPLES:
        raise ValueError("this smoke model expects exactly 0.6 seconds of audio; no silent crop or padding")
    return torch.from_numpy(audio.copy()).unsqueeze(0)


def predict(args: argparse.Namespace) -> None:
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    codec = PresetCodec()
    if (checkpoint["contract"] != CONTRACT or checkpoint["template_json"] != codec.template.to_json(canonical=True)
            or checkpoint.get("preset_contract") != codec.output_contract.spec):
        raise ValueError("checkpoint does not match this decoder/template contract")
    destination = Path(args.output)
    if destination.exists():
        raise FileExistsError(destination)
    device = device_from_name(args.device)
    model = AudioToDense(checkpoint=None).to(device).eval()
    model.load_state_dict(checkpoint["model"])
    with torch.no_grad():
        dense = model(read_audio(args.audio).to(device))[0]
    codec.save(codec.decode(dense), destination)
    print(json.dumps({"output": str(destination), "dense": dense.tolist(), "preset_contract": codec.output_contract.spec["id"]}))


def validate_preset(args: argparse.Namespace) -> None:
    contract = PresetContract()
    document, static_report = contract.read(args.preset)
    runtime_checked = static_report.valid and (args.runtime or args.render)
    report = contract.validate(document, runtime=True) if runtime_checked else static_report
    result = {"contract": contract.spec["id"], "preset_version": contract.spec["preset_version"],
              "static_valid": static_report.valid, "runtime_checked": runtime_checked,
              "render_checked": False, **report.to_dict()}
    if report.valid and args.render:
        try:
            source = DynamicAudio(PresetCodec(), args.plugin_path)
            audio = source.render(VitalPreset(document, contract.template.schema)).numpy()
            frequency = float(np.fft.rfftfreq(len(audio), 1 / SAMPLE_RATE)[np.abs(np.fft.rfft(audio)).argmax()])
            expected = 261.625565 * 2 ** (document["settings"]["osc_1_transpose"] / 12)
            if abs(frequency - expected) > 3:
                raise RuntimeError("rendered pitch does not match the requested octave")
            result.update(render_checked=True, renderer_id=source.renderer.renderer_id, peak_hz=frequency)
        except Exception as error:
            result["valid"] = False
            result["diagnostics"].append({"code": "contract.render", "severity": "error", "message": str(error)})
    print(json.dumps(result, allow_nan=False))
    if not result["valid"]:
        raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    training = commands.add_parser("train", help="generate fresh Vital audio and train the two small networks")
    training.add_argument("--output", required=True, help="new output directory (never overwritten)")
    training.add_argument("--proxy-steps", type=int, default=100)
    training.add_argument("--steps", type=int, default=100)
    training.add_argument("--batch-size", type=int, default=4)
    training.add_argument("--seed", type=int, default=7)
    training.add_argument("--plugin-path")
    training.add_argument("--device", choices=("cpu", "xpu"), default="xpu")
    training.set_defaults(func=train)
    inference = commands.add_parser("predict", help="audio -> dense logits -> validated .vital JSON; no renderer needed")
    inference.add_argument("--checkpoint", required=True)
    inference.add_argument("--audio", required=True)
    inference.add_argument("--output", required=True)
    inference.add_argument("--device", choices=("cpu", "xpu"), default="xpu")
    inference.set_defaults(func=predict)
    validation = commands.add_parser("validate", help="strictly validate a generated .vital file against the output contract")
    validation.add_argument("preset")
    validation.add_argument("--runtime", action="store_true", help="also validate Vita's preset round-trip")
    validation.add_argument("--render", action="store_true", help="also load in official Vital and verify audible output/pitch")
    validation.add_argument("--plugin-path")
    validation.set_defaults(func=validate_preset)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
