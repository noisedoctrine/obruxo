"""Qualify the extracted SpanSynth encoder and Stage 2 connectors on Intel XPU."""

# ruff: noqa: E402
import argparse
import importlib
import json
import math
from pathlib import Path
import sys
from time import perf_counter
from types import SimpleNamespace

import torch

TRAINING = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(TRAINING.parent / "data_generation"), str(TRAINING)]

from obruxo_data.midi import Performance
from obruxo_training.issue187_stage2 import (
    EventSetEncoder, Issue187Adapter, MIDI_ENCODER_REVISION,
    encode_midi_spans, load_midi_encoder, performance_to_spans, trainable_parameter_count,
)
from obruxo_training.codec import PilotCodec
from obruxo_training.surrogate_codec import SurrogateCodec


def check_gradients(module, controls, frozen):
    if controls.grad is None or not torch.isfinite(controls.grad).all() or controls.grad.norm() == 0:
        norm = None if controls.grad is None else float(controls.grad.norm())
        raise RuntimeError(f"adapter loss did not produce a finite, nonzero patch-control gradient (norm={norm})")
    active = [p for p in module.parameters() if p.requires_grad]
    gradients = [p.grad for p in active if p.grad is not None]
    if len(gradients) != len(active) or any(not torch.isfinite(grad).all() for grad in gradients):
        raise RuntimeError("adapter has missing or nonfinite trainable gradients")
    if not any(grad.norm() > 0 for grad in gradients):
        raise RuntimeError("adapter loss did not produce a nonzero trainable gradient")
    if any(p.requires_grad or p.grad is not None for p in frozen.parameters()):
        raise RuntimeError("a pretrained parameter was trainable or accumulated a gradient")


def performance():
    score = Performance(bpm=120)
    score.add_note(pitch=48, velocity=72, start_tick=0, duration_ticks=360)
    score.add_note(pitch=55, velocity=104, start_tick=120, duration_ticks=720)
    score.add_note(pitch=60, velocity=48, start_tick=240, duration_ticks=240)
    return score


def compare_midi_encoder(encoder_path: Path, upstream_source: Path, device: torch.device) -> dict:
    sys.path.insert(0, str(upstream_source))
    upstream = importlib.import_module("spansynth.model")
    encoder = load_midi_encoder(encoder_path)
    reference = upstream.EventSetEncoder(upstream.EventSetConfig())
    reference.load_state_dict(encoder.state_dict(), strict=True)
    reference.eval().requires_grad_(False)
    notes = performance_to_spans(performance(), default_program=80)
    rows = encode_midi_spans(notes, duration_seconds=1.0)
    local_cpu = EventSetEncoder().eval().requires_grad_(False)
    local_cpu.load_state_dict(encoder.state_dict(), strict=True)
    with torch.inference_mode():
        cpu_result = local_cpu(rows["kind_id"], rows["pitch_id"], rows["numeric"], rows["event_valid"])
        ref_cpu = reference(rows["kind_id"], rows["pitch_id"], rows["numeric"], rows["event_valid"])
    if not torch.equal(cpu_result, ref_cpu):
        raise RuntimeError(f"CPU checkpoint parity failed; max absolute difference={(cpu_result-ref_cpu).abs().max().item()}")
    xpu_rows = {key: value.to(device) for key, value in rows.items() if key != "frame_valid"}
    xpu_encoder = load_midi_encoder(encoder_path, device=device)
    reference = reference.to(device)
    torch.xpu.reset_peak_memory_stats(device)
    started = perf_counter()
    with torch.inference_mode():
        xpu_result = xpu_encoder(xpu_rows["kind_id"], xpu_rows["pitch_id"],
                                 xpu_rows["numeric"], xpu_rows["event_valid"])
        ref_xpu = reference(xpu_rows["kind_id"], xpu_rows["pitch_id"],
                            xpu_rows["numeric"], xpu_rows["event_valid"])
        torch.xpu.synchronize(device)
    elapsed = perf_counter() - started
    if not torch.allclose(xpu_result, ref_xpu, rtol=2e-5, atol=2e-5):
        raise RuntimeError(f"XPU checkpoint parity failed; max absolute difference={(xpu_result-ref_xpu).abs().max().item()}")
    difference = (xpu_result.cpu() - cpu_result).abs()
    if not torch.allclose(xpu_result.cpu(), cpu_result, rtol=2e-5, atol=2e-5):
        raise RuntimeError(f"CPU/XPU numerical qualification failed; max absolute difference={difference.max().item()}")
    return {"shape": list(xpu_result.shape), "device": str(xpu_result.device),
            "cpu_reference_max_abs": float((cpu_result - ref_cpu).abs().max()),
            "xpu_reference_max_abs": float((xpu_result.cpu() - ref_xpu.cpu()).abs().max()),
            "cpu_xpu_max_abs": float(difference.max()), "forward_seconds": elapsed,
            "peak_allocated_bytes": int(torch.xpu.max_memory_allocated(device)),
            "encoder_parameters_frozen": all(not p.requires_grad and p.grad is None for p in xpu_encoder.parameters())}


def load_generator(source: Path, snapshot: Path, device: torch.device):
    sys.path.insert(0, str(source))
    from stable_audio_3.loading_utils import load_diffusion_cond

    config = json.loads((snapshot / "model_config.json").read_text(encoding="utf-8"))
    conditioning = config["model"]["conditioning"]
    conditioning["configs"] = [row for row in conditioning["configs"] if row["id"] != "prompt"]
    model = load_diffusion_cond(config, str(snapshot / "model.safetensors"), device=device, model_half=True)
    return model, int(conditioning["cond_dim"])


def check_direct_arm(backbone: str, encoder_tokens: torch.Tensor, encoder_mask: torch.Tensor,
                     codec, device: torch.device) -> dict:
    torch.manual_seed(183)
    adapter = Issue187Adapter(backbone, SurrogateCodec(PilotCodec(), expanded=True).width, 256).to(device)
    controls = torch.randn(1, SurrogateCodec(PilotCodec(), expanded=True).width, device=device, requires_grad=True)
    frames = max(1, codec.sample_rate // codec.downsampling_ratio)
    tokens = encoder_tokens.to(device)
    torch.xpu.reset_peak_memory_stats(device)
    started = perf_counter()
    output = adapter(controls, tokens, encoder_mask.to(device), target_frames=frames,
                     target_hz=codec.sample_rate / codec.downsampling_ratio)
    codec_frozen = all(not p.requires_grad for p in codec.parameters())
    if not codec_frozen:
        raise RuntimeError("codec weights were not frozen")
    sentinel = next(codec.parameters()).detach().clone()
    decoded = codec.decode_audio(output)
    decoded.square().mean().backward()
    torch.xpu.synchronize(device)
    elapsed = perf_counter() - started
    check_gradients(adapter, controls, codec)
    if not torch.equal(next(codec.parameters()), sentinel):
        raise RuntimeError("frozen codec weights changed")
    return {"backbone": backbone, "trainable_parameters": trainable_parameter_count(adapter),
            "latent_shape": list(output.shape), "audio_shape": list(decoded.shape),
            "forward_backward_seconds": elapsed,
            "peak_allocated_bytes": int(torch.xpu.max_memory_allocated(device)),
            "patch_gradient_norm": float(controls.grad.norm()), "codec_frozen": True}


def check_generator_arm(backbone: str, encoder_tokens: torch.Tensor, encoder_mask: torch.Tensor,
                        generator, condition_width: int, codec, device: torch.device) -> dict:
    from issue187_surrogate_adapters import Adapter as ExistingStage2Adapter

    if any(p.requires_grad or p.grad is not None for p in generator.parameters()):
        raise RuntimeError("pretrained generator was not frozen before sampler qualification")
    torch.manual_seed(183)
    latent_hop = codec.downsampling_ratio
    frames = max(1, generator.min_input_length // latent_hop, math.ceil(codec.sample_rate / latent_hop))
    duration = frames * latent_hop / codec.sample_rate
    token_frames = math.ceil(duration * 25)
    control_width = SurrogateCodec(PilotCodec(), expanded=True).width
    adapter = Issue187Adapter(backbone, control_width, condition_width, generator=True).to(device)
    controls = torch.randn(1, control_width, device=device, requires_grad=True)
    torch.xpu.reset_peak_memory_stats(device)
    started = perf_counter()
    conditioning, mask = adapter(controls, encoder_tokens.to(device), encoder_mask.to(device),
                                 target_frames=token_frames, target_hz=25)
    valid_length = int(mask[0].sum().item())
    if valid_length == 0:
        raise RuntimeError("generator conditioning has no valid MIDI frames")
    conditioning, mask = conditioning[:, :valid_length], mask[:, :valid_length]
    conditioning.retain_grad()
    latent = torch.randn(1, 256, frames, device=device)
    sentinel = next(generator.parameters()).detach().clone()
    # Reuse PR #188's tested Stable Audio prompt/flow connector for this adapter's tokens.
    bridge = SimpleNamespace(codec=SimpleNamespace(hop=latent_hop), generator=generator)
    steps = 2
    trajectory = []
    endpoint_velocity = ExistingStage2Adapter.velocity(bridge, latent, latent.new_ones((1,)),
                                                         conditioning, duration, prompt_mask=mask)
    endpoint = {"time": 1.0, "latent_input_finite": bool(torch.isfinite(latent).all()),
                "velocity_finite": bool(torch.isfinite(endpoint_velocity).all()),
                "velocity_abs_max": float(endpoint_velocity.detach().abs().max())}
    if not endpoint["latent_input_finite"] or not endpoint["velocity_finite"]:
        raise RuntimeError(f"Stable Audio endpoint produced a nonfinite tensor: {endpoint}")
    for step in range(steps):
        time = latent.new_full((1,), 1 - (step + .5) / steps)
        velocity = ExistingStage2Adapter.velocity(bridge, latent, time, conditioning, duration,
                                                   prompt_mask=mask)
        trajectory.append({"time": float(time[0]), "latent_finite": bool(torch.isfinite(latent).all()),
                           "latent_abs_max": float(latent.detach().abs().max()),
                           "velocity_finite": bool(torch.isfinite(velocity).all()),
                           "velocity_abs_max": float(velocity.detach().abs().max())})
        if not trajectory[-1]["velocity_finite"]:
            raise RuntimeError(f"Stable Audio sampler velocity became nonfinite: {trajectory}")
        latent = latent - velocity.to(latent.dtype) / steps
        if not torch.isfinite(latent).all():
            raise RuntimeError(f"Stable Audio sampler latent became nonfinite: {trajectory}")
    latent.retain_grad()
    audio = codec.decode_audio(latent * generator.pretransform.scale)
    if not torch.isfinite(audio).all():
        raise RuntimeError(f"Stable Audio decoder produced nonfinite audio after sampler: {trajectory}")
    audio.float().square().mean().backward()
    torch.xpu.synchronize(device)
    elapsed = perf_counter() - started
    if any(p.requires_grad or p.grad is not None for p in generator.parameters()):
        raise RuntimeError("pretrained generator became trainable or accumulated gradients")
    if controls.grad is not None and not torch.isfinite(controls.grad).all():
        cond_grad = conditioning.grad
        latent_grad = latent.grad
        grad_summary = lambda value: "missing" if value is None else f"finite={bool(torch.isfinite(value).all())}, abs_max={float(value.detach().abs().max())}"
        raise RuntimeError(f"generator backward has nonfinite patch gradients; trajectory={trajectory}, "
                           f"audio_finite={bool(torch.isfinite(audio).all())}, "
                           f"conditioning_gradient=({grad_summary(cond_grad)}), "
                           f"decoder_latent_gradient=({grad_summary(latent_grad)}), "
                           f"forward_backward_seconds={elapsed}, "
                           f"peak_allocated_bytes={int(torch.xpu.max_memory_allocated(device))}, "
                           f"adapter_parameters={trainable_parameter_count(adapter)}")
    check_gradients(adapter, controls, generator)
    if not torch.equal(next(generator.parameters()), sentinel):
        raise RuntimeError("frozen generator weights changed")
    return {"backbone": backbone, "trainable_parameters": trainable_parameter_count(adapter),
            "conditioning_shape": list(conditioning.shape), "velocity_shape": list(velocity.shape),
            "decoded_audio_shape": list(audio.shape), "sampler_steps": steps,
            "endpoint_diagnostic": endpoint, "sampler_trajectory": trajectory,
            "forward_backward_seconds": elapsed,
            "peak_allocated_bytes": int(torch.xpu.max_memory_allocated(device)),
            "patch_gradient_norm": float(controls.grad.norm()), "generator_frozen": True,
            "pr188_flow_connector_reused": True}


def check_generator_training_objective(backbone: str, encoder_tokens: torch.Tensor, encoder_mask: torch.Tensor,
                                       generator, condition_width: int, codec, device: torch.device) -> dict:
    from issue187_surrogate_adapters import Adapter as ExistingStage2Adapter

    torch.manual_seed(183)
    latent_hop = codec.downsampling_ratio
    frames = max(1, generator.min_input_length // latent_hop, math.ceil(codec.sample_rate / latent_hop))
    duration = frames * latent_hop / codec.sample_rate
    token_frames = math.ceil(duration * 25)
    control_width = SurrogateCodec(PilotCodec(), expanded=True).width
    adapter = Issue187Adapter(backbone, control_width, condition_width, generator=True).to(device)
    controls = torch.randn(1, control_width, device=device, requires_grad=True)
    target = torch.randn(1, 256, frames, device=device)
    scale = generator.pretransform.scale
    scaled_target = target / scale
    noise = torch.randn_like(scaled_target)
    time = torch.full((1,), .5, device=device)
    noisy = torch.lerp(scaled_target, noise, time)
    torch.xpu.reset_peak_memory_stats(device)
    started = perf_counter()
    conditioning, mask = adapter(controls, encoder_tokens.to(device), encoder_mask.to(device),
                                 target_frames=token_frames, target_hz=25)
    valid_length = int(mask[0].sum().item())
    if valid_length == 0:
        raise RuntimeError("generator training objective has no valid MIDI frames")
    conditioning, mask = conditioning[:, :valid_length], mask[:, :valid_length]
    bridge = SimpleNamespace(codec=SimpleNamespace(hop=latent_hop), generator=generator)
    velocity = ExistingStage2Adapter.velocity(bridge, noisy, time, conditioning, duration, prompt_mask=mask)
    loss = torch.nn.functional.mse_loss(velocity.float(), (noise - scaled_target).float())
    if not torch.isfinite(velocity).all() or not torch.isfinite(loss):
        raise RuntimeError("native flow-matching objective produced nonfinite predictions or loss")
    loss.backward()
    torch.xpu.synchronize(device)
    elapsed = perf_counter() - started
    check_gradients(adapter, controls, generator)
    return {"backbone": backbone, "trainable_parameters": trainable_parameter_count(adapter),
            "conditioning_shape": list(conditioning.shape), "velocity_shape": list(velocity.shape),
            "loss": float(loss.detach()), "velocity_abs_max": float(velocity.detach().abs().max()),
            "forward_backward_seconds": elapsed,
            "peak_allocated_bytes": int(torch.xpu.max_memory_allocated(device)),
            "patch_gradient_norm": float(controls.grad.norm()), "generator_frozen": True,
            "pr188_flow_connector_reused": True}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--encoder", type=Path, required=True)
    parser.add_argument("--upstream-source", type=Path, required=True)
    parser.add_argument("--stable-audio-3-source", type=Path, required=True)
    parser.add_argument("--stable-audio-tools-source", type=Path, required=True)
    parser.add_argument("--same-s-codec-snapshot", type=Path, required=True)
    parser.add_argument("--same-s-generator-snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not torch.xpu.is_available():
        raise RuntimeError("Intel XPU is unavailable")
    sys.path[:0] = [str(args.stable_audio_tools_source), str(args.stable_audio_3_source)]
    device = torch.device("xpu:0")
    torch.manual_seed(187)
    encoder = load_midi_encoder(args.encoder)
    result = {"device": str(device), "midi_encoder_revision": MIDI_ENCODER_REVISION,
              "midi_encoder": compare_midi_encoder(args.encoder, args.upstream_source, device)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    notes = performance_to_spans(performance(), default_program=80)
    rows = encode_midi_spans(notes, duration_seconds=1.0)
    with torch.inference_mode():
        tokens = encoder(rows["kind_id"], rows["pitch_id"], rows["numeric"], rows["event_valid"])
    mask = rows["frame_valid"]
    from check_issue187_pretrained_xpu import load_codec

    codec = load_codec("same-codec", args.same_s_codec_snapshot, device)
    result["direct_decoder"] = [check_direct_arm(name, tokens, mask, codec, device)
                                 for name in ("residual_mlp", "transformer")]
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    generator, condition_width = load_generator(args.stable_audio_3_source,
                                               args.same_s_generator_snapshot, device)
    result["generator_training_objective"] = []
    result["generator_conditioning"] = []
    for name in ("residual_mlp", "transformer"):
        for key, check in (("generator_training_objective", check_generator_training_objective),
                           ("generator_conditioning", check_generator_arm)):
            try:
                result[key].append(check(name, tokens, mask, generator, condition_width, codec, device))
            except (RuntimeError, ValueError) as exc:
                result[key].append({"backbone": name, "status": "failed", "error": str(exc),
                                    "phase": key, "generator_frozen": True})
            args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    result["generator_condition_width"] = condition_width
    del generator, codec
    torch.xpu.empty_cache()
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
