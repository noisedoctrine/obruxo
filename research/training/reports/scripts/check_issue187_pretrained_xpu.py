"""Exercise the pinned Stable Audio models on Intel XPU with local checkpoints."""

import argparse
import json
import math
from pathlib import Path

import torch
from safetensors import safe_open


def load_codec(kind, snapshot, device):
    config_path = snapshot / "model_config.json"
    weights = snapshot / "model.safetensors"
    if kind == "same-codec":
        from stable_audio_3.loading_utils import load_autoencoder

        codec = load_autoencoder(str(config_path), str(weights), device="cpu")
        codec.pretransform.enable_grad = True
    else:
        from stable_audio_tools.models.autoencoders import create_autoencoder_from_config

        config = json.loads(config_path.read_text(encoding="utf-8"))
        codec = create_autoencoder_from_config({
            "sample_rate": config["sample_rate"], "model": config["model"]["pretransform"]["config"],
        })
        prefix = "pretransform.model."
        with safe_open(weights, framework="pt", device="cpu") as checkpoint:
            codec.load_state_dict({
                key.removeprefix(prefix): checkpoint.get_tensor(key)
                for key in checkpoint.keys() if key.startswith(prefix)
            }, strict=True)
    return codec.eval().requires_grad_(False).to(device)


def check_codec(kind, snapshot, device):
    codec = load_codec(kind, snapshot, device)
    length = math.ceil(codec.sample_rate / codec.downsampling_ratio) * codec.downsampling_ratio
    time = torch.arange(length, device=device) / codec.sample_rate
    stereo = torch.stack(((time * (math.tau * 220)).sin(), 0.5 * (time * (math.tau * 330)).sin()))
    audio = (stereo[None] * 0.2).contiguous()
    with torch.no_grad():
        latents = codec.encode_audio(audio)
    latents = latents.detach().requires_grad_(True)
    decoded = codec.decode_audio(latents)
    if decoded.device != audio.device or decoded.shape[:2] != audio.shape[:2] or decoded.shape[-1] < length:
        raise RuntimeError(f"unexpected decoded audio device or shape: {decoded.device} {tuple(decoded.shape)}")
    (decoded[..., :length] - audio).square().mean().backward()
    if latents.grad is None or not torch.isfinite(decoded).all() or not torch.isfinite(latents.grad).all():
        raise RuntimeError("codec did not produce finite audio and latent gradients")
    grad_norm = latents.grad.norm().item()
    if grad_norm == 0:
        raise RuntimeError("decoder did not propagate a nonzero gradient into its input")
    return {"device": str(decoded.device), "latent_shape": list(latents.shape),
            "audio_shape": list(decoded.shape), "latent_gradient_norm": grad_norm}


def check_generator(snapshot, device, text_encoder):
    from stable_audio_3.loading_utils import load_diffusion_cond
    from stable_audio_3.model import StableAudioModel

    config = json.loads((snapshot / "model_config.json").read_text(encoding="utf-8"))
    for conditioner in config["model"]["conditioning"]["configs"]:
        if conditioner["type"] == "t5gemma":
            options = conditioner["config"]
            options["model_path"] = str(text_encoder or snapshot)
            if text_encoder is not None:
                options.pop("subfolder", None)
    model = load_diffusion_cond(config, str(snapshot / "model.safetensors"), device=device, model_half=True)
    parameter = next(model.model.parameters())
    if parameter.device.type != "xpu" or parameter.dtype != torch.float16:
        raise RuntimeError(f"generator placement is {parameter.device} {parameter.dtype}; expected FP16 on XPU")
    generator = StableAudioModel(model, config, device, model_half=True)
    seconds = 2
    audio = generator.generate(prompt="A soft sustained synthesizer note", duration=seconds,
                               duration_padding_sec=0, steps=2, seed=187)
    expected = (1, 2, seconds * model.sample_rate)
    if tuple(audio.shape) != expected or not torch.isfinite(audio).all() or audio.abs().max().item() == 0:
        raise RuntimeError(f"expected finite, non-silent stereo audio with shape {expected}; got {tuple(audio.shape)}")
    return {"device": str(parameter.device), "dtype": str(parameter.dtype), "audio_shape": list(audio.shape)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=("open-small-codec", "same-codec", "stable-audio-3-generator"))
    parser.add_argument("snapshot", type=Path, help="local snapshot with model_config.json and model.safetensors")
    parser.add_argument("--text-encoder", type=Path, help="local copy of the same T5Gemma checkpoint and tokenizer")
    args = parser.parse_args()
    if not torch.xpu.is_available():
        parser.error("PyTorch cannot access an Intel XPU")
    torch.set_num_threads(4)
    torch.manual_seed(187)
    device = torch.device("xpu:0")
    snapshot = args.snapshot.resolve()
    if args.kind == "stable-audio-3-generator":
        result = check_generator(snapshot, device, args.text_encoder.resolve() if args.text_encoder else None)
    else:
        result = check_codec(args.kind, snapshot, device)
    print(json.dumps({"kind": args.kind, "revision": snapshot.name, **result}), flush=True)


if __name__ == "__main__":
    main()
