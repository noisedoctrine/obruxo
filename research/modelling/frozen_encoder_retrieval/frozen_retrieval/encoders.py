from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import sys
import subprocess

from .core import MODEL_REGISTRY, canonical_json, digest, masked_temporal_pools


LOCK_VERSION = 1
ADAPTER_ID = "frozen_audio_adapters_v1"


class CacheIdentityDriftError(ValueError):
    """A feature cache exists under an identity that no longer matches its source contract."""


def file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def checkpoint_sha256(path: Path) -> str:
    if path.is_file():
        return file_sha256(path)
    if not path.is_dir():
        raise FileNotFoundError(path)
    hasher = hashlib.sha256()
    files = sorted(item for item in path.rglob("*") if item.is_file())
    if not files:
        raise ValueError("checkpoint directory is empty")
    for item in files:
        if item.is_symlink() or item.is_junction():
            raise ValueError("checkpoint directory cannot contain links")
        relative = item.relative_to(path).as_posix().encode("utf-8")
        hasher.update(len(relative).to_bytes(4, "little"))
        hasher.update(relative)
        hasher.update(bytes.fromhex(file_sha256(item)))
    return hasher.hexdigest()


def load_checkpoint_lock(path: Path) -> dict:
    lock = json.loads(path.read_text(encoding="utf-8"))
    if lock.get("version") != LOCK_VERSION or not isinstance(lock.get("models"), dict):
        raise ValueError("unsupported checkpoint lock")
    return lock


def resolve_checkpoint(model_name: str, checkpoint_root: Path, lock: dict) -> tuple[Path, dict, str]:
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"unknown model: {model_name}")
    entry = lock["models"].get(model_name)
    if not isinstance(entry, dict):
        raise ValueError(f"checkpoint lock has no entry for {model_name}")
    if entry.get("repository_id") != MODEL_REGISTRY[model_name]["repository_id"]:
        raise ValueError(f"checkpoint lock must name the exact upstream repository for {model_name}")
    relative = entry.get("path")
    if not isinstance(relative, str) or Path(relative).is_absolute() or ".." in Path(relative).parts:
        raise ValueError(f"checkpoint path for {model_name} must be relative to the declared checkpoint root")
    root = checkpoint_root.resolve(strict=True)
    path = root / relative
    if path.is_symlink() or path.is_junction():
        raise ValueError("checkpoint path cannot be a link")
    path = path.resolve(strict=True)
    if not path.is_relative_to(root):
        raise ValueError("checkpoint path escapes the declared checkpoint root")
    actual = checkpoint_sha256(path)
    expected = entry.get("sha256")
    if expected != actual:
        raise ValueError(f"checkpoint digest mismatch for {model_name}: expected exact locked digest")
    if MODEL_REGISTRY[model_name].get("expected_sha256") not in (None, actual):
        raise ValueError(f"checkpoint digest differs from the exact requested checkpoint for {model_name}")
    revision = entry.get("revision")
    if model_name == "cir_ast":
        if revision != "google-drive:1rbudnDYkc8-7KHm2jYk8ouyhob0zzCHI":
            raise ValueError("CIR lock must identify the authors' exact single-timbre Google Drive checkpoint")
    elif not isinstance(revision, str) or re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        raise ValueError(f"checkpoint lock must pin a 40-character Hugging Face source revision for {model_name}")
    loader_revision = entry.get("loader_revision")
    if not isinstance(loader_revision, str) or not loader_revision.strip():
        raise ValueError(f"checkpoint lock must record an exact loader/source revision for {model_name}")
    if entry.get("license_status") != "approved":
        raise PermissionError(f"checkpoint license is not approved in the lock for {model_name}")
    required_runtime = {package for package in dependency_status(model_name)}
    if not required_runtime.issubset(entry.get("runtime_versions", {})):
        raise ValueError(f"checkpoint lock must pin every runtime package version for {model_name}")
    return path, entry, actual


def _version(package: str) -> str:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return "missing"


def dependency_status(model_name: str) -> dict:
    modules = {
        "clap_music": {"numpy": "numpy", "scipy": "scipy", "torch": "torch", "laion_clap": "laion_clap"},
        "muq_mulan": {"numpy": "numpy", "scipy": "scipy", "torch": "torch", "muq": "muq"},
        "cir_ast": {"numpy": "numpy", "scipy": "scipy", "torch": "torch", "transformers": "transformers",
                     "librosa": "librosa"},
        "beats_iter3_plus": {"numpy": "numpy", "scipy": "scipy", "torch": "torch", "torchaudio": "torchaudio"},
        "encodec_24khz": {"numpy": "numpy", "scipy": "scipy", "torch": "torch", "transformers": "transformers"},
        "dac_44khz": {"numpy": "numpy", "scipy": "scipy", "torch": "torch", "transformers": "transformers"},
    }[model_name]
    return {package: {"module": module, "version": _version(package)} for package, module in modules.items()}


def _decode_audio(audio, np):
    audio = np.asarray(audio)
    if np.issubdtype(audio.dtype, np.integer):
        info = np.iinfo(audio.dtype)
        audio = audio.astype(np.float32) / float(max(abs(info.min), abs(info.max)))
    else:
        audio = audio.astype(np.float32, copy=False)
    if audio.ndim == 2:
        return audio.mean(axis=1)
    if audio.ndim != 1:
        raise ValueError("audio must be mono or frames-by-channels")
    return audio


def pipeline_dependency_status() -> dict:
    modules = {"PyYAML": "yaml", "mido": "mido", "numpy": "numpy", "scipy": "scipy",
               "DawDreamer": "dawdreamer", "torch": "torch"}
    packages = {"PyYAML": "PyYAML", "mido": "mido", "numpy": "numpy", "scipy": "scipy",
                "DawDreamer": "dawdreamer", "torch": "torch"}
    return {name: {"module": module, "version": _version(packages[name]), "available": importlib.util.find_spec(module) is not None}
            for name, module in modules.items()}


def _module(name: str):
    return importlib.import_module(name)


@dataclass
class FrozenAdapter:
    name: str
    checkpoint: Path
    lock_entry: dict
    checkpoint_digest: str
    device: str
    model: object
    torch: object
    numpy: object

    @property
    def sample_rate(self) -> int:
        return int(MODEL_REGISTRY[self.name]["sample_rate"])

    def _audio(self, path: Path):
        from scipy.io import wavfile
        import numpy as np
        from scipy.signal import resample_poly

        sample_rate, audio = wavfile.read(path)
        audio = _decode_audio(audio, np)
        if not np.isfinite(audio).all() or len(audio) == 0:
            raise ValueError("audio contains no finite samples")
        if sample_rate != self.sample_rate:
            factor = math.gcd(int(sample_rate), self.sample_rate)
            audio = resample_poly(audio, self.sample_rate // factor, int(sample_rate) // factor).astype(np.float32)
        duration = len(audio) / self.sample_rate
        if duration < 2.0 or duration > 10.0:
            raise ValueError("encoder input must be 2 to 10 seconds including the fixed release tail")
        return audio

    def extract(self, wav_path: Path) -> dict[str, list[float]]:
        waveform = self._audio(wav_path)
        torch, np, model, device = self.torch, self.numpy, self.model, self.device
        signal = torch.from_numpy(waveform).unsqueeze(0).to(device=device, dtype=torch.float32)
        with torch.inference_mode():
            if self.name == "clap_music":
                output = model.get_audio_embedding_from_data(x=[waveform], use_tensor=False)
                return {"official_clip": _clip_vector(output[0], self.name, torch, np)}
            if self.name == "muq_mulan":
                output = model(wavs=signal)
                return {"official_clip": _clip_vector(output[0], self.name, torch, np)}
            if self.name == "cir_ast":
                output = model.compute_audio_embeddings([signal], self.sample_rate)
                return {"official_clip": _clip_vector(output[0], self.name, torch, np)}
            if self.name == "beats_iter3_plus":
                features, _ = model.extract_features(signal, padding_mask=None)
                patch_size = int(model.input_patch_size)
                frequency_patches = 128 // patch_size
                if features.ndim != 3 or features.shape[-1] != 768 or features.shape[1] % frequency_patches:
                    raise ValueError("BEATs sequence output shape does not match the pinned iter3+ architecture")
                time_patches = features.shape[1] // frequency_patches
                time_series = features.reshape(1, frequency_patches, time_patches, 768).mean(dim=1)
                fbanks = model.preprocess(signal)
                valid_time = min(time_patches, fbanks.shape[1] // patch_size)
                frames = time_series[0, :valid_time].detach().float().cpu().numpy()
                return _pool_sequence(frames, self.name)
            if self.name == "encodec_24khz":
                encoder_input = signal.unsqueeze(1)
                if bool(getattr(model.config, "normalize", False)):
                    scale = encoder_input.mean(dim=1, keepdim=True).pow(2).mean(dim=-1, keepdim=True).sqrt() + 1e-8
                    encoder_input = encoder_input / scale
                features = model.encoder(encoder_input)
                _check_sequence_shape(features, self.name)
                frames = features[0].transpose(0, 1).detach().float().cpu().numpy()
                return _pool_sequence(frames, self.name)
            if self.name == "dac_44khz":
                encoder_input = signal.unsqueeze(1)
                features = model.encoder(encoder_input)
                _check_sequence_shape(features, self.name)
                frames = features[0].transpose(0, 1).detach().float().cpu().numpy()
                return _pool_sequence(frames, self.name)
        raise ValueError(f"adapter is not implemented for {self.name}")


def _as_vector(value, torch, np) -> list[float]:
    if hasattr(value, "detach"):
        value = value.detach().float().cpu().numpy()
    array = np.asarray(value, dtype=np.float32).reshape(-1)
    if array.size == 0 or not np.isfinite(array).all():
        raise ValueError("encoder returned an empty or non-finite vector")
    return array.tolist()


def _clip_vector(value, model_name: str, torch, np) -> list[float]:
    vector = _as_vector(value, torch, np)
    if len(vector) != MODEL_REGISTRY[model_name]["dimension"]:
        raise ValueError(f"{model_name} clip embedding dimension differs from its pinned adapter contract")
    return vector


def _check_sequence_shape(features, model_name: str) -> None:
    expected = MODEL_REGISTRY[model_name]["dimension"]
    if getattr(features, "ndim", 0) != 3 or features.shape[0] != 1 or features.shape[1] != expected or features.shape[2] <= 0:
        raise ValueError(f"{model_name} encoder tap must have shape [1,{expected},T] before quantization")


def _pool_sequence(frames, model_name: str) -> dict[str, list[float]]:
    if getattr(frames, "ndim", 0) != 2 or frames.shape[0] < 1 or frames.shape[1] != MODEL_REGISTRY[model_name]["dimension"]:
        raise ValueError(f"{model_name} pooled sequence shape does not match its frozen contract")
    values = frames.tolist()
    pools = masked_temporal_pools(values, valid_frames=len(values), bins=8)
    return {"masked_mean": pools["masked_mean"], "temporal_pyramid_8": pools["temporal_pyramid_8"]}


def load_adapter(model_name: str, checkpoint: Path, lock_entry: dict, checkpoint_digest: str, *, device: str = "cpu",
                 source_root: Path | None = None) -> FrozenAdapter:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    import numpy as np
    import torch

    for package, expected in lock_entry.get("runtime_versions", {}).items():
        actual = _version(package)
        if actual != expected:
            raise RuntimeError(f"runtime package version mismatch for {package}: expected {expected}, found {actual}")
    source_models = {"cir_ast", "beats_iter3_plus"}
    if model_name in source_models:
        if source_root is None:
            raise ValueError(f"{model_name} requires its exact local source checkout")
        current = subprocess.run(["git", "-C", str(source_root), "rev-parse", "HEAD"], check=True,
                                 capture_output=True, text=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(source_root), "status", "--porcelain"], check=True,
                               capture_output=True, text=True).stdout
        if current != lock_entry["loader_revision"] or dirty:
            raise ValueError(f"{model_name} source checkout differs from the clean locked loader revision")
    else:
        loader_package = {"clap_music": "laion_clap", "muq_mulan": "muq", "encodec_24khz": "transformers",
                          "dac_44khz": "transformers"}[model_name]
        if lock_entry["loader_revision"] != f"{loader_package}=={_version(loader_package)}":
            raise RuntimeError(f"installed {loader_package} version differs from the checkpoint lock")

    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if device == "xpu" and (not hasattr(torch, "xpu") or not torch.xpu.is_available()):
        raise RuntimeError("XPU was requested but is unavailable")
    if device not in {"cpu", "cuda", "xpu"}:
        raise ValueError("device must be explicitly cpu, cuda or xpu")
    if model_name == "clap_music":
        from laion_clap import CLAP_Module
        model = CLAP_Module(enable_fusion=False, amodel="HTSAT-base", device=device)
        model.load_ckpt(str(checkpoint), verbose=False)
    elif model_name == "muq_mulan":
        from muq import MuQMuLan
        model = MuQMuLan.from_pretrained(str(checkpoint)).to(device)
    elif model_name == "cir_ast":
        if source_root is None:
            raise ValueError("CIR requires a local checkout pinned to the checkpoint's source revision")
        source_root = source_root.resolve(strict=True)
        if str(source_root) not in sys.path:
            sys.path.insert(0, str(source_root))
        from AST import ASTFineTuned
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model = ASTFineTuned.from_custom_config(state["custom_config"])
        model.load_state_dict(state["model_state_dict"])
        model = model.to(device)
        if model.feature_extractor.sampling_rate != MODEL_REGISTRY[model_name]["sample_rate"]:
            raise ValueError("CIR feature-extractor sample rate differs from the frozen adapter contract")
    elif model_name == "beats_iter3_plus":
        if source_root is None:
            raise ValueError("BEATs requires a local checkout of Microsoft's official loader source")
        source_root = source_root.resolve(strict=True)
        if str(source_root) not in sys.path:
            sys.path.insert(0, str(source_root))
        from BEATs import BEATs, BEATsConfig
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        cfg = BEATsConfig(state["cfg"])
        model = BEATs(cfg)
        model.load_state_dict(state["model"])
        model.to(device)
    elif model_name in {"encodec_24khz", "dac_44khz"}:
        from transformers import AutoModel
        model = AutoModel.from_pretrained(str(checkpoint), local_files_only=True).to(device)
    else:
        raise ValueError(f"unknown model: {model_name}")
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("frozen encoder has trainable parameters")
    return FrozenAdapter(model_name, checkpoint, lock_entry, checkpoint_digest, device, model, torch, np)


def cache_feature(adapter: FrozenAdapter, wav_path: Path, *, request_id: str, render_metadata: dict,
                  output: Path, source_precision: str = "fp32") -> dict:
    audio_hash = file_sha256(wav_path)
    cache_identity = {"adapter_id": ADAPTER_ID, "model": adapter.name,
                      "checkpoint_sha256": adapter.checkpoint_digest, "source_revision": adapter.lock_entry["revision"],
                      "loader_revision": adapter.lock_entry["loader_revision"], "audio_file_sha256": audio_hash,
                      "renderer_request_id": request_id, "tap": MODEL_REGISTRY[adapter.name]["tap"],
                      "sample_rate": adapter.sample_rate,
                      "preprocessing": "pcm_scale_stereo_mean_mono_resample_poly_no_gain_norm_v2",
                      "encoder_internal_normalization": ("encodec_config_rms" if adapter.name == "encodec_24khz"
                                                          and bool(getattr(adapter.model.config, "normalize", False)) else "none"),
                      "pooling": "native_clip_v1" if MODEL_REGISTRY[adapter.name]["kind"] == "clip" else "masked_mean_and_8bin_mean_std_v1",
                      "precision": source_precision, "device": adapter.device}
    feature_id = digest(cache_identity)
    output.mkdir(parents=True, exist_ok=True)
    array_path, metadata_path = output / f"{feature_id}.npz", output / f"{feature_id}.json"
    if metadata_path.is_file():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise CacheIdentityDriftError("feature cache metadata is unreadable") from error
        if metadata.get("cache_identity") != cache_identity or metadata.get("feature_id") != feature_id:
            raise CacheIdentityDriftError("feature cache identity drift; stale features cannot be reused")
        if array_path.is_file() and file_sha256(array_path) == metadata.get("feature_sha256"):
            return {"feature_id": feature_id, "cache_path": array_path.name, "reused": True,
                    "model": adapter.name, "request_id": request_id, "audio_sha256": audio_hash}
    if render_metadata.get("provenance", {}).get("request_id") != request_id:
        raise ValueError("render provenance differs from feature request")
    features = adapter.extract(wav_path)
    np = adapter.numpy
    temporary = array_path.with_suffix(".partial.npz")
    np.savez_compressed(temporary, **{name: np.asarray(vector, dtype=np.float32) for name, vector in features.items()})
    temporary.replace(array_path)
    metadata = {"feature_id": feature_id, "cache_identity": cache_identity,
                "dimensions": {name: len(values) for name, values in features.items()},
                "frozen": True, "eval_mode": True, "feature_sha256": file_sha256(array_path)}
    metadata_path.write_text(json.dumps(metadata, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return {"feature_id": feature_id, "cache_path": array_path.name, "reused": False,
            "model": adapter.name, "request_id": request_id, "audio_sha256": audio_hash}


def available_candidates(checkpoint_root: Path | None, lock_path: Path | None) -> list[dict]:
    lock = load_checkpoint_lock(lock_path) if lock_path and lock_path.is_file() else {"models": {}}
    rows = []
    for name, spec in MODEL_REGISTRY.items():
        reasons = []
        entry = lock["models"].get(name)
        if checkpoint_root is None or not checkpoint_root.is_dir():
            reasons.append("checkpoint root not configured")
        elif entry is None:
            reasons.append("no verified local checkpoint lock entry")
        else:
            try:
                resolve_checkpoint(name, checkpoint_root, lock)
            except Exception as error:
                reasons.append(str(error))
        dependencies = dependency_status(name)
        missing = [package for package, value in dependencies.items() if value["version"] == "missing"]
        if missing:
            reasons.append("missing Python packages: " + ", ".join(missing))
        rows.append({"id": name, "name": spec["name"], "available": not reasons,
                     "unavailable_reasons": reasons, "sample_rate": spec["sample_rate"],
                     "tap": spec["tap"], "checkpoint": spec["checkpoint"], "source": spec.get("source"),
                     "license": spec["license"], "dependencies": dependencies})
    return rows


def list_clip_cache(cache_dir: Path, model_name: str, *, feature_ids: set[str],
                    expected_checkpoint_sha256: str) -> dict[str, dict[str, Path]]:
    """Load only this extraction manifest's checkpoint-matched cache entries."""
    import numpy as np
    entries = {}
    for metadata_path in sorted(cache_dir.glob("*.json")):
        if metadata_path.stem not in feature_ids:
            continue
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        identity = metadata.get("cache_identity", {})
        if (identity.get("model") != model_name or identity.get("checkpoint_sha256") != expected_checkpoint_sha256
                or identity.get("adapter_id") != ADAPTER_ID or metadata.get("feature_id") != metadata_path.stem):
            raise ValueError("feature manifest identity is invalid")
        array_path = cache_dir / f"{metadata_path.stem}.npz"
        if not array_path.is_file() or file_sha256(array_path) != metadata.get("feature_sha256"):
            raise ValueError("feature array is missing or has changed")
        with np.load(array_path, allow_pickle=False) as arrays:
            request_id = identity["renderer_request_id"]
            if request_id in entries:
                raise ValueError("feature manifest maps multiple caches to one renderer request")
            entries[request_id] = {key: arrays[key].astype(np.float32, copy=False) for key in arrays.files}
    if {path.stem for path in cache_dir.glob("*.json") if path.stem in feature_ids} != feature_ids:
        raise ValueError("feature manifest references a missing cache entry")
    return entries
