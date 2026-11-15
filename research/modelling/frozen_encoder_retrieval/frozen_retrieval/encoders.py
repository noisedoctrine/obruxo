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
ADAPTER_ID = "frozen_audio_adapters_v2"


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
    _resolve_local_configs(root, model_name, entry)
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
        "clap_music": {"numpy": "numpy", "scipy": "scipy", "torch": "torch", "laion_clap": "laion_clap",
                       "transformers": "transformers"},
        "muq_mulan": {"numpy": "numpy", "scipy": "scipy", "torch": "torch", "muq": "muq",
                       "torchaudio": "torchaudio", "librosa": "librosa", "transformers": "transformers",
                       "huggingface_hub": "huggingface_hub", "einops": "einops", "easydict": "easydict",
                       "x_clip": "x_clip"},
        "cir_ast": {"numpy": "numpy", "scipy": "scipy", "torch": "torch", "transformers": "transformers",
                     "librosa": "librosa"},
        "beats_iter3_plus": {"numpy": "numpy", "scipy": "scipy", "torch": "torch", "torchaudio": "torchaudio"},
        "encodec_24khz": {"numpy": "numpy", "scipy": "scipy", "torch": "torch", "transformers": "transformers"},
        "dac_44khz": {"numpy": "numpy", "scipy": "scipy", "torch": "torch", "transformers": "transformers"},
    }[model_name]
    return {package: {"module": module, "version": _version(package)} for package, module in modules.items()}


def _resolve_local_configs(root: Path, model_name: str, entry: dict) -> dict[str, Path]:
    requirements = {
        "muq_mulan": {"audio": ("OpenMuQ/MuQ-large-msd-iter", "OpenMuQ/MuQ-large-msd-iter", ("config.json",)),
                       "text": ("FacebookAI/xlm-roberta-base", "xlm-roberta-base", ("config.json",))},
        "cir_ast": {"ast": (None, None, ("config.json", "preprocessor_config.json"))},
    }.get(model_name, {})
    if not requirements:
        return {}
    records = entry.get("local_configs")
    if not isinstance(records, dict):
        raise ValueError(f"{model_name} requires SHA-locked local architecture/config assets")
    paths = {}
    for name, (expected_repository_id, expected_model_name, filenames) in requirements.items():
        record = records.get(name)
        if not isinstance(record, dict):
            raise ValueError(f"{model_name} checkpoint lock is missing local config bundle {name}")
        if expected_repository_id is not None and record.get("repository_id") != expected_repository_id:
            raise ValueError(f"{model_name} local config bundle {name} has the wrong source repository")
        if expected_model_name is not None and record.get("model_name") != expected_model_name:
            raise ValueError(f"{model_name} local config bundle {name} has the wrong model config name")
        if not isinstance(record.get("repository_id"), str) or not record["repository_id"].strip():
            raise ValueError(f"{model_name} local config bundle {name} must identify its source repository")
        if not isinstance(record.get("model_name"), str) or not record["model_name"].strip():
            raise ValueError(f"{model_name} local config bundle {name} must identify its architecture config")
        if not isinstance(record.get("revision"), str) or re.fullmatch(r"[0-9a-f]{40}", record["revision"]) is None:
            raise ValueError(f"{model_name} local config bundle {name} must pin a 40-character source revision")
        relative = record.get("path")
        if not isinstance(relative, str) or Path(relative).is_absolute() or ".." in Path(relative).parts:
            raise ValueError(f"{model_name} local config bundle {name} must stay under the checkpoint root")
        path = root / relative
        if path.is_symlink() or path.is_junction():
            raise ValueError(f"{model_name} local config bundle {name} cannot be a link")
        path = path.resolve(strict=True)
        if not path.is_relative_to(root) or not path.is_dir():
            raise ValueError(f"{model_name} local config bundle {name} must be a local directory")
        if any(not (path / filename).is_file() for filename in filenames):
            raise ValueError(f"{model_name} local config bundle {name} is missing required config files")
        if checkpoint_sha256(path) != record.get("sha256"):
            raise ValueError(f"{model_name} local config bundle {name} digest differs from the lock")
        paths[name] = path
    return paths


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


def _freeze_model(model) -> None:
    """Freeze a module or a loader wrapper exposing its module as ``.model``."""
    wrapped = getattr(model, "model", None)
    targets = [model] if wrapped is None or wrapped is model else [model, wrapped]
    for target in targets:
        evaluate = getattr(target, "eval", None)
        if callable(evaluate):
            evaluate()
    parameters = next((target.parameters() for target in targets if callable(getattr(target, "parameters", None))), None)
    if parameters is None:
        raise TypeError("encoder loader did not expose a torch module with parameters")
    for parameter in parameters:
        parameter.requires_grad_(False)
    if any(parameter.requires_grad for parameter in getattr(wrapped or model, "parameters")()):
        raise RuntimeError("frozen encoder has trainable parameters")


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
                fbanks = model.preprocess(signal)
                frames = _beats_sequence(features, int(model.input_patch_size), fbanks.shape[1]).detach().float().cpu().numpy()
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


def _beats_sequence(features, patch_size: int, valid_fbank_frames: int):
    if patch_size <= 0 or 128 % patch_size:
        raise ValueError("BEATs input patch size must divide the 128-bin mel axis")
    frequency_patches = 128 // patch_size
    if features.ndim != 3 or features.shape[0] != 1 or features.shape[-1] != 768:
        raise ValueError("BEATs features must have shape [1,time_patches*frequency_patches,768]")
    if features.shape[1] % frequency_patches:
        raise ValueError("BEATs feature sequence does not contain complete time-frequency grids")
    time_patches = features.shape[1] // frequency_patches
    valid_time = min(time_patches, valid_fbank_frames // patch_size)
    # BEATs flattens [B,C,time_patches,frequency_patches], so frequency is the fastest patch axis.
    return features.reshape(1, time_patches, frequency_patches, 768).mean(dim=2)[0, :valid_time]


def _safe_torch_load(torch, path: Path):
    """Load tensor-only checkpoint state without allowing arbitrary pickle globals."""
    return torch.load(path, map_location="cpu", weights_only=True)


def _make_muq_audio_initializer(MuQ, MuQConfig, audio_config: dict):
    def init_audio(instance, model_name):
        # Upstream AudioSpectrogramTransformerPretrained invokes this method for its side effect.
        if model_name != "OpenMuQ/MuQ-large-msd-iter":
            raise ValueError("MuQ-MuLan requested an unexpected audio architecture")
        instance.model = MuQ(MuQConfig(**audio_config))

    return init_audio


def _load_muq_mulan_offline(checkpoint: Path, config_paths: dict[str, Path], torch, device: str):
    from muq.muq.muq import MuQ, MuQConfig
    from muq.muq_mulan.models.audio import AudioSpectrogramTransformerPretrained
    from muq.muq_mulan.models.text import TextTransformerPretrained
    from muq.muq_mulan.muq_mulan import MuQMuLan
    from transformers import XLMRobertaConfig, XLMRobertaModel

    config = json.loads((checkpoint / "config.json").read_text(encoding="utf-8"))
    if config.get("audio_model", {}).get("name") != "OpenMuQ/MuQ-large-msd-iter":
        raise ValueError("MuQ-MuLan audio backbone differs from the expected official checkpoint config")
    if config.get("text_model", {}).get("name") not in {"xlm-roberta-base", "FacebookAI/xlm-roberta-base"}:
        raise ValueError("MuQ-MuLan text architecture differs from the expected official checkpoint config")
    audio_config = json.loads((config_paths["audio"] / "config.json").read_text(encoding="utf-8"))
    audio_init, text_init = AudioSpectrogramTransformerPretrained._init_pretrained_model, TextTransformerPretrained._init_pretrained_model

    init_audio = _make_muq_audio_initializer(MuQ, MuQConfig, audio_config)

    def init_text(_instance):
        # Text is not used by this audio-only benchmark; its architecture config is enough to load the full state dict.
        local_config = XLMRobertaConfig.from_pretrained(str(config_paths["text"]), local_files_only=True)
        return XLMRobertaModel(local_config)

    AudioSpectrogramTransformerPretrained._init_pretrained_model = init_audio
    TextTransformerPretrained._init_pretrained_model = init_text
    try:
        model = MuQMuLan(config)
    finally:
        AudioSpectrogramTransformerPretrained._init_pretrained_model = audio_init
        TextTransformerPretrained._init_pretrained_model = text_init
    state = _safe_torch_load(torch, checkpoint / "pytorch_model.bin")
    model.load_state_dict(state, strict=True)
    if model.sr != MODEL_REGISTRY["muq_mulan"]["sample_rate"]:
        raise ValueError("MuQ-MuLan checkpoint sample rate differs from the frozen adapter contract")
    return model.to(device)


def _load_cir_ast_offline(checkpoint: Path, config_path: Path, lock_entry: dict, torch):
    from AST import ASTFineTuned
    from transformers import ASTConfig, ASTFeatureExtractor

    state = _safe_torch_load(torch, checkpoint)
    custom_config = state.get("custom_config") if isinstance(state, dict) else None
    model_config = custom_config.get("model", {}) if isinstance(custom_config, dict) else {}
    local_config = lock_entry["local_configs"]["ast"]
    if model_config.get("pretrained") != local_config.get("model_name"):
        raise ValueError("CIR checkpoint base architecture does not match its pinned local config bundle")
    config = ASTConfig.from_pretrained(str(config_path), local_files_only=True)
    model = ASTFineTuned(config)
    model.custom_config = custom_config
    model.feature_extractor = ASTFeatureExtractor.from_pretrained(str(config_path), local_files_only=True)
    if model.feature_extractor.sampling_rate != MODEL_REGISTRY["cir_ast"]["sample_rate"]:
        raise ValueError("CIR feature-extractor sample rate differs from the frozen adapter contract")
    # The embedding path returns normalized AST CLS/distillation tokens before the classifier head.
    model.classifier = torch.nn.Identity()
    weights = state.get("model_state_dict")
    if not isinstance(weights, dict):
        raise ValueError("CIR checkpoint is missing its model tensor state")
    weights = {name: value for name, value in weights.items() if not name.startswith("classifier.")}
    model.load_state_dict(weights, strict=True)
    return model


def _clap_roberta_config(config_class):
    """Rebuild the exact roberta-base tensor dimensions embedded in the LAION-CLAP checkpoint."""
    return config_class(max_position_embeddings=514, type_vocab_size=1)


def _load_clap_music_offline(checkpoint: Path, device: str):
    """Build CLAP's text-side scaffolding from config only; the audio-only task needs no HF fetch."""
    from unittest.mock import patch
    from laion_clap import CLAP_Module
    from transformers import RobertaConfig, RobertaModel, RobertaTokenizer

    def init_roberta(cls, model_name, *args, **kwargs):
        if model_name != "roberta-base":
            raise ValueError("LAION-CLAP requested an unexpected text architecture")
        return cls(_clap_roberta_config(RobertaConfig))

    def no_tokenizer(cls, model_name, *args, **kwargs):
        if model_name != "roberta-base":
            raise ValueError("LAION-CLAP requested an unexpected tokenizer")
        return None

    with patch.object(RobertaModel, "from_pretrained", classmethod(init_roberta)), \
            patch.object(RobertaTokenizer, "from_pretrained", classmethod(no_tokenizer)):
        model = CLAP_Module(enable_fusion=False, amodel="HTSAT-base", tmodel="roberta", device=device)
    model.load_ckpt(str(checkpoint), verbose=False)
    return model


def load_adapter(model_name: str, checkpoint: Path, lock_entry: dict, checkpoint_digest: str, *,
                 checkpoint_root: Path, device: str = "cpu", source_root: Path | None = None) -> FrozenAdapter:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    import numpy as np
    import torch
    local_configs = _resolve_local_configs(checkpoint_root.resolve(strict=True), model_name, lock_entry)

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
        model = _load_clap_music_offline(checkpoint, device)
    elif model_name == "muq_mulan":
        model = _load_muq_mulan_offline(checkpoint, local_configs, torch, device)
    elif model_name == "cir_ast":
        if source_root is None:
            raise ValueError("CIR requires a local checkout pinned to the checkpoint's source revision")
        source_root = source_root.resolve(strict=True)
        if str(source_root) not in sys.path:
            sys.path.insert(0, str(source_root))
        model = _load_cir_ast_offline(checkpoint, local_configs["ast"], lock_entry, torch).to(device)
    elif model_name == "beats_iter3_plus":
        if source_root is None:
            raise ValueError("BEATs requires a local checkout of Microsoft's official loader source")
        source_root = source_root.resolve(strict=True)
        if str(source_root) not in sys.path:
            sys.path.insert(0, str(source_root))
        from BEATs import BEATs, BEATsConfig
        state = _safe_torch_load(torch, checkpoint)
        cfg = BEATsConfig(state["cfg"])
        model = BEATs(cfg)
        model.load_state_dict(state["model"])
        model.to(device)
    elif model_name in {"encodec_24khz", "dac_44khz"}:
        from transformers import AutoModel
        model = AutoModel.from_pretrained(str(checkpoint), local_files_only=True).to(device)
    else:
        raise ValueError(f"unknown model: {model_name}")
    _freeze_model(model)
    return FrozenAdapter(model_name, checkpoint, lock_entry, checkpoint_digest, device, model, torch, np)


def cache_feature(adapter: FrozenAdapter, wav_path: Path, *, request_id: str, render_metadata: dict,
                  output: Path, source_precision: str = "fp32") -> dict:
    audio_hash = file_sha256(wav_path)
    cache_identity = {"adapter_id": ADAPTER_ID, "model": adapter.name,
                      "checkpoint_sha256": adapter.checkpoint_digest, "source_revision": adapter.lock_entry["revision"],
                      "loader_revision": adapter.lock_entry["loader_revision"], "audio_file_sha256": audio_hash,
                      "local_config_pins": {name: {key: record[key] for key in ("repository_id", "revision", "sha256")}
                                             for name, record in adapter.lock_entry.get("local_configs", {}).items()},
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
                     "license": (entry or {}).get("license", spec["license"]),
                     "license_status": (entry or {}).get("license_status", "unverified"),
                     "dependencies": dependencies})
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
