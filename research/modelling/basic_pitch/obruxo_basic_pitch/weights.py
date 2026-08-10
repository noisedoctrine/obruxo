"""Strict importer for Spotify's pinned ICASSP 2022 ONNX checkpoint."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import torch
from onnx import numpy_helper

from .constants import (
    FRONTEND_BATCHNORM_EPS,
    MODEL_ID,
    SPOTIFY_ONNX_GIT_BLOB_SHA1,
    SPOTIFY_ONNX_PATH,
    SPOTIFY_ONNX_SIZE_BYTES,
    SPOTIFY_REPOSITORY,
    SPOTIFY_REVISION,
)
from .model import BasicPitchICASSP2022

SOURCE_REVISION = SPOTIFY_REVISION
SOURCE_PATH = SPOTIFY_ONNX_PATH
SOURCE_GIT_BLOB_SHA1 = SPOTIFY_ONNX_GIT_BLOB_SHA1
SOURCE_SIZE_BYTES = SPOTIFY_ONNX_SIZE_BYTES

_SOURCE_NAMES = {
    "cqt_real": "const_fold_opt__655",
    "cqt_imag": "const_fold_opt__664",
    "cqt_lengths": "model_1/cq_t2010v2_1/Sqrt;model_1/cq_t2010v2_1/Sqrt",
    "lowpass": "const_fold_opt__734",
    "frontend_bn_weight": "model_1/batch_normalization/FusedBatchNormV3;model_1/batch_normalization/FusedBatchNormV3",
    "frontend_bn_bias": "model_1/batch_normalization/FusedBatchNormV3;model_1/batch_normalization/FusedBatchNormV31",
    "contour_conv1": "const_fold_opt__727",
    "contour_conv1_bias": "model_1/re_lu_1/Relu;model_1/re_lu_1/Relu;model_1/batch_normalization_2/FusedBatchNormV3;model_1/batch_normalization_2/FusedBatchNormV3;model_1/conv2d_1/BiasAdd/ReadVariableOp;model_1/conv2d_1/BiasAdd/ReadVariableOp;model_1/conv2d_1/BiasAdd;model_1/conv2d_1/BiasAdd;model_1/conv2d_1/Conv2D;model_1/conv2d_1/Conv2D",
    "contour_conv2": "const_fold_opt__710",
    "contour_conv2_bias": "model_1/contours-reduced/BiasAdd/ReadVariableOp;model_1/contours-reduced/BiasAdd/ReadVariableOp",
    "note_conv1": "const_fold_opt__738",
    "note_conv1_bias": "model_1/conv2d_2/BiasAdd/ReadVariableOp;model_1/conv2d_2/BiasAdd/ReadVariableOp",
    "note_conv2": "const_fold_opt__702",
    "note_conv2_bias": "model_1/conv2d_3/BiasAdd/ReadVariableOp;model_1/conv2d_3/BiasAdd/ReadVariableOp",
    "onset_conv1": "const_fold_opt__707",
    "onset_conv1_bias": "model_1/re_lu_3/Relu;model_1/re_lu_3/Relu;model_1/batch_normalization_3/FusedBatchNormV3;model_1/batch_normalization_3/FusedBatchNormV3;model_1/conv2d_4/BiasAdd/ReadVariableOp;model_1/conv2d_4/BiasAdd/ReadVariableOp;model_1/conv2d_4/BiasAdd;model_1/conv2d_4/BiasAdd;model_1/conv2d_2/Conv2D;model_1/conv2d_2/Conv2D;model_1/conv2d_4/Conv2D;model_1/conv2d_4/Conv2D",
    "onset_conv2": "const_fold_opt__680",
    "onset_conv2_bias": "model_1/conv2d_5/BiasAdd/ReadVariableOp;model_1/conv2d_5/BiasAdd/ReadVariableOp",
}

_ALLOWED_UNMAPPED_FLOATS = {
    "model_1/normalized_log_1/truediv;model_1/normalized_log_1/truediv;model_1/normalized_log_1/Log_1;model_1/normalized_log_1/Log_1",
    "model_1/normalized_log_1/div_no_nan;model_1/normalized_log_1/div_no_nan",
    "model_1/normalized_log_1/add/y;model_1/normalized_log_1/add/y",
    "model_1/normalized_log_1/Const;model_1/normalized_log_1/Const",
    "model_1/cq_t2010v2_1/conv1d_25;model_1/cq_t2010v2_1/conv1d_25",
    "model_1/conv2d_5/Conv2D;model_1/conv2d_5/Conv2D",
}


@dataclass(frozen=True)
class ConversionMetadata:
    model_id: str
    source_revision: str
    source_path: str
    source_git_blob_sha1: str
    source_sha256: str
    source_size_bytes: int
    torch_version: str
    onnx_version: str


def _git_blob_sha1(content: bytes) -> str:
    header = f"blob {len(content)}\0".encode("ascii")
    return hashlib.sha1(header + content).hexdigest()


def _source_metadata(content: bytes) -> ConversionMetadata:
    return ConversionMetadata(
        model_id=MODEL_ID,
        source_revision=SOURCE_REVISION,
        source_path=SOURCE_PATH,
        source_git_blob_sha1=SOURCE_GIT_BLOB_SHA1,
        source_sha256=hashlib.sha256(content).hexdigest(),
        source_size_bytes=len(content),
        torch_version=torch.__version__,
        onnx_version=onnx.__version__,
    )


def _reachable_initializers(model: onnx.ModelProto) -> set[str]:
    producers = {output: node for node in model.graph.node for output in node.output}
    initializer_names = {initializer.name for initializer in model.graph.initializer}
    pending = [output.name for output in model.graph.output]
    visited_nodes: set[int] = set()
    reachable: set[str] = set()
    node_ids = {id(node): index for index, node in enumerate(model.graph.node)}
    while pending:
        value = pending.pop()
        if value in initializer_names:
            reachable.add(value)
            continue
        node = producers.get(value)
        if node is None:
            continue
        node_id = node_ids[id(node)]
        if node_id in visited_nodes:
            continue
        visited_nodes.add(node_id)
        pending.extend(node.input)
    return reachable


def _check_shape(value: onnx.ValueInfoProto, expected: tuple[int, ...]) -> None:
    dims = value.type.tensor_type.shape.dim
    if len(dims) != len(expected):
        raise ValueError(f"unexpected rank for {value.name}: {len(dims)}")
    for actual, required in zip(dims[1:], expected[1:]):
        if actual.dim_value != required:
            raise ValueError(f"unexpected shape for {value.name}: {tuple(d.dim_value for d in dims)}")


def _verify_graph(model: onnx.ModelProto) -> set[str]:
    if len(model.graph.input) != 1 or model.graph.input[0].name != "serving_default_input_2:0":
        raise ValueError("unexpected Basic Pitch ONNX input signature")
    input_value = model.graph.input[0]
    _check_shape(input_value, (0, 43_844, 1))
    expected_outputs = {
        "StatefulPartitionedCall:0": (0, 172, 264),
        "StatefulPartitionedCall:1": (0, 172, 88),
        "StatefulPartitionedCall:2": (0, 172, 88),
    }
    output_names = {output.name for output in model.graph.output}
    if output_names != set(expected_outputs):
        raise ValueError(f"unexpected Basic Pitch ONNX outputs: {sorted(output_names)}")
    for output in model.graph.output:
        _check_shape(output, expected_outputs[output.name])
    if not model.opset_import or model.opset_import[0].version != 15:
        raise ValueError("unexpected ONNX opset")

    reachable = _reachable_initializers(model)
    required = set(_SOURCE_NAMES.values())
    missing = required - reachable
    if missing:
        raise ValueError(f"required source tensors are not reachable: {sorted(missing)}")
    float_names = {
        initializer.name
        for initializer in model.graph.initializer
        if numpy_helper.to_array(initializer).dtype.kind == "f" and initializer.name in reachable
    }
    unmapped = float_names - required - _ALLOWED_UNMAPPED_FLOATS
    if unmapped:
        raise ValueError(f"unexpected reachable floating-point tensors: {sorted(unmapped)}")
    return reachable


def _load_verified(path: Path) -> tuple[onnx.ModelProto, ConversionMetadata]:
    source_path = path.resolve(strict=True)
    content = source_path.read_bytes()
    if len(content) != SOURCE_SIZE_BYTES:
        raise ValueError(f"unexpected ONNX size: {len(content)}")
    if _git_blob_sha1(content) != SOURCE_GIT_BLOB_SHA1:
        raise ValueError("ONNX Git blob SHA-1 does not match the pinned Spotify artifact")
    model = onnx.load_model(source_path, load_external_data=False)
    onnx.checker.check_model(model)
    _verify_graph(model)
    return model, _source_metadata(content)


def _source_arrays(model: onnx.ModelProto) -> dict[str, np.ndarray]:
    return {initializer.name: np.array(numpy_helper.to_array(initializer), copy=True) for initializer in model.graph.initializer}


def _identity_batchnorm(state: dict[str, torch.Tensor], prefix: str, channels: int, eps: float) -> None:
    state[f"{prefix}.weight"] = torch.ones(channels, dtype=torch.float32)
    state[f"{prefix}.bias"] = torch.zeros(channels, dtype=torch.float32)
    state[f"{prefix}.running_mean"] = torch.zeros(channels, dtype=torch.float32)
    state[f"{prefix}.running_var"] = torch.full((channels,), 1.0 - eps, dtype=torch.float32)
    state[f"{prefix}.num_batches_tracked"] = torch.zeros((), dtype=torch.int64)


def _tensor(arrays: dict[str, np.ndarray], name: str, target_shape: tuple[int, ...], *, squeeze: tuple[int, ...] = ()) -> torch.Tensor:
    value = arrays[_SOURCE_NAMES[name]]
    for axis in sorted(squeeze, reverse=True):
        if value.shape[axis] != 1:
            raise ValueError(f"cannot squeeze source tensor {name} at axis {axis}: {value.shape}")
        value = np.squeeze(value, axis=axis)
    if tuple(value.shape) != target_shape:
        raise ValueError(f"unexpected shape for {name}: {value.shape}, expected {target_shape}")
    if value.dtype != np.float32:
        raise TypeError(f"unexpected dtype for {name}: {value.dtype}")
    return torch.from_numpy(value.copy())


def import_onnx_state_dict(onnx_path: str | Path) -> tuple[dict[str, torch.Tensor], ConversionMetadata]:
    """Verify and import the pinned ONNX graph into the native module state."""
    model_proto, metadata = _load_verified(Path(onnx_path))
    arrays = _source_arrays(model_proto)
    model = BasicPitchICASSP2022()
    state = {key: value.detach().clone() for key, value in model.state_dict().items()}

    state["frontend.cqt_kernels_real"] = _tensor(arrays, "cqt_real", (36, 1, 256), squeeze=(2,))
    state["frontend.cqt_kernels_imag"] = _tensor(arrays, "cqt_imag", (36, 1, 256), squeeze=(2,))
    state["frontend.lowpass_filter"] = _tensor(arrays, "lowpass", (1, 1, 256), squeeze=(2,))
    state["frontend.cqt_lengths"] = _tensor(arrays, "cqt_lengths", (309,), squeeze=(1, 2))
    state["frontend.normalization.weight"] = _tensor(arrays, "frontend_bn_weight", (1,))
    state["frontend.normalization.bias"] = _tensor(arrays, "frontend_bn_bias", (1,))
    _identity_batchnorm(state, "contour_bn", 8, 1e-3)
    _identity_batchnorm(state, "onset_bn", 32, 1e-3)
    state["frontend.normalization.running_mean"] = torch.zeros(1, dtype=torch.float32)
    state["frontend.normalization.running_var"] = torch.full((1,), 1.0 - FRONTEND_BATCHNORM_EPS, dtype=torch.float32)
    state["frontend.normalization.num_batches_tracked"] = torch.zeros((), dtype=torch.int64)

    direct = {
        "contour_conv1": ("contour_conv1.weight", (8, 8, 3, 39)),
        "contour_conv1_bias": ("contour_conv1.bias", (8,)),
        "contour_conv2": ("contour_conv2.weight", (1, 8, 5, 5)),
        "contour_conv2_bias": ("contour_conv2.bias", (1,)),
        "note_conv1": ("note_conv1.weight", (32, 1, 7, 7)),
        "note_conv1_bias": ("note_conv1.bias", (32,)),
        "note_conv2": ("note_conv2.weight", (1, 32, 7, 3)),
        "note_conv2_bias": ("note_conv2.bias", (1,)),
        "onset_conv1": ("onset_conv1.weight", (32, 8, 5, 5)),
        "onset_conv1_bias": ("onset_conv1.bias", (32,)),
        "onset_conv2": ("onset_conv2.weight", (1, 33, 3, 3)),
        "onset_conv2_bias": ("onset_conv2.bias", (1,)),
    }
    for source_name, (target_name, shape) in direct.items():
        state[target_name] = _tensor(arrays, source_name, shape)

    if set(state) != set(model.state_dict()):
        raise ValueError("importer produced an incomplete or unexpected native state dict")
    model.load_state_dict(state, strict=True)
    model.eval()
    with torch.inference_mode():
        model(torch.zeros(1, 43_844, 1, dtype=torch.float32))
    return state, metadata


def _approved_destination(path: Path, source_path: Path) -> Path:
    root = Path(__file__).resolve().parents[4]
    workspace = root / "research" / "modelling" / "basic_pitch"
    approved = [workspace / "artifacts", workspace / "reports", workspace / "outputs"]
    destination = path.resolve(strict=False)
    source_directory = source_path.resolve(strict=True).parent
    if destination == source_path.resolve(strict=True) or destination.is_relative_to(source_directory):
        raise ValueError("refusing to write inside the ONNX input directory")
    if not any(destination.is_relative_to(directory) for directory in approved):
        raise ValueError("destination is outside the approved Basic Pitch artifact/report/output areas")
    return destination


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, prefix=".metadata-", suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _atomic_checkpoint(path: Path, state: dict[str, torch.Tensor]) -> None:
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".checkpoint-", suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
        torch.save(state, temporary)
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def write_imported_checkpoint(
    onnx_path: str | Path,
    checkpoint_path: str | Path,
    metadata_path: str | Path,
    *,
    force: bool = False,
) -> ConversionMetadata:
    """Import, smoke-test, and atomically write the sole authorized public checkpoint."""
    source = Path(onnx_path)
    checkpoint = _approved_destination(Path(checkpoint_path), source)
    metadata_file = _approved_destination(Path(metadata_path), source)
    if checkpoint.parent != metadata_file.parent:
        raise ValueError("checkpoint and metadata must share an approved destination directory")
    if not force and (checkpoint.exists() or metadata_file.exists()):
        raise FileExistsError("refusing to overwrite an existing checkpoint or metadata file without force=True")
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    state, metadata = import_onnx_state_dict(source)
    _atomic_checkpoint(checkpoint, state)
    _atomic_json(
        metadata_file,
        {
            "format_version": 1,
            "model_id": metadata.model_id,
            "source": {
                "repository": SPOTIFY_REPOSITORY,
                "revision": metadata.source_revision,
                "path": metadata.source_path,
                "git_blob_sha1": metadata.source_git_blob_sha1,
                "sha256": metadata.source_sha256,
                "size_bytes": metadata.source_size_bytes,
            },
            "conversion": {
                "torch_version": metadata.torch_version,
                "onnx_version": metadata.onnx_version,
            },
        },
    )
    return metadata
