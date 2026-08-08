from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable

from scipy.io import wavfile

from .base import RenderRequest, Renderer
from .qa import audio_float32_sha256


@dataclass(frozen=True)
class BatchSummary:
    rendered: int
    skipped: int
    request_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {"rendered": self.rendered, "skipped": self.skipped, "request_ids": list(self.request_ids)}


def load_requests(path: Path | str) -> list[RenderRequest]:
    requests = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            requests.append(RenderRequest.from_dict(json.loads(line)))
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise ValueError(f"invalid render request at line {line_number}: {error}") from error
    return requests


def write_requests(path: Path | str, requests: Iterable[RenderRequest]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(request.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":")) for request in requests]
    destination.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8", newline="\n")


def _is_complete(wav_path: Path, result_path: Path, request_id: str) -> bool:
    if not wav_path.is_file() or not result_path.is_file():
        return False
    try:
        metadata = json.loads(result_path.read_text(encoding="utf-8"))
        _, audio = wavfile.read(wav_path)
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    return (
        metadata.get("provenance", {}).get("request_id") == request_id
        and metadata.get("qa", {}).get("audio_float32_sha256") == audio_float32_sha256(audio)
    )


def run_batch(renderer: Renderer, requests: Iterable[RenderRequest], output: Path | str, *, workers: int = 1,
              resume: bool = True) -> BatchSummary:
    if workers <= 0:
        raise ValueError("workers must be positive")
    if renderer.max_workers is not None and workers > renderer.max_workers:
        raise ValueError(f"renderer supports at most {renderer.max_workers} worker(s), got {workers}")
    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)
    ordered = list(requests)

    def run(request: RenderRequest) -> tuple[str, bool]:
        wav_path = output_path / f"{request.request_id}.wav"
        result_path = output_path / f"{request.request_id}.json"
        if resume and _is_complete(wav_path, result_path, request.request_id):
            return request.request_id, True
        result = renderer.render(request)
        result.write_wav(wav_path, force=True)
        result.write_json(result_path, force=True)
        return request.request_id, False

    if workers == 1:
        outcomes = [run(request) for request in ordered]
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            outcomes = list(executor.map(run, ordered))
    skipped = sum(1 for _, was_skipped in outcomes if was_skipped)
    return BatchSummary(len(outcomes) - skipped, skipped, tuple(request_id for request_id, _ in outcomes))
