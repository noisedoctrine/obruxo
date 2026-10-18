from copy import deepcopy
from dataclasses import dataclass
import json

import pytest

torch = pytest.importorskip("torch")

from obruxo_training.features import AudioFeatures  # noqa: E402
from obruxo_training.pilot import PilotPool, PilotSampler  # noqa: E402
from obruxo_training.stream import StreamConfig  # noqa: E402
from obruxo_training.trainer import TrainConfig, load_replay_seed, train_pilot  # noqa: E402
from test_stream import FakeRenderer, FakeRendererFactory  # noqa: E402


@dataclass(frozen=True)
class SelectiveFactory:
    rejected_request: str

    def __call__(self):
        return SelectiveRenderer(self.rejected_request)


class SelectiveRenderer(FakeRenderer):
    def __init__(self, rejected_request):
        self.rejected_request = rejected_request

    def render(self, request):
        self.mode = "clipped" if request.request_id == self.rejected_request else "valid"
        return super().render(request)


@pytest.fixture(scope="module")
def rejected_prefix(tmp_path_factory):
    root = tmp_path_factory.mktemp("rejected-prefix")
    pool = PilotPool.create(12, seed=2)
    sampler = PilotSampler(pool, seed=0)
    config = TrainConfig(steps=4, replay_capacity=3, batch_size=2, checkpoint_every=2, render_budget=8, cpu_threads=2)
    stream = StreamConfig(prefetch_clips=1, timeout_seconds=20)
    train_pilot(pool, root / "source", config=config, stream_config=stream,
                renderer_factory=SelectiveFactory(sampler.recipe_at(0).render_request().request_id), stop_after=0)
    source = torch.load(root / "source/checkpoint.pt", weights_only=True)
    assert source["cursor"]["next_sequence_id"] == 4
    assert [item["sequence_id"] for item in source["replay"]] == [1, 2, 3]
    frontend = AudioFeatures(source["basic_pitch"])
    return root, pool, sampler, config, stream, source, frontend


def test_rejected_prefix_seeds_fresh_learning_and_exact_recovery(rejected_prefix):
    root, pool, _, config, stream, source, _ = rejected_prefix

    def run(name, **kwargs):
        return train_pilot(pool, root / name, config=config, stream_config=stream,
                           renderer_factory=FakeRendererFactory("crash"), **kwargs)

    full = run("full", seed_replay=root / "source/checkpoint.pt")
    run("partial", seed_replay=root / "source/checkpoint.pt", stop_after=2)
    resumed = run("resumed", resume=root / "partial/checkpoint.pt")
    assert full["model_state_sha256"] == resumed["model_state_sha256"]
    assert full["final_metrics"] == resumed["final_metrics"]
    assert full["cursor"] == resumed["cursor"] == source["cursor"]
    assert full["replay_seed"]["history"]["rejected"] == 1
    assert full["replay_seed"]["acknowledged_recipes"] == 4
    assert full["seeded_feature_examples"] == 3
    assert full["totals"]["render_attempts"] == full["feature_seconds_this_segment"] == 0
    assert full["fresh_accepted_this_segment"] == 0
    saved = torch.load(root / "full/checkpoint.pt", weights_only=True)
    assert [item["sequence_id"] for item in saved["replay"]] == [1, 2, 3]
    for original, retained in zip(source["replay"], saved["replay"], strict=True):
        assert original["audio_sha256"] == retained["audio_sha256"]
        assert all(torch.equal(original["features"][key], retained["features"][key]) for key in original["features"])


@pytest.mark.parametrize("change", ["missing_log", "missing_attempt", "duplicate_attempt", "wrong_recipe", "wrong_status",
                                    "wrong_renderer", "wrong_audio", "wrong_counts", "evicted_feature", "heldout_state"])
def test_rejected_prefix_refuses_incomplete_or_changed_history(rejected_prefix, tmp_path, change):
    root, pool, sampler, config, stream, original, frontend = rejected_prefix
    source = deepcopy(original)
    history = [json.loads(line) for line in (root / "source/renders.jsonl").read_text().splitlines()]
    if change == "missing_attempt":
        history.pop(0)
    elif change == "duplicate_attempt":
        history[0] = history[1]
    elif change == "wrong_recipe":
        history[0]["recipe_id"] = "0" * 64
    elif change == "wrong_status":
        history[0]["status"] = "skipped"
    elif change == "wrong_renderer":
        history[0]["provenance"]["engine_fingerprint"] = "different"
    elif change == "wrong_audio":
        history[1]["qa"]["audio_float32_sha256"] = "0" * 64
    elif change == "wrong_counts":
        source["totals"]["rejected"] = 0
    elif change == "evicted_feature":
        source["replay"].pop(0)
    elif change == "heldout_state":
        source["replay"][0]["state_id"] = next(state for state, assignment in pool.manifest.by_state.items()
                                                 if assignment.split != "train")
    path = tmp_path / "checkpoint.pt"
    torch.save(source, path)
    if change != "missing_log":
        (tmp_path / "renders.jsonl").write_text("\n".join(json.dumps(row) for row in history) + "\n")
    with pytest.raises(ValueError):
        load_replay_seed(path, pool, sampler, frontend, config, stream)
