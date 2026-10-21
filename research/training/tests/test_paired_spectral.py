import copy

import pytest

torch = pytest.importorskip("torch")

from obruxo_training.paired_spectral import complete_pairs, paired_change_loss, require_aligned_pairs  # noqa: E402
from obruxo_training.pilot import PilotPool  # noqa: E402
from obruxo_training.spectral_controls import BasicControlAdapter  # noqa: E402
from obruxo_training.temporal_proxy import collate_frames  # noqa: E402
from obruxo_training.temporal_proxy_data import RecordCache, TemporalProxyConfig, collect_frames, make_plan  # noqa: E402
from obruxo_training.temporal_proxy_screen import fit_temporal_proxy  # noqa: E402
from test_temporal_proxy import records  # noqa: E402


@pytest.fixture(scope="module")
def pool():
    return PilotPool.create(80, seed=0)


def test_signed_change_loss_detects_reversed_effect_and_backpropagates():
    target, _ = collate_frames(records())
    for value in (*target["magnitudes"], target["envelope"]):
        value[1] *= 1.5
    assert paired_change_loss(target, target).item() == 0
    prediction = {"magnitudes": tuple(value.flip(0).clone().requires_grad_() for value in target["magnitudes"]),
                  "envelope": target["envelope"].flip(0).clone().requires_grad_()}
    loss = paired_change_loss(prediction, target).mean()
    assert torch.isfinite(loss) and loss > 0
    loss.backward()
    assert all(torch.isfinite(value.grad).all() and value.grad.abs().sum() > 0
               for value in (*prediction["magnitudes"], prediction["envelope"]))
    assert all(value.grad is None for value in (*target["magnitudes"], target["envelope"]))


def test_pairs_require_same_times_performance_and_ancestry():
    rows = records()
    for row in rows:
        row.update(pair_id=0, source_state_id="same")
    require_aligned_pairs(rows)
    rows[1]["times"][0][0, 0] += .01
    with pytest.raises(ValueError, match="absolute frame times"):
        require_aligned_pairs(rows)


def test_paired_plan_changes_one_active_control_at_a_time(pool):
    config = TemporalProxyConfig(training_states=6, steps=1, validation_states=2, paired_changes=True)
    plan, recipes, _ = make_plan(pool, config)
    assert plan["paired_objective"]["weight"] == .25 and "vita.py" in plan["source"]
    for offset in range(0, len(recipes), 8):
        for view in range(4):
            first, second = recipes[offset + view], recipes[offset + view + 4]
            assert first["pair_id"] == second["pair_id"] and first["performance"] == second["performance"]
            if view < 2:
                assert sum(a != b for a, b in zip(first["target"]["continuous"], second["target"]["continuous"], strict=True)) == 1
                assert first["target"]["categorical"] == second["target"]["categorical"]
    with pytest.raises(ValueError, match="eight views"):
        TemporalProxyConfig(views=4, paired_changes=True)


def test_pair_training_resume_reproduces_selections_and_updates():
    torch.set_num_threads(2)
    class Cache(list):
        pass
    base = records()[0]
    cache = Cache(copy.deepcopy(base) for _ in range(48))
    cache.rows = [{"sequence": i} for i in range(48)]
    for index, row in enumerate(cache):
        row.update(pair_id=index // 8 * 4 + index % 4, source_state_id=str(index // 8))
    assert len(complete_pairs(cache)) == 24
    config = TemporalProxyConfig(training_states=6, steps=6, batch_size=2, checkpoint_every=2, paired_changes=True)
    adapter, contract = BasicControlAdapter(), {"mode": "paired_fixture"}
    _, reference = fit_temporal_proxy(cache, adapter, config, contract)
    saved = {}
    def checkpoint(value):
        saved.clear()
        saved.update(copy.deepcopy(value))
    def interrupt(step):
        if step == 3:
            raise InterruptedError
    with pytest.raises(InterruptedError):
        fit_temporal_proxy(cache, adapter, config, contract, checkpoint=checkpoint, after_step=interrupt)
    _, recovered = fit_temporal_proxy(cache, adapter, config, contract, saved=saved)
    assert recovered["model_sha256"] == reference["model_sha256"] and recovered["trace"] == reference["trace"]
    assert torch.equal(recovered["batch_rng"], reference["batch_rng"])
    cache.rows = cache.rows[:12]
    with pytest.raises(ValueError, match="too few complete"):
        fit_temporal_proxy(cache, adapter, config, contract)


def test_collected_pairs_keep_exact_same_sample_times(pool, tmp_path):
    torch.set_num_threads(2)
    config = TemporalProxyConfig(training_states=6, steps=1, validation_states=2, paired_changes=True)
    plan, recipes, _ = make_plan(pool, config)
    wave = (.01 * torch.sin(torch.arange(65536) / 10)).numpy()
    import numpy as np
    class Session:
        def render(self, target, performance, phase, label):
            return np.stack((wave, wave), axis=1), {"status": "accepted"}
    session = Session()
    session.plan = plan
    rows = collect_frames(pool, tmp_path, recipes, session)
    cache = RecordCache(tmp_path, rows)
    pairs = complete_pairs(cache)
    assert len(pairs) == 24
    for first, second in pairs:
        require_aligned_pairs([cache[first], cache[second]])
