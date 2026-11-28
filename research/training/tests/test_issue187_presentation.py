import ast
from datetime import datetime, timezone
import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "reports/scripts/show_issue187_progress.py"
SPEC = importlib.util.spec_from_file_location("show_issue187_progress", SCRIPT)
MONITOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MONITOR)
RUNNER = SCRIPT.with_name("issue187_surrogate_adapters.py")
CLI = SCRIPT.with_name("run_issue187_surrogates.py")


def test_stage2_configurations_use_adapter_then_frozen_model_names():
    expected = {
        "latent-residual_mlp": "Residual MLP + SAME-L",
        "latent-transformer": "Transformer + SAME-L",
        "generator-residual_mlp": "Residual MLP + Stable Audio 3 Medium",
        "generator-transformer": "Transformer + Stable Audio 3 Medium",
    }
    assert MONITOR.CONFIGURATION_NAMES == expected

    def nodes(tree):
        for node in tree:
            yield node
            yield from nodes(node.get("children", []))

    tree = list(nodes(MONITOR.plan_tree()))
    assert [node["title"] for node in tree if node["id"].endswith("-arm")] == list(expected.values())
    assert MONITOR.display_text("Stage 2 · latent-transformer updates") == (
        "Stage 2 · Transformer + SAME-L updates")
    mlp = next(node for node in tree if node["id"] == "latent-residual_mlp-arm")
    subtasks = {node["id"]: node for node in mlp["children"]}
    assert subtasks["latent-residual_mlp-load"]["artifact"] == "arm-latent-residual_mlp.json"
    assert subtasks["latent-residual_mlp-val-0"]["artifact"] == "validation-latent-residual_mlp-0.json"
    assert subtasks["latent-residual_mlp-val-500"]["artifact"] == "validation-latent-residual_mlp-500.json"
    assert subtasks["latent-residual_mlp-updates"]["training_report"] == "arm-latent-residual_mlp.json"


def test_monitor_distinguishes_capped_training_from_an_active_arm():
    capped = {"updates": 5_483, "stopped": "the arm reached its five-hour cap during an update"}
    active = {"updates": 1_200, "stopped": None}
    complete = {"updates": 5_500, "stopped": None}
    assert MONITOR.training_status(capped, active=False) == "Stopped"
    assert MONITOR.training_status(active, active=True) == "In progress"
    assert MONITOR.training_status(complete, active=False) == "Complete"
    assert MONITOR.training_status({"updates": 5_800, "stopped": None}, active=False) == "Checkpoint"


def test_monitor_keeps_elapsed_history_from_before_a_worker_restart(tmp_path):
    output = tmp_path / "run"
    output.mkdir()
    (output / "events.log").write_text(
        "2026-11-28T09:00:00+00:00 Stage 2 · arm updates: 0/100 (running)\n"
        "2026-11-28T09:00:12+00:00 Stage 2 · arm updates: 100/100 (complete)\n"
        "2026-11-28T09:10:00+00:00 Stage 1 · Verify input receipts: 0/1 (running)\n"
        "2026-11-28T09:10:01+00:00 Stage 1 · Verify input receipts: 1/1 (running)\n",
        encoding="utf-8",
    )
    records = MONITOR.read_timeline(output, datetime(2026, 11, 28, 9, 11, tzinfo=timezone.utc))
    assert records["Stage 2 · arm updates"]["seconds"] == 12


def test_monitor_hides_curve_points_above_the_current_update_budget(tmp_path):
    (tmp_path / "arm-latent-residual_mlp.json").write_text(
        '{"logs":[{"update":5500,"loss":0.2},{"update":5800,"loss":0.1}]}', encoding="utf-8")
    for step, value in ((5500, 0.2), (8000, 0.1)):
        (tmp_path / f"validation-latent-residual_mlp-{step}.json").write_text(
            f'{{"metrics":{{"mel_log1p":{value},"spectral_loss":{value}}}}}', encoding="utf-8")
    curves = MONITOR.read_learning_curves(tmp_path)
    assert curves["training"]["latent-residual_mlp"] == [(5500, 0.2)]
    assert curves["validation"]["latent-residual_mlp"] == [(5500, 0.2)]


def test_formal_stage2_runs_to_its_update_budget_without_wall_clock_cutoff():
    tree = ast.parse(RUNNER.read_text(encoding="utf-8"))
    def names(target):
        if isinstance(target, ast.Name):
            yield target.id
        elif isinstance(target, (ast.Tuple, ast.List)):
            for child in target.elts:
                yield from names(child)

    cap = next(node.value for node in ast.walk(tree) if isinstance(node, ast.Assign)
               and "CAP_SECONDS" in {name for target in node.targets for name in names(target)})
    assert isinstance(cap, ast.Tuple) and isinstance(cap.elts[0], ast.Attribute) and cap.elts[0].attr == "inf"
    maximum = next(node.value for node in ast.walk(tree) if isinstance(node, ast.Assign)
                   and "MAX_UPDATES" in {name for target in node.targets for name in names(target)})
    assert isinstance(maximum, ast.Tuple) and maximum.elts[1].value == 5_500
    cli = ast.parse(CLI.read_text(encoding="utf-8"))
    target_argument = next(node for node in ast.walk(cli) if isinstance(node, ast.Call)
                           and isinstance(node.func, ast.Attribute) and node.func.attr == "add_argument"
                           and node.args and isinstance(node.args[0], ast.Constant)
                           and node.args[0].value == "--target-step")
    choices = next(keyword.value for keyword in target_argument.keywords if keyword.arg == "choices")
    assert tuple(value.value for value in choices.elts) == (500, 5500)


def test_stage2_finishes_stable_audio_3_medium_arms_before_same_l():
    tree = ast.parse(RUNNER.read_text(encoding="utf-8"))
    modes = next(node.value for node in ast.walk(tree) if isinstance(node, ast.Assign)
                 and any(isinstance(target, ast.Name) and target.id == "available" for target in node.targets))
    assert isinstance(modes, ast.IfExp)
    assert tuple(element.value for element in modes.body.elts) == ("generator", "latent")
    assert MONITOR.MILESTONES[-1] == 5_500
    assert MONITOR.plan_tree()[1]["children"][2]["children"][1]["title"] == "Train adapter (up to 5,500 updates)"


def test_report_defines_embedded_decoder_and_medium_trunk_once():
    report = SCRIPT.parents[1] / "issue187_pretrained_vital_surrogates" / "STAGE2_PREPARATION.md"
    content = report.read_text(encoding="utf-8")
    assert content.count("SAME-L names the embedded autoencoder's decoder") == 1
    assert content.count("Stable Audio 3 Medium names the frozen DiT together with that same decoder") == 1
    for name in MONITOR.CONFIGURATION_NAMES.values():
        assert f"| {name} |" in content
