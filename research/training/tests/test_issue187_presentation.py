import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "reports/scripts/show_issue187_progress.py"
SPEC = importlib.util.spec_from_file_location("show_issue187_progress", SCRIPT)
MONITOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MONITOR)


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
    complete = {"updates": 8_000, "stopped": None}
    assert MONITOR.training_status(capped, active=False) == "Stopped"
    assert MONITOR.training_status(active, active=True) == "In progress"
    assert MONITOR.training_status(complete, active=False) == "Complete"


def test_report_defines_embedded_decoder_and_medium_trunk_once():
    report = SCRIPT.parents[1] / "issue187_pretrained_vital_surrogates" / "STAGE2_PREPARATION.md"
    content = report.read_text(encoding="utf-8")
    assert content.count("SAME-L names the embedded autoencoder's decoder") == 1
    assert content.count("Stable Audio 3 Medium names the frozen DiT together with that same decoder") == 1
    for name in MONITOR.CONFIGURATION_NAMES.values():
        assert f"| {name} |" in content
