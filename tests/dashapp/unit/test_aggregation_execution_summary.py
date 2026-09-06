from __future__ import annotations

from types import SimpleNamespace

from scalebridge.dashapp.services.aggregation import execution


def test_definition_summary_surfaces_parent_lineage(monkeypatch, tmp_path):
    fake = SimpleNamespace(
        aggregation_campaign_id="agg_demo",
        parent_generation_campaign_id="gen_demo",
        machine_id="laptop",
        case_ids=("case1", "case2"),
        case_limit=None,
        plan_requests=(
            SimpleNamespace(rule_set=SimpleNamespace(value="legacy_v1")),
        ),
        requested_strategy_values=["identity"],
        requested_weight_mode_values=["equal"],
        max_variables=None,
        preview_rows=20,
        write_legacy_pickle=False,
        continue_on_error=True,
        mlflow_enabled=False,
        mlflow_tracking_uri="http://127.0.0.1:5000",
        mlflow_experiment_name=None,
    )
    path = tmp_path / "agg_demo.json"
    monkeypatch.setattr(execution, "load_definition", lambda _: fake)
    monkeypatch.setattr(execution, "definition_path", lambda _: path)
    monkeypatch.setattr(execution, "command_text", lambda _: "python runner --campaign-definition agg_demo.json")

    summary = execution.definition_summary("agg_demo")
    assert summary["aggregation_campaign_id"] == "agg_demo"
    assert summary["parent_generation_campaign_id"] == "gen_demo"
    assert summary["selected_case_count"] == 2
    assert summary["plan_request_count"] == 1
    assert summary["strategies"] == ["identity"]
    assert summary["weight_modes"] == ["equal"]
    assert summary["rule_sets"] == ["legacy_v1"]
