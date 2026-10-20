import json

from streamlit.testing.v1 import AppTest

from ml_pipeline import agent_loop as agent
from ml_pipeline.benchmark_agent import create_workspace, read_json, write_json
from ml_pipeline.tests.test_forecasting import panel, config
from ml_pipeline.tests.test_agent_evidence import native, review


def test_forecasting_configuration_and_high_local_reasoning_default(tmp_path):
    workspace = create_workspace(tmp_path, [("series.csv", panel().to_csv(index=False).encode())])
    cfg = {**config(), "table": "series.csv"}
    write_json(workspace / "plan.json", cfg)
    source = f'''
from pathlib import Path
from ml_pipeline.playground_app import _configuration, _provider_settings
from ml_pipeline.benchmark_agent import read_json, write_json
workspace = Path({str(workspace)!r})
_provider_settings()
reviewed = _configuration(read_json(workspace / "plan.json"), workspace)
if reviewed:
    write_json(workspace / "reviewed.json", reviewed)
'''
    app = AppTest.from_string(source, default_timeout=120).run()
    assert not app.exception
    next(item for item in app.selectbox if item.label == "LLM provider").select("local").run()
    assert not app.exception
    assert next(item for item in app.selectbox if item.label == "Reasoning effort").value == "xhigh"
    reviewed = read_json(workspace / "reviewed.json")
    assert reviewed["task"] == "forecasting" and reviewed["features"] == []
    for key in ["series_column", "forecast_horizon", "context_length", "min_context", "backtest_windows"]:
        assert reviewed[key] == cfg[key]
    methods = next(item for item in app.multiselect if item.label == "Methods to compare")
    assert "timestral_209m" in methods.options and "chronos_2" in methods.options


def test_real_structured_notes_reopen_in_streamlit(tmp_path, monkeypatch):
    workspace = create_workspace(tmp_path, [("series.csv", panel().to_csv(index=False).encode())])
    cfg = {**config(), "table": "series.csv"}
    folder = workspace / "agents" / "session"
    agent.create_session(folder, panel(), cfg, max_trials=1, context="Seasonal forecasting")
    monkeypatch.setattr(agent, "complete_message", lambda *a, **k: native("request_model_selection", "approve", models=cfg["models"]))
    agent.advance(folder, {})
    agent.approve_models(folder, cfg["models"])
    replies = iter([native("run_benchmark", "baseline", overrides_json="{}"),
                    native("review_trial", "review", **review(1, "stop", model="naive_forecaster", metric="mae"))])
    monkeypatch.setattr(agent, "complete_message", lambda *a, **k: next(replies))
    state = agent.advance(folder, {})
    assert state["status"] == "completed"
    agent.evaluate_selected(folder)
    write_json(workspace / "config.json", cfg)
    source = f'''
from pathlib import Path
from ml_pipeline.playground_app import _iterative_assistant
from ml_pipeline.benchmark_agent import read_json
workspace = Path({str(workspace)!r})
_iterative_assistant(workspace, {{}}, "", read_json(workspace / "config.json"), 120)
'''
    app = AppTest.from_string(source, default_timeout=120).run()
    assert not app.exception
    assert len(app.dataframe) == 1 and len(app.dataframe[0].value) == 2
    assert len(app.chat_message) == 1
    assert any("Hypotheses, not established causes" in item.value for item in app.markdown)
    assert any("naive_forecaster" in item.value for item in app.markdown)
    assert (folder / "trial_1" / "research_note.json").exists()
