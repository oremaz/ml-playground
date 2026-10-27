"""Exercise the real standard-comparison workflow and its saved report."""

import json
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

pytest.importorskip("skore")


@pytest.mark.parametrize("task", ["classification", "regression"])
def test_standard_workflow_fits_raw_table_and_reopens_without_refit(tmp_path, monkeypatch, task):
    from ml_pipeline import standard_benchmark_app as ui

    monkeypatch.setattr(ui, "RESULTS_DIR_OUT", str(tmp_path / "results"))
    evaluate = Mock(wraps=ui.evaluate_models)
    monkeypatch.setattr(ui, "evaluate_models", evaluate)
    frame = pd.DataFrame({"number": np.arange(40, dtype=float), "category": ["a", "b"] * 20,
                          "target": [0, 1] * 20 if task == "classification" else np.arange(40) / 3})
    frame.loc[2, "number"] = np.nan
    csv = tmp_path / "data.csv"
    frame.to_csv(csv, index=False)
    app = AppTest.from_file(str(Path(ui.__file__).with_name("benchmark_app.py")), default_timeout=180).run()
    app.radio[0].set_value("Standard comparison").run()
    app.radio(key="standard_source").set_value("Path").run()
    app.selectbox(key="standard_task").set_value(task)
    app.text_input(key="standard_path").set_value(str(csv)).run()
    app.selectbox(key="standard_target").set_value("target").run()
    app.text_input(key="standard_name").set_value("example")
    app.text_area(key="standard_params").set_value('{"random_forest": {"n_estimators": 5}}')
    app.button(key="standard_run").click().run()
    assert not app.exception, app.exception
    assert not app.error, [error.value for error in app.error]
    evaluate.assert_called_once()
    X = evaluate.call_args.args[1]
    assert X.columns.tolist() == ["number", "category"]
    assert X["number"].isna().sum() == 1
    destination = tmp_path / "results/standard_comparisons/example"
    protocol = json.loads((destination / "protocol.json").read_text())
    assert len(protocol["splits"]) == 5
    assert protocol["preprocessing"] == "skrub"
    assert (destination / "report.joblib").is_file()
    app.run()
    assert not app.exception
    evaluate.assert_called_once()


def test_ui_splitters_respect_groups_and_time():
    from ml_pipeline.standard_benchmark_app import comparison_splitter

    X = np.arange(24).reshape(12, 2)
    groups = np.repeat(["a", "b", "c"], 4)
    for train, valid in comparison_splitter("regression", "Disjoint groups", 3, 42).split(X, groups=groups):
        assert set(groups[train]).isdisjoint(groups[valid])
    for train, valid in comparison_splitter("regression", "Chronological row order", 3, 42).split(X):
        assert train.max() < valid.min()
