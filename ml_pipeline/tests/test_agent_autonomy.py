import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from ml_pipeline import agent_loop as agent
from ml_pipeline.agent_validation import development_splits
from ml_pipeline.agent_workflow import start, select_models
from ml_pipeline.agent_evaluation import compare
from ml_pipeline.benchmark_agent import read_json, write_json
from ml_pipeline.playground_runtime import benchmark, split_indices
from ml_pipeline.utils.playground_trace import read_events


def dataset():
    rng = np.random.default_rng(7)
    x = rng.normal(size=180)
    return pd.DataFrame({'x': x, 'z': rng.normal(size=180), 'target': x * 2 + rng.normal(size=180),
                         'date': pd.date_range('2020-01-01', periods=180),
                         'end': pd.date_range('2020-01-04', periods=180), 'person': np.repeat(np.arange(30), 6)})


def config():
    return {'task': 'regression', 'target': 'target', 'features': ['x', 'z'], 'table': 'data.csv',
            'models': ['random_forest_regressor', 'linear_regressor'], 'augmentations': ['none'],
            'model_params': {'random_forest_regressor': {'n_estimators': 5, 'n_jobs': 1}},
            'split': 'temporal', 'split_column': 'date', 'label_end_column': 'end', 'validation_fraction': .2,
            'validation_protocol': 'robust', 'validation_folds': 3, 'model_workers': 1,
            'preprocessing': 'standard', 'metric': 'rmse', 'seed': 42}


def call(name, **args):
    return {'id': 'call_' + name, 'type': 'function', 'function': {'name': name, 'arguments': json.dumps({'reason': 'Measure this comparison.', **args})}}


def test_temporal_and_group_protocols_have_disjoint_validations_and_purged_labels():
    frame, cfg = dataset(), config()
    seen = set()
    for train, valid in development_splits(frame, frame.target.to_numpy(), cfg):
        assert frame.end.iloc[train].max() < frame.date.iloc[valid].min()
        assert not seen & set(valid)
        seen.update(valid)
    cfg.update(split='group', split_column='person', label_end_column='')
    seen = set()
    for train, valid in development_splits(frame, frame.target.to_numpy(), cfg):
        assert not set(frame.person.iloc[train]) & set(frame.person.iloc[valid])
        assert not seen & set(valid)
        seen.update(valid)
    assert len(seen) == len(frame)


@pytest.mark.parametrize('classification', [False, True])
def test_multifold_controls_parallel_equality_and_diagnostics(tmp_path, classification):
    frame, cfg = dataset(), config()
    if classification:
        frame['target'] = (frame.x > 0).astype(int)
        cfg.update(task='classification', metric='pr_auc', models=['random_forest_classifier', 'linear_classifier'],
                   model_params={'random_forest_classifier': {'n_estimators': 5, 'n_jobs': 1}})
    folder = tmp_path / 'session'
    state = agent.create_session(folder, frame, cfg, max_trials=1, context='Predict from x and z')
    request = call('request_model_selection', models=cfg['models'])
    state['pending'] = [request]
    state['messages'].append({'role': 'assistant', 'tool_calls': [request], 'content': ''})
    agent.dispatch(folder, state, request)
    agent.save_state(folder, state)
    state = agent.approve_models(folder, cfg['models'], source='simulated_user')
    result = agent._benchmark_tool(folder, state, {'overrides_json': '{}', 'reason': 'Baseline comparison'})
    assert result['status'] == 'completed', result
    trial = state['trials'][0]
    assert all(row['completed_folds'] == row['expected_folds'] == 3 for row in trial['summary'])
    assert any(row['model'].startswith('constant_') for row in trial['summary'])
    output = folder / 'trial_1' / 'output'
    splits = read_json(output / 'splits.json')
    dev = joblib.load(folder / 'development.joblib')
    for fold, part in enumerate(splits, 1):
        transformer = joblib.load(output / f'preprocessor_fold_{fold}.joblib')
        np.testing.assert_allclose(transformer[0].named_transformers_['numeric'].statistics_, dev.iloc[part['train']][['x','z']].median())
    parallel = benchmark(dev, {**trial['config'], 'model_workers': 2}, tmp_path / 'parallel')
    serial = pd.read_csv(output / 'summary.csv')
    pd.testing.assert_frame_equal(serial, parallel, check_exact=False, rtol=1e-7, atol=1e-7)
    state['trials'][0]['status'] = 'running'
    state['best'] = None
    pending = call('run_benchmark', overrides_json='{}')
    state['messages'].append({'role': 'assistant', 'tool_calls': [pending], 'content': ''})
    state['pending'] = [pending]
    agent.save_state(folder, state)
    assert agent.restore_interrupted_trial(folder)
    state = read_json(folder / 'state.json')
    assert state['trials'][0]['status'] == 'completed' and not state['pending']
    diagnostics = agent._trial_diagnostics(folder, state, 1)
    assert all(len(item['validation_folds']) == 3 for item in diagnostics['diagnostics'])
    state['status'] = 'completed'
    agent.save_state(folder, state)
    final = agent.evaluate_selected(folder)
    assert final['checkpoint_fold'] == 3
    assert final['test_rows'] == len(joblib.load(folder / 'evaluation.joblib'))
    manifest = Path(state['config']['_split_manifest'])
    manifest.write_text('[]')
    with pytest.raises(ValueError, match='changed'):
        split_indices(dev, dev.target.to_numpy(), state['config'])


def test_autonomous_preparation_repair_single_approval_and_completion(tmp_path, monkeypatch):
    from ml_pipeline import agent_workflow as workflow
    from ml_pipeline.benchmark_agent import create_workspace
    workspace = create_workspace(tmp_path, [('data.csv', dataset().to_csv(index=False).encode())])
    cfg = config()
    monkeypatch.setattr(workflow, 'propose', lambda *a, **k: cfg)
    attempts = []
    def generate(*args, **kwargs):
        attempts.append(kwargs.get('instruction', args[-1] if len(args) > 3 else ''))
        if len(attempts) == 1:
            raise SyntaxError('Generated source was incomplete')
        path = workspace / 'revision_0001.py'
        path.write_text('def build_dataset(tables, config):\n    return tables[config["table"]].copy()\n')
        return path, 'Copy the input.'
    monkeypatch.setattr(workflow, 'generate_script', generate)
    failures = []
    def message(*args, **kwargs):
        state = read_json(Path(kwargs['workspace']) / 'state.json')
        if not state['approval_requested']:
            action = call('request_model_selection', models=cfg['models'])
        elif not state['trials']:
            if not failures:
                failures.append(True)
                raise RuntimeError('Temporary inference failure before producing a call')
            action = call('run_benchmark', overrides_json='{}')
        else:
            action = call('review_trial', trial=1, evidence_ids=['t1.random_forest_regressor.none.rmse'],
                          hypothesis='', alternative='One comparison does not establish generalization.',
                          next_test='', distinguishes='', decision='stop', overrides_json='{}', recipes_json='[]')
        return {'role': 'assistant', 'content': '', 'tool_calls': [action]}
    monkeypatch.setattr(agent, 'complete_message', message)
    state = start(workspace, {'provider': 'test'}, 'Predict target', max_trials=1, model_workers=1)
    assert state['status'] == 'awaiting_models'
    from streamlit.testing.v1 import AppTest
    app = AppTest.from_file(str(Path(agent.__file__).parent / 'benchmark_app.py'), default_timeout=240)
    app.session_state['playground_workspace'] = str(workspace)
    app.run(timeout=240)
    assert not app.exception
    button = next(item for item in app.button if item.label == 'Approve models and continue')
    button.click().run(timeout=240)
    assert not app.exception and not app.error
    state = read_json(Path(read_json(workspace / 'autonomous.json')['folder']) / 'state.json')
    assert state['status'] == 'completed' 
    flow = read_json(workspace / 'autonomous.json')
    assert flow['phase'] == 'completed' and len(flow['recoveries']) == 2
    assert flow['native_retries'] == 1
    assert 'incomplete' in attempts[1]
    assert sum(event['event'] == 'agent.model_selection' for event in read_events(Path(flow['folder']))) == 1


def test_llm_comparison_rejects_changed_protocol():
    with pytest.raises(ValueError, match='identical'):
        compare([{'comparison_fingerprint': 'a'}, {'comparison_fingerprint': 'b'}])


def test_automatic_control_can_be_repeated_but_never_tuned():
    state = {'config': config(), 'trials': [{}], 'approved_models': config()['models'], 'recipes': []}
    overrides = {'models': ['linear_regressor', 'constant_regressor'], 'model_params': {'constant_regressor': {}}}
    resolved = agent._trial_config(state, overrides)
    assert resolved['models'] == overrides['models'] and resolved['model_params']['constant_regressor'] == {}
    with pytest.raises(ValueError, match='untuned'):
        agent._trial_config(state, {**overrides, 'model_params': {'constant_regressor': {'constant': 7}}})
    with pytest.raises(ValueError, match='approval'):
        agent._trial_config(state, {'models': ['catboost_regressor']})


def test_multiple_models_in_a_later_trial_and_iid_outer_holdout(tmp_path):
    frame, cfg = dataset(), config()
    cfg.update(split='kfold', split_column='', label_end_column='', model_workers=2)
    folder = tmp_path / 'session'
    state = agent.create_session(folder, frame, cfg, max_trials=2, context='Compare two simple families')
    request = call('request_model_selection', models=cfg['models'])
    state['pending'] = [request]
    state['messages'].append({'role': 'assistant', 'tool_calls': [request], 'content': ''})
    agent.dispatch(folder, state, request)
    agent.save_state(folder, state)
    state = agent.approve_models(folder, cfg['models'], source='simulated_user')
    baseline = agent._benchmark_tool(folder, state, {'overrides_json': '{}', 'reason': 'Baseline'})
    assert baseline['status'] == 'completed'
    overrides = {'models': cfg['models'], 'model_params': {'random_forest_regressor': {'n_estimators': 7}, 'linear_regressor': {'alpha': .5}}}
    result = agent._review_trial(folder, state, {'trial': 1, 'evidence_ids': ['t1.linear_regressor.none.rmse'],
        'hypothesis': 'Do these per-family settings improve either candidate?',
        'alternative': 'Either candidate may worsen.', 'next_test': 'Compare both updated candidates.',
        'distinguishes': 'Each family has one changed setting on the same frozen folds.', 'decision': 'continue',
        'overrides_json': json.dumps(overrides), 'recipes_json': '[]'})
    assert result['benchmark']['status'] == 'completed'
    assert set(state['trials'][1]['config']['models']) == set(cfg['models']) | {'constant_regressor'}
    assert all(row['completed_folds'] == 3 for row in state['trials'][1]['summary'])
    assert len(read_json(folder / 'validation_splits.json')) == 3
