"""Replay frozen-fold model outputs and verify an autonomous experiment."""

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.multioutput import MultiOutputRegressor
from threadpoolctl import threadpool_limits

from ml_pipeline.agent_loop import apply_features
from ml_pipeline.agent_evidence import trial_evidence
from ml_pipeline.agent_metrics import score_metric
from ml_pipeline.benchmark_agent import read_json, write_json
from ml_pipeline.playground_runtime import prepare_dataset, reference_scores, score_predictions, split_indices
from ml_pipeline.utils.playground_trace import read_events


def audit(workspace):
    with threadpool_limits(limits=4):
        return _audit(workspace)


def _audit(workspace):
    workspace = Path(workspace)
    flow = read_json(workspace / 'autonomous.json')
    folder = Path(flow['folder'])
    state = read_json(folder / 'state.json')
    assert state['status'] == 'completed'
    config = state['config']
    frame = joblib.load(folder / 'development.joblib')
    test = joblib.load(folder / 'evaluation.joblib')
    parts = read_json(folder / 'partitions.json')
    assert not set(parts['development']) & set(parts['test'])
    _, y, _ = prepare_dataset(frame, config)
    splits = split_indices(frame, y, config)
    assert len(splits) == 3
    column = config.get('split_column')
    if config['split'] == 'temporal':
        assert pd.to_datetime(frame[column]).max() < pd.to_datetime(test[column]).min()
        if config.get('label_end_column'):
            assert pd.to_datetime(frame[config['label_end_column']]).max() < pd.to_datetime(test[column]).min()
    elif config['split'] == 'group':
        assert not set(frame[column]) & set(test[column])
    for train, valid in splits:
        assert not set(train) & set(valid)
        if config['split'] == 'temporal':
            assert pd.to_datetime(frame[column].iloc[train]).max() < pd.to_datetime(frame[column].iloc[valid]).min()
            if config.get('label_end_column'):
                assert pd.to_datetime(frame[config['label_end_column']].iloc[train]).max() < pd.to_datetime(frame[column].iloc[valid]).min()
        elif config['split'] == 'group':
            assert not set(frame[column].iloc[train]) & set(frame[column].iloc[valid])
    from ml_pipeline.pipelines_torch.models import CLASSIFICATION_MODEL_REGISTRY, REGRESSION_MODEL_REGISTRY
    registry = CLASSIFICATION_MODEL_REGISTRY if config['task'] == 'classification' else REGRESSION_MODEL_REGISTRY
    replays = []
    candidates = []
    for trial in state['trials']:
        if trial['status'] != 'completed':
            continue
        output = folder / f"trial_{trial['number']}" / 'output'
        assert read_json(output / 'splits.json') == read_json(folder / 'validation_splits.json')
        data = apply_features(frame, config['features'], trial['recipes'])
        X, encoded, _ = prepare_dataset(data, trial['config'])
        folds = pd.read_csv(output / 'fold_metrics.csv')
        for row in folds.to_dict('records'):
            train, valid = splits[int(row['fold']) - 1]
            transformer = joblib.load(output / f"preprocessor_fold_{int(row['fold'])}.joblib")
            if trial['config'].get('preprocessing', 'standard') == 'standard':
                numeric = X.iloc[train].select_dtypes(include='number')
                np.testing.assert_allclose(transformer[0].named_transformers_['numeric'].statistics_, numeric.median().fillna(0))
            model = joblib.load(output / f"{row['checkpoint_id']}.pkl")
            fitted = model.estimators_[0] if isinstance(model, MultiOutputRegressor) else model
            for key, value in trial['config'].get('model_params', {}).get(row['model'], {}).items():
                assert fitted.get_params()[key] == value, (row['model'], key)
            wrapper = registry[row['model']](**trial['config'].get('model_params', {}).get(row['model'], {}))
            wrapper.model = model
            values = np.asarray(transformer.transform(X.iloc[valid]), dtype=np.float32)
            predicted = np.asarray(wrapper.predict(values)).reshape(-1)
            probabilities = wrapper.predict_proba(values) if config['task'] == 'classification' else None
            scores = score_predictions(encoded[valid], predicted, probabilities, config['task'])
            saved = pd.read_csv(output / f"predictions_{row['checkpoint_id']}.csv")
            assert saved.row.tolist() == valid.tolist()
            np.testing.assert_array_equal(saved.y_true, encoded[valid])
            np.testing.assert_allclose(saved.y_pred.to_numpy(dtype=predicted.dtype), predicted, rtol=1e-7, atol=1e-7)
            for key,value in scores.items():
                assert abs(value - row[key]) < 1e-6, (trial['number'], row['model'], key)
            if probabilities is not None:
                np.testing.assert_allclose(saved[[f'probability_{i}' for i in range(probabilities.shape[1])]], probabilities, rtol=1e-6, atol=1e-8)
            reference = reference_scores(encoded[train], encoded[valid], config['task'])
            for key,value in reference['scores'].items():
                assert abs(value - row['reference_' + key]) < 1e-6
            replays.append({'trial': trial['number'], 'model': row['model'], 'fold': int(row['fold']), 'scores': scores})
        labels = list(read_json(output / 'labels.json').values()) if (output / 'labels.json').exists() else None
        for (model_name, augmentation), group in folds.groupby(['model','augmentation']):
            prediction = pd.concat([pd.read_csv(output / f'predictions_{row.checkpoint_id}.csv') for row in group.itertuples()], ignore_index=True)
            columns = [name for name in prediction if name.startswith('probability_')]
            probability = prediction[columns].to_numpy() if columns else None
            for definition in state.get('metric_definitions', []):
                actual = score_metric(definition, frame.iloc[prediction.row], prediction.y_true, prediction.y_pred, probability, labels)
                recorded = next(item for item in trial['metric_results'] if item['model'] == model_name and item['augmentation'] == augmentation)
                assert actual == next(item for item in recorded['metrics'] if item['name'] == definition['name'])
        for summary in trial['summary']:
            same = folds[(folds.model == summary['model']) & (folds.augmentation == summary['augmentation'])]
            if len(same) == len(splits):
                assert abs(same[config['metric']].mean() - summary[config['metric']]) < 1e-6
                candidates.append((trial['number'], summary))
    facts = {fact['id']: fact for trial in state['trials'] for fact in trial_evidence(trial, config['metric'])}
    for note in state['notes']:
        assert all(fact == facts[fact['id']] for fact in note['measured'])
        if note['decision'] == 'continue':
            following = state['trials'][note['trial']]
            assert following['config'] == note['next_config'] and following['recipes'] == note['next_recipes']
    select = min if config['metric'] in ['rmse','mae','smape'] else max
    winner = select(candidates, key=lambda item: item[1][config['metric']])
    assert winner[0] == state['best']['trial'] and winner[1]['model'] == state['best']['model']
    def final_score(number, model_name, augmentation):
        trial = state['trials'][number - 1]
        output = folder / f'trial_{number}' / 'output'
        rows = pd.read_csv(output / 'fold_metrics.csv')
        row = rows[(rows.model == model_name) & (rows.augmentation == augmentation)].sort_values('fold').iloc[-1]
        prepared = apply_features(test, config['features'], trial['recipes'])
        X, target, labels = prepare_dataset(prepared, trial['config'])
        if labels is not None:
            encoding = {value:int(key) for key,value in read_json(output / 'labels.json').items()}
            target = np.array([encoding[str(value)] for value in prepared[config['target']]])
        transformer = joblib.load(output / f'preprocessor_fold_{int(row.fold)}.joblib')
        model = joblib.load(output / f'{row.checkpoint_id}.pkl')
        values = np.asarray(transformer.transform(X), dtype=np.float32)
        p = model.predict_proba(values) if config['task'] == 'classification' else None
        return score_predictions(target, model.predict(values), p, config['task'])
    first = select([row for number,row in candidates if number == 1], key=lambda row: row[config['metric']])
    baseline = final_score(1, first['model'], first['augmentation'])
    control = next(row for number,row in candidates if number == 1 and row['model'].startswith('constant_'))
    constant_final = final_score(1, control['model'], control['augmentation'])
    best = state['best']
    selected = final_score(best['trial'], best['model'], best['augmentation'])
    for key,value in read_json(folder / 'evaluation.json')['scores'].items():
        assert abs(value - selected[key]) < 1e-6
    events = read_events(folder)
    assert sum(event['event'] == 'agent.model_selection' for event in events) == 1
    calls = [call['id'] for message in state['messages'] for call in message.get('tool_calls', [])]
    replies = [message['tool_call_id'] for message in state['messages'] if message['role'] == 'tool']
    assert calls == replies
    final_time = next(event['timestamp'] for event in events if event['event'] == 'agent.final_evaluation')
    assert all(event['timestamp'] < final_time for event in events if event['event'] == 'llm.request')
    starts = [event for event in events if event['event'] == 'execution.start' and event['operation'] == 'model.benchmark']
    workers = {event['data']['arguments'].get('worker_pid') for event in starts}
    intervals = []
    ends = {event['operation_id']: event['timestamp'] for event in events if event['event'] in ['execution.end','execution.error'] and event['operation'] == 'model.benchmark'}
    for event in starts:
        if event['operation_id'] in ends:
            intervals.extend([(event['timestamp'], 1), (ends[event['operation_id']], -1)])
    active = peak = 0
    for _, change in sorted(intervals):
        active += change
        peak = max(peak, active)
    result = {'replay_threads': 4, 'parallel_max_overlap': peak, 'all_checks_passed': True, 'replayed_fits': replays, 'validation_folds': len(splits),
              'baseline_validation': first, 'baseline_final_scores': baseline, 'selected_final_scores': selected,
              'constant_final_scores': constant_final,
              'single_model_approval': True, 'native_calls_answered': len(calls), 'worker_pids': sorted(workers),
              'primary_metric': config['metric'], 'protocol': 'Three frozen development folds; final prediction uses the last-fold saved fit.'}
    write_json(folder / 'audit.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('workspace', type=Path)
    args = parser.parse_args()
    print(audit(args.workspace))
