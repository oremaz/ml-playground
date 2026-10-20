"""Persistent CSV-to-results workflow with a single model-approval pause."""

from pathlib import Path

import joblib

from ml_pipeline.benchmark_agent import generate_script, propose, read_json, write_json
from ml_pipeline.agent_loop import advance, approve_models, create_session, evaluate_selected, prepare_reviewed_dataset, visualize, restore_interrupted_trial, save_state
from ml_pipeline.playground_runtime import prepare_dataset
from ml_pipeline.utils.playground_trace import append_event


def configure(plan, *, seed=42, model_workers=2):
    if not plan.get('target'):
        raise ValueError('No defensible target was proposed; provide an objective before starting')
    plan.setdefault('augmentations', ['none'])
    plan.setdefault('preprocessing', 'none' if plan['task'] == 'forecasting' else 'standard')
    config = {**plan, 'seed': seed, 'validation_fraction': .2, 'device': 'cpu', 'model_workers': model_workers}
    if plan['task'] != 'forecasting':
        config.update(validation_protocol='robust', validation_folds=3)
    for name in config['models']:
        params = config.setdefault('model_params', {}).setdefault(name, {})
        if name.startswith(('random_forest_', 'lightgbm_', 'xgboost_')):
            params.setdefault('n_estimators', 100)
            params['n_jobs'] = max(1, min(4, int(params.get('n_jobs', 4))))
            params['n_estimators'] = max(1, min(400, int(params['n_estimators'])))
    for params in config.get('model_params', {}).values():
        if any(params.get(key, 'cpu') not in ('cpu', None) for key in ['device', 'device_type', 'task_type']):
            raise ValueError('The autonomous CPU policy does not allow GPU model parameters')
    return config


def start(workspace, settings, context, *, max_trials=5, model_workers=2, timeout=900):
    workspace = Path(workspace).resolve()
    path = workspace / 'autonomous.json'
    if not path.exists():
        write_json(path, {'phase': 'proposal', 'context': context, 'max_trials': max_trials,
                          'model_workers': model_workers, 'timeout': timeout, 'attempts': {}, 'recoveries': []})
    return resume(workspace, settings)


def _recover(workspace, flow, phase, exc):
    error = {'phase': phase, 'error': f'{type(exc).__name__}: {exc}'}
    flow['recoveries'].append(error)
    append_event(workspace, 'workflow.recovery', data=error)
    write_json(workspace / 'autonomous.json', flow)
    return error


def resume(workspace, settings, *, progress=None):
    workspace = Path(workspace).resolve()
    flow = read_json(workspace / 'autonomous.json')
    while flow['phase'] in ('proposal', 'preparation'):
        phase = flow['phase']
        attempts = flow['attempts'].get(phase, 0)
        if attempts >= 3:
            raise RuntimeError(f'{phase} exhausted its three-attempt recovery budget; inspect saved logs')
        flow['attempts'][phase] = attempts + 1
        write_json(workspace / 'autonomous.json', flow)
        try:
            if phase == 'proposal':
                retry = '\nPrevious attempt failed: ' + flow['recoveries'][-1]['error'] if attempts else ''
                plan = propose(workspace, settings, flow['context'] + retry)
                flow['config'] = configure(plan, model_workers=flow['model_workers'])
                flow['phase'] = 'preparation'
            else:
                instruction = ('Prepare only the reviewed deterministic transformation. Do not enumerate feature lists; '
                               'for unchanged tables return a copy. Do not fit learned transforms or drop targets silently.')
                if flow['recoveries']:
                    instruction += '\nActual previous failure: ' + flow['recoveries'][-1]['error']
                    instruction += f'\nRecovery attempt {attempts + 1}/3: keep source minimal and explanation to one sentence. Trust the runtime contract; do not duplicate its checks.'
                script, _ = generate_script(workspace, settings, flow['config'], instruction)
                folder = prepare_reviewed_dataset(workspace, script.read_text(), flow['config'], flow['timeout'])
                prepared = joblib.load(folder / 'prepared.joblib')
                prepare_dataset(prepared, flow['config'])
                create_session(folder, prepared, flow['config'], max_trials=flow['max_trials'], context=flow['context'], timeout=flow['timeout'])
                flow.update(phase='agent', folder=str(folder))
                write_json(workspace / 'active_agent.json', {'session': folder.name})
        except Exception as exc:
            _recover(workspace, flow, phase, exc)
            continue
        write_json(workspace / 'autonomous.json', flow)
    folder = Path(flow['folder'])
    if flow['phase'] == 'completed':
        return read_json(folder / 'state.json')
    if restore_interrupted_trial(folder):
        _recover(workspace, flow, 'interrupted_trial', RuntimeError('Recovered the saved worker state without repeating completed fits'))
    for attempt in range(3):
        try:
            actual_settings = {**settings, 'compact_context': bool(flow.get('native_retries', 0))}
            state = advance(folder, actual_settings, progress=progress)
            break
        except Exception as exc:
            _recover(workspace, flow, 'agent', exc)
            flow['native_retries'] = flow.get('native_retries', 0) + 1
            write_json(workspace / 'autonomous.json', flow)
            if flow['native_retries'] >= 3:
                raise RuntimeError('Native inference exhausted its three-attempt recovery budget') from exc
            saved = read_json(folder / 'state.json')
            saved['messages'].append({'role': 'user', 'content':
                'Automatic inference recovery: the preceding request failed before returning a message. '
                + str(exc)[:400] + ' Continue from the code-owned measured snapshot. Emit one concise tool call; do not repeat fits or request model approval again.'})
            save_state(folder, saved)
            if settings.get('provider') == 'local':
                import gc
                import torch
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    if state['status'] == 'completed':
        visualize(folder, state)
        evaluate_selected(folder)
        flow['phase'] = 'completed'
    write_json(workspace / 'autonomous.json', flow)
    return state


def select_models(workspace, settings, models, *, source='user', progress=None):
    workspace = Path(workspace).resolve()
    flow = read_json(workspace / 'autonomous.json')
    approve_models(Path(flow['folder']), models, source=source)
    return resume(workspace, settings, progress=progress)
