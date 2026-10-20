"""Comparable agent-run records without an LLM judging its own explanations."""

import argparse
import hashlib
import json
import joblib
import pandas as pd
from pathlib import Path

from ml_pipeline.benchmark_agent import read_json, write_json
from ml_pipeline.utils.playground_trace import read_events


def summarize(workspace, settings):
    workspace = Path(workspace)
    flow = read_json(workspace / 'autonomous.json')
    folder = Path(flow['folder'])
    state = read_json(folder / 'state.json')
    events = read_events(folder)
    calls = [call for message in state['messages'] for call in message.get('tool_calls', [])]
    replies = [message['tool_call_id'] for message in state['messages'] if message['role'] == 'tool']
    assert [call['id'] for call in calls] == replies
    approval = [event for event in events if event['event'] == 'agent.model_selection']
    assert len(approval) == 1
    config = state['config']
    protocol = {key: config.get(key) for key in ['task', 'target', 'metric', 'seed', 'split', 'split_column', 'label_end_column', 'validation_protocol', '_split_sha256', 'model_workers']}
    datasets = {key: item['sha256'] for key, item in read_json(workspace / 'tables.json').items()}
    prepared = joblib.load(folder / 'prepared.joblib')
    columns = list(dict.fromkeys([config['target']] + [config[key] for key in ['split_column','label_end_column'] if config.get(key)]))
    target_hash = hashlib.sha256(pd.util.hash_pandas_object(prepared[columns], index=True).values.tobytes()).hexdigest()
    package = Path(__file__).resolve().parent
    harness_hash = {name: hashlib.sha256((package / name).read_bytes()).hexdigest() for name in
                    ['agent_loop.py','agent_workflow.py','agent_validation.py','agent_diagnostics.py','agent_metrics.py',
                     'benchmark_agent.py','playground_runtime.py','pipelines_torch/models.py','utils/local_qwen.py']}
    identity = {'datasets': datasets, 'protocol': protocol, 'target_and_row_identity': target_hash,
                'objective': flow['context'], 'harness': harness_hash,
                'max_trials': state['max_trials'], 'approved_models': sorted(state['approved_models'])}
    flag_words = ['intrinsic', 'ceiling', 'near-optimal', 'practical limit', 'main driver']
    flags = [{'trial': note['trial'], 'field': key, 'text': note[key]} for note in state.get('notes', [])
             for key in ['hypothesis', 'alternative', 'distinguishes'] if any(word in note[key].lower() for word in flag_words)]
    return {'settings': settings, 'comparison_identity': identity,
            'comparison_fingerprint': hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest(),
            'completed': state['status'] == 'completed', 'operator_interventions': len(flow.get('source_updates', [])),
            'automatic_recoveries': flow['recoveries'], 'single_model_approval': True,
            'native_tool_calls': len(calls), 'all_calls_answered': True, 'trials': len(state['trials']),
            'models_per_trial': [trial['config']['models'] for trial in state['trials']],
            'selected': state['best'], 'final_evaluation': read_json(folder / 'evaluation.json'),
            'explanation_review_candidates': flags,
            'explanation_review_note': 'Keyword flags request human review; they are not unsupported-claim verdicts. Negated cautions may be flagged.',
            'evaluation_caveat': 'These four dataset final periods were inspected in earlier work. Reruns are retrospective, not fresh prospective tests.'}


def compare(records):
    if len(records) < 2 or len({record['comparison_fingerprint'] for record in records}) != 1:
        raise ValueError('A controlled LLM comparison requires identical datasets, folds, approval set, seed, workers and trial budget')
    return [{'settings': record['settings'], 'completed': record['completed'],
             'recoveries': len(record['automatic_recoveries']), 'calls': record['native_tool_calls'],
             'final_scores': record['final_evaluation']['scores']} for record in records]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('records', type=Path, nargs='+')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    write_json(args.output, compare([read_json(path) for path in args.records]))
