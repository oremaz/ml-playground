"""Run the same autonomous CSV workflow on four logged research cases."""

import argparse
import hashlib
import json
import os
from pathlib import Path

from ml_pipeline.agent_domain_experiment import CONTEXTS
from ml_pipeline.agent_workflow import start, resume, select_models
from ml_pipeline.agent_evaluation import summarize
from ml_pipeline.benchmark_agent import create_workspace, read_json, write_json


FINANCE = '''At Friday's close, rank S&P 500 stocks by risk of a drop of at least 5% over the
next five trading observations. Use validation average precision (pr_auc), with ROC-AUC and
precision/recall diagnostics. The input contains daily stock prices, current company metadata and
a daily S&P 500 index. Construct past-only trailing returns (5/21/63 observations), 21-observation
return volatility, 21-observation volume change and a past 21-observation index return. Sector is
available as static metadata with historical survivorship limitations. Price and volume of the current
close are known. Do not use Symbol, current market capitalization, future prices, forward returns,
label-end timestamps, company identifiers or post-outcome fields as inputs. Explicitly derive the
binary target from the future adjusted price return and retain the timestamp at which its fifth next
observation is known as label_end_column. Use exact Date joins and many-to-one cardinality checks
for static metadata and index rows; do not join across symbols. Retain Symbol as an audit column in the returned table, excluded from model inputs. Keep Friday decision rows with
available past features and observed future targets, explicitly documenting dropped warmup/end
rows. Use chronological validation with label-boundary purging. No trading edge or causal claims.'''


def run(args):
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('Run local 27B inference only inside a GPU allocation')
    torch.set_num_threads(4)
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    pointer = root / 'workspace.json'
    settings = {'provider': 'local', 'model': 'Qwen/Qwen3.8-27B', 'reasoning': 'xhigh', 'device': 'cuda', 'quantization': 'none'}
    package = Path(__file__).resolve().parent
    sources = {str(path.relative_to(package)): hashlib.sha256(path.read_bytes()).hexdigest()
               for path in package.glob('agent*.py')}
    for name in ['benchmark_agent.py', 'playground_runtime.py', 'playground_app.py', 'pipelines_torch/models.py', 'utils/local_qwen.py']:
        sources[name] = hashlib.sha256((package / name).read_bytes()).hexdigest()
    provenance = root / 'provenance.json'
    if provenance.exists():
        assert read_json(provenance)['sources'] == sources, 'Source changed during experiment; preserve this attempt and version the change before resuming'
    else:
        for name in sources:
            dest = root / 'source' / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes((package / name).read_bytes())
        write_json(provenance, {'sources': sources, 'settings': settings, 'slurm_job_id': os.getenv('SLURM_JOB_ID'),
                   'data_reuse': 'Previously inspected public datasets; retrospective evaluation only.'})
    if not pointer.exists():
        if args.domain == 'finance':
            old = Path('results/playground_memory/23d12ed876c64475afc2b7a92b15e896').resolve()
            uploads = [(name, (old / item['path']).read_bytes()) for name,item in read_json(old / 'tables.json').items()]
            context = FINANCE
        else:
            csv = Path('ml_pipeline/data/agent_domains') / f'{args.domain}.csv'
            uploads = [(csv.name, csv.read_bytes())]
            context = CONTEXTS[args.domain]
        context += ('\nAvailable resources: CPU Random Forest, LightGBM, XGBoost and the new regularized linear models '
                    'from the registry. Optional pretrained backends are unavailable for these runs. Choose any promising '
                    'families from those choices; the simulated user will approve your proposed families once. '
                    'The runtime includes an untuned constant fallback automatically, so you need not request it. '
                    'You can compare multiple approved models in every trial, with bounded parallel CPU fits. '
                    'Use standard preprocessing initially. Primary metric and target follow the objective above. '
                    'At most five trials; use diagnostics, feature subsets, and model-specific parameters when justified.')
        workspace = create_workspace(root / 'memory', uploads)
        write_json(pointer, {'path': str(workspace), 'context': context})
    info = read_json(pointer)
    workspace = Path(info['path'])
    def progress(state):
        print(args.domain, state['status'], 'step', state['llm_steps'], 'trials', len(state['trials']), flush=True)
    if (workspace / 'autonomous.json').exists():
        state = resume(workspace, settings, progress=progress)
    else:
        state = start(workspace, settings, info['context'], max_trials=5, model_workers=3, timeout=1200)
    if state['status'] == 'awaiting_models':
        eligible = {f'{family}_{suffix}' for family in ['random_forest', 'lightgbm', 'xgboost', 'linear', 'constant']
                    for suffix in ['classifier', 'regressor']}
        if set(state['proposed_models']) - eligible:
            raise ValueError('Proposed models exceed the stated available CPU resources')
        state = select_models(workspace, settings, state['proposed_models'], source='simulated_user', progress=progress)
    if state['status'] != 'completed':
        raise RuntimeError(f"Experiment ended in state {state['status']}")
    record = summarize(workspace, settings)
    write_json(root / 'record.json', record)
    print('COMPLETE', args.domain, json.dumps(record['selected']), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--domain', choices=['finance','energy','bikes','activity'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args())
