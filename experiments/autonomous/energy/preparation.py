import pandas as pd
from ml_pipeline.playground_runtime import run_experiment

def build_dataset(tables, config):
    df = tables[config['table']].copy()
    split_col = config.get('split_column')
    if split_col:
        df[split_col] = pd.to_datetime(df[split_col])
        dt = df[split_col]
        derived = {
            'hour': dt.dt.hour,
            'dayofweek': dt.dt.dayofweek,
            'month': dt.dt.month,
            'is_weekend': (dt.dt.dayofweek >= 5).astype(int),
            'is_evening_17_22': dt.dt.hour.between(17, 22).astype(int),
        }
        for name, value in derived.items():
            if name in config['features']:
                df[name] = value
    keep = [config['target'], *config['features']]
    if split_col:
        keep.append(split_col)
    return df[list(dict.fromkeys(keep))]

if __name__ == '__main__':
    run_experiment(build_dataset)
