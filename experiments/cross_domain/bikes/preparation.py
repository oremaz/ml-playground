from ml_pipeline.playground_runtime import run_experiment
import pandas as pd

def build_dataset(tables, config):
    df = tables[config['table']].copy()
    target = config['target']
    if df[target].isna().any():
        raise ValueError('Target contains missing values.')

    df['timestamp'] = pd.to_datetime(df['dteday']) + pd.to_timedelta(df['hr'], unit='h')
    if df['timestamp'].isna().any():
        raise ValueError('timestamp contains missing values.')
    if df['timestamp'].duplicated().any():
        raise ValueError('timestamp is not unique; refusing to join previous-hour weather.')

    label_end_col = config.get('label_end_column')
    if label_end_col:
        df[label_end_col] = df['timestamp'] + pd.Timedelta(hours=1)
    df['prev_timestamp'] = df['timestamp'] - pd.Timedelta(hours=1)

    previous = (
        df[['timestamp', 'weathersit', 'temp', 'atemp', 'hum', 'windspeed']]
        .rename(columns={
            'timestamp': 'prev_timestamp',
            'weathersit': 'weathersit_prev',
            'temp': 'temp_prev',
            'atemp': 'atemp_prev',
            'hum': 'hum_prev',
            'windspeed': 'windspeed_prev',
        })
    )
    if previous['prev_timestamp'].duplicated().any():
        raise ValueError('Previous-hour timestamp is not unique.')

    merged = df.merge(previous, how='left', on='prev_timestamp', validate='one_to_one')
    if len(merged) != len(df):
        raise ValueError('Previous-hour join changed row count.')

    merged['is_commuter_focus'] = (
        (merged['workingday'] == 1) & (merged['hr'].isin([7, 8, 9, 16, 17, 18, 19]))
    ).astype(int)

    required = list(config['features']) + [target]
    if config.get('split_column'):
        required.append(config['split_column'])
    if label_end_col:
        required.append(label_end_col)
    required = list(dict.fromkeys(required))
    missing = [col for col in required if col not in merged.columns]
    if missing:
        raise ValueError(f'Missing required columns: {missing}')

    out = merged[required].copy()
    if config.get('split_column'):
        out = out.sort_values(config['split_column'], kind='mergesort').reset_index(drop=True)
    return out

if __name__ == '__main__':
    run_experiment(build_dataset)
