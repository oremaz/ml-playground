from ml_pipeline.playground_runtime import run_experiment

def build_dataset(tables, config):
    import pandas as pd

    df = tables[config['table']].copy()
    target = config['target']
    features = list(config['features'])
    split_column = config.get('split_column') or ''
    label_end_column = config.get('label_end_column') or ''

    required = {'dteday', 'hr', 'workingday', 'weathersit', 'temp', 'atemp', 'hum', 'windspeed', target}
    missing_required = required - set(df.columns)
    if missing_required:
        raise ValueError(f'Missing required source columns: {sorted(missing_required)}')

    forbidden = {'instant', 'dteday', 'casual', 'registered', 'weathersit', 'temp', 'atemp', 'hum', 'windspeed', target}
    if split_column:
        forbidden.add(split_column)
    if label_end_column:
        forbidden.add(label_end_column)
    leaked = set(features) & forbidden
    if leaked:
        raise ValueError(f'Features must not include unavailable or split columns: {sorted(leaked)}')

    if not pd.api.types.is_numeric_dtype(df[target]):
        raise ValueError('Target must be numeric.')
    if df[target].isna().any():
        raise ValueError('Target contains missing values; refusing to drop rows silently.')

    timestamp = pd.to_datetime(df['dteday'], errors='raise') + pd.to_timedelta(df['hr'], unit='h')
    if timestamp.isna().any():
        raise ValueError('timestamp contains missing values.')
    if timestamp.duplicated().any():
        raise ValueError('timestamp is not unique; previous-hour join would be ambiguous.')

    df['timestamp'] = timestamp
    df['label_end'] = df['timestamp'] + pd.Timedelta(hours=1)

    left = df.copy()
    left['prev_timestamp'] = left['timestamp'] - pd.Timedelta(hours=1)
    right = df[['timestamp', 'weathersit', 'temp', 'atemp', 'hum', 'windspeed']].rename(
        columns={
            'timestamp': 'prev_timestamp',
            'weathersit': 'prev_weathersit',
            'temp': 'prev_temp',
            'atemp': 'prev_atemp',
            'hum': 'prev_hum',
            'windspeed': 'prev_windspeed',
        }
    )
    right['_prev_match'] = True

    merged = left.merge(right, on='prev_timestamp', how='left', validate='one_to_one')
    if len(merged) != len(left):
        raise ValueError('Previous-hour join changed row count.')

    matched = int(merged['_prev_match'].notna().sum())
    missing_previous = len(merged) - matched
    if missing_previous == len(merged):
        raise ValueError('No previous-hour weather matches found; check timestamp construction.')

    merged = merged.drop(columns=['prev_timestamp', '_prev_match'])
    merged = merged.sort_values('timestamp', kind='mergesort').reset_index(drop=True)

    merged['is_morning_commuter'] = ((merged['workingday'] == 1) & (merged['hr'].isin([7, 8, 9]))).astype(int)
    merged['is_evening_commuter'] = ((merged['workingday'] == 1) & (merged['hr'].isin([16, 17, 18, 19]))).astype(int)

    final_columns = list(dict.fromkeys(features + [target] + ([split_column] if split_column else []) + ([label_end_column] if label_end_column else [])))
    missing_final = [c for c in final_columns if c not in merged.columns]
    if missing_final:
        raise ValueError(f'Missing required output columns: {missing_final}')

    return merged[final_columns].copy()

if __name__ == '__main__':
    run_experiment(build_dataset)
