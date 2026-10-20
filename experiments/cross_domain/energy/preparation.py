import pandas as pd
from ml_pipeline.playground_runtime import run_experiment


def build_dataset(tables, config):
    table = config['table']
    if table not in tables:
        raise KeyError(f'Table {table!r} not found in provided tables.')
    df = tables[table].copy()

    target = config['target']
    features = list(dict.fromkeys(config['features']))
    split_column = config.get('split_column') or None
    label_end_column = config.get('label_end_column') or None

    if target not in df.columns:
        raise KeyError(f'Target column {target!r} is missing from {table}.')
    if split_column and split_column not in df.columns:
        raise KeyError(f'Split column {split_column!r} is missing from {table}.')
    if label_end_column and label_end_column not in df.columns:
        raise KeyError(f'label_end_column {label_end_column!r} is missing from {table}.')

    df[target] = pd.to_numeric(df[target], errors='raise')
    if df[target].isna().any():
        raise ValueError(f'Target column {target!r} contains missing values; refusing to drop rows.')

    if split_column:
        df[split_column] = pd.to_datetime(df[split_column], errors='raise')
        if df[split_column].isna().any():
            raise ValueError(f'Split column {split_column!r} contains missing values.')
        if df[split_column].duplicated().any():
            raise ValueError(f'Split column {split_column!r} contains duplicate timestamps; refusing to drop rows.')
        df = df.sort_values(split_column, kind='mergesort').reset_index(drop=True)
        ts = df[split_column]
        df['hour'] = ts.dt.hour.astype('int32')
        df['minute'] = ts.dt.minute.astype('int32')
        df['dayofweek'] = ts.dt.dayofweek.astype('int32')
        df['is_weekend'] = (ts.dt.dayofweek >= 5).astype('int32')
        df['month'] = ts.dt.month.astype('int32')
        df['day'] = ts.dt.day.astype('int32')
        df['is_evening'] = ((ts.dt.hour >= 17) & (ts.dt.hour <= 22)).astype('int32')

    calendar_features = {'hour', 'minute', 'dayofweek', 'is_weekend', 'month', 'day', 'is_evening'}
    sensor_features = [c for c in features if c not in calendar_features and c != split_column and c != label_end_column and c != target]
    for col in sensor_features:
        if col not in df.columns:
            raise KeyError(f'Feature column {col!r} is missing from {table}.')
        df[col] = pd.to_numeric(df[col], errors='raise')

    missing_features = [c for c in features if c not in df.columns]
    if missing_features:
        raise KeyError(f'Missing feature columns after preparation: {missing_features}')

    if label_end_column:
        features = [c for c in features if c != label_end_column]

    columns = list(features)
    if target not in columns:
        columns.append(target)
    if split_column and split_column not in columns:
        columns.append(split_column)
    if label_end_column and label_end_column not in columns:
        columns.append(label_end_column)

    return df[columns].copy()


if __name__ == '__main__':
    run_experiment(build_dataset)
