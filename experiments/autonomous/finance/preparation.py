import numpy as np
import pandas as pd

def build_dataset(tables, config):
    df = tables[config['table']].copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(['Symbol', 'Date'], kind='mergesort').reset_index(drop=True)
    g = df.groupby('Symbol', sort=False)
    adj = df['Adj Close']
    df['ret_5'] = adj / g['Adj Close'].shift(5) - 1
    df['ret_21'] = adj / g['Adj Close'].shift(21) - 1
    df['ret_63'] = adj / g['Adj Close'].shift(63) - 1
    df['daily_ret'] = adj / g['Adj Close'].shift(1) - 1
    df['vol_21'] = df.groupby('Symbol', sort=False)['daily_ret'].transform(lambda x: x.rolling(21, min_periods=21).std())
    df['log_vol'] = np.log1p(df['Volume'])
    df['volume_change_21'] = df['log_vol'] - df.groupby('Symbol', sort=False)['log_vol'].shift(21)
    future_adj = g['Adj Close'].shift(-5)
    future_ret = future_adj / adj - 1
    target = (future_ret <= -0.05).astype(float)
    target[future_ret.isna()] = np.nan
    df[config['target']] = target
    df[config['label_end_column']] = g['Date'].shift(-5)
    df = df[df['Date'].dt.dayofweek == 4].copy()
    if df.empty:
        raise ValueError('No Friday rows remain.')
    comp = tables['sp500_companies.csv'].copy()
    if comp['Symbol'].duplicated().any():
        raise ValueError('Company Symbol is not unique.')
    comp = comp[['Symbol', 'Sector']]
    if not df['Symbol'].isin(comp['Symbol']).all():
        raise ValueError('Some stock symbols are missing from company metadata.')
    df = df.merge(comp, on='Symbol', how='left', validate='many_to_one')
    idx = tables['sp500_index.csv'].copy()
    idx['Date'] = pd.to_datetime(idx['Date'])
    if idx['Date'].duplicated().any():
        raise ValueError('Index Date is not unique.')
    idx = idx.sort_values('Date')
    idx['index_ret_21'] = idx['S&P500'] / idx['S&P500'].shift(21) - 1
    idx = idx[['Date', 'index_ret_21']]
    if not df['Date'].isin(idx['Date']).all():
        df = df[df['Date'].isin(idx['Date'])]
    if df.empty:
        raise ValueError('No Friday rows remain after index date matching.')
    df = df.merge(idx, on='Date', how='left', validate='many_to_one')
    split_col = config.get('split_column')
    label_col = config.get('label_end_column')
    required = list(config['features']) + [config['target']]
    if label_col:
        required.append(label_col)
    if split_col:
        required.append(split_col)
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError('Missing required columns: ' + str(missing))
    df = df[df[required].notna().all(axis=1)]
    if df.empty:
        raise ValueError('No rows remain after requiring non-null features, target, and label end date.')
    df[config['target']] = df[config['target']].astype(int)
    return df[required + ['Symbol']].copy()

if __name__ == '__main__':
    from ml_pipeline.playground_runtime import run_experiment
    run_experiment(build_dataset)