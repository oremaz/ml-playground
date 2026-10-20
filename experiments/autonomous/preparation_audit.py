"""Dataset fixture checks kept separate from the dataset-agnostic agent."""

import json
from pathlib import Path
import runpy

import joblib
import numpy as np
import pandas as pd

from ml_pipeline.benchmark_agent import read_json, write_json


def audit(domain, workspace):
    workspace = Path(workspace)
    flow = read_json(workspace / 'autonomous.json')
    folder = Path(flow['folder'])
    config = read_json(folder / 'state.json')['config']
    tables = {name: pd.read_csv(workspace / item['path']) for name,item in read_json(workspace / 'tables.json').items()}
    frame = joblib.load(folder / 'prepared.joblib')
    build = runpy.run_path(str(folder / 'preparation.py'), run_name='audit_preparation')['build_dataset']
    original = build({key:value.copy() for key,value in tables.items()}, config)
    pd.testing.assert_frame_equal(original, frame)
    raw = tables[config['table']]
    target, features, time = config['target'], config['features'], config.get('split_column')
    checked = ['saved preparation reproduced']
    if domain != 'finance':
        assert len(frame) == len(raw)
        np.testing.assert_array_equal(frame[target], raw[target])
        modified = {key:value.copy() for key,value in tables.items()}
        modified[config['table']][target] = raw[target].iloc[::-1].to_numpy()
        rebuilt = build(modified, config)
        pd.testing.assert_frame_equal(frame[features], rebuilt[features])
        checked.append('original rows/targets preserved; model inputs independent of target values')
    if domain == 'activity':
        np.testing.assert_array_equal(frame[time], raw.subject)
        assert time not in features and 'subject' not in features
        checked.append('participant IDs retained only for grouping')
    elif domain in ['energy','bikes']:
        expected = pd.to_datetime(raw.date if domain == 'energy' else raw.dteday, utc=True)
        if domain == 'bikes':
            expected += pd.to_timedelta(raw.hr, unit='h')
            np.testing.assert_array_equal(pd.to_datetime(frame[config['label_end_column']], utc=True), expected + pd.Timedelta(hours=1))
            assert not set(['weathersit','temp','atemp','hum','windspeed','casual','registered','instant']) & set(features)
            checked.append('bike observation-end times and excluded current weather/target components checked')
        else:
            assert not set(['lights','rv1','rv2','T_out','Press_mm_hg','RH_out','Windspeed','Visibility','Tdewpoint']) & set(features)
        np.testing.assert_array_equal(pd.to_datetime(frame[time], utc=True), expected)
        modified = {key:value.copy() for key,value in tables.items()}
        cut = len(raw) // 2
        names = [f'T{i}' for i in range(1,10)] + [f'RH_{i}' for i in range(1,10)] if domain == 'energy' else ['weathersit','temp','atemp','hum','windspeed']
        first = cut + 1 if domain == 'energy' else cut
        continuous = [name for name in names if name != 'weathersit']
        modified[config['table']] = modified[config['table']].astype({name:float for name in continuous})
        modified[config['table']].loc[first:,continuous] *= .9
        if 'weathersit' in names:
            modified[config['table']].loc[first:,'weathersit'] = modified[config['table']].loc[first:,'weathersit'] % 3 + 1
        rebuilt = build(modified, config)
        pd.testing.assert_frame_equal(frame.iloc[:cut+1][features], rebuilt.iloc[:cut+1][features], check_dtype=False)
        checked.append('future sensor perturbation cannot change past model inputs')
        if domain == 'bikes':
            weather = raw.set_index(expected)
            mappings = {}
            for field in ['weathersit','temp','atemp','hum','windspeed']:
                expected_lag = weather[field].reindex(expected - pd.Timedelta(hours=1)).to_numpy()
                matches = [name for name in features if pd.api.types.is_numeric_dtype(frame[name]) and np.allclose(frame[name], expected_lag, equal_nan=True)]
                mappings[field] = matches
            checked.append({'exact_previous_hour_weather_feature_matches': mappings})
    else:
        assert 'Symbol' in frame and 'Symbol' not in features
        assert pd.to_datetime(frame[time], utc=True).dt.dayofweek.eq(4).all()
        stocks = tables['sp500_stocks.csv'].copy()
        stocks['Date'] = pd.to_datetime(stocks.Date, utc=True)
        stocks = stocks.dropna(subset=['Symbol','Date','Adj Close']).sort_values(['Symbol','Date'])
        assert stocks['Adj Close'].gt(0).all()
        group = stocks.groupby('Symbol', sort=False)
        future = group['Adj Close'].shift(-5)
        stocks['future_adjusted'] = future
        stocks['expected_target'] = (future / stocks['Adj Close'] - 1 <= -.05).astype(int)
        stocks['expected_end'] = group['Date'].shift(-5)
        keys = pd.MultiIndex.from_arrays([frame.Symbol, pd.to_datetime(frame[time], utc=True)], names=['Symbol','Date'])
        assert keys.is_unique
        expected = stocks.set_index(['Symbol','Date']).reindex(keys)
        assert expected.expected_end.notna().all() and expected.future_adjusted.notna().all() and expected['Adj Close'].notna().all()
        np.testing.assert_array_equal(frame[target].astype(int), expected.expected_target.astype(int))
        np.testing.assert_array_equal(pd.to_datetime(frame[config['label_end_column']], utc=True), expected.expected_end)
        modified = {key:value.copy() for key,value in tables.items()}
        stock = modified['sp500_stocks.csv']
        cutoff = pd.to_datetime(stock.Date, utc=True).sort_values().iloc[len(stock)//2]
        future_rows = pd.to_datetime(stock.Date, utc=True) >= cutoff
        numeric = [column for column in ['Adj Close','Close','Open','High','Low','Volume'] if column in stock]
        stock[numeric] = stock[numeric].astype(float)
        stock.loc[future_rows,numeric] *= 1.1
        rebuilt = build(modified, config)
        left = frame.loc[pd.to_datetime(frame[time], utc=True) < cutoff].set_index(['Symbol',time])[features].sort_index()
        right = rebuilt.loc[pd.to_datetime(rebuilt[time], utc=True) < cutoff].set_index(['Symbol',time])[features].sort_index()
        pd.testing.assert_frame_equal(left,right)
        checked.extend(['Friday-only unique decisions', 'independent five-observation adjusted-price labels and observation ends',
                        'future stock price/volume mutation cannot change past features or row eligibility'])
    result = {'domain':domain,'all_checked':True,'checks':checked,'rows':len(frame),'features':len(features)}
    write_json(folder / 'preparation_audit.json',result)
    return result
