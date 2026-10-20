"""Frozen development protocols and simple registered control models."""

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, StratifiedKFold, KFold


def development_splits(frame, y, config):
    count = int(config.get('validation_folds', 3))
    if count not in range(2, 6):
        raise ValueError('Use two to five development folds')
    strategy, seed = config['split'], int(config.get('seed', 42))
    if strategy == 'temporal':
        times = pd.to_datetime(frame[config['split_column']], utc=True)
        unique = np.sort(times.unique())
        width = max(1, int(len(unique) * min(float(config.get('validation_fraction', .2)), 1 / (count + 1))))
        if len(unique) - count * width < 2:
            raise ValueError('Not enough distinct timestamps for expanding validation')
        ends = pd.to_datetime(frame[config['label_end_column']], utc=True) if config.get('label_end_column') else times
        splits = []
        for fold in range(count):
            start = len(unique) - (count - fold) * width
            boundary = unique[start]
            stop = unique[start + width] if start + width < len(unique) else None
            train = np.flatnonzero((times < boundary) & (ends < boundary))
            valid = (times >= boundary) & ((times < stop) if stop is not None else True)
            splits.append((train, np.flatnonzero(valid)))
    elif strategy == 'group':
        splits = list(GroupKFold(count).split(frame, y, frame[config['split_column']]))
    else:
        splitter = StratifiedKFold(count, shuffle=True, random_state=seed) if config['task'] == 'classification' else KFold(count, shuffle=True, random_state=seed)
        splits = list(splitter.split(frame, y))
    for train, valid in splits:
        if len(train) < 2 or len(valid) < 2 or set(train) & set(valid):
            raise ValueError('Invalid development fold')
        if config['task'] == 'classification' and set(y[train]) != set(y):
            raise ValueError('A development training fold is missing a class')
    return splits


def control_model(task):
    return 'constant_classifier' if task == 'classification' else 'constant_regressor'
