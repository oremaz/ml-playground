"""Development-only error and stability measurements for frozen folds."""

from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from ml_pipeline.benchmark_agent import read_json, write_json
from ml_pipeline.playground_runtime import prepare_dataset, score_predictions, split_indices


def diagnose(folder, state, trial):
    root = Path(folder) / f"trial_{trial['number']}" / 'output'
    frame = joblib.load(root.parent / 'input.joblib')
    config = trial['config']
    X, y, _ = prepare_dataset(frame, config)
    splits = split_indices(frame, y, config)
    rows = pd.read_csv(root / 'fold_metrics.csv')
    diagnostics = []
    for (name, augmentation), records in rows.groupby(['model', 'augmentation'], sort=False):
        predictions = pd.concat([pd.read_csv(root / f'predictions_{row.checkpoint_id}.csv').assign(fold=int(row.fold))
                                 for row in records.itertuples()], ignore_index=True)
        true, predicted = predictions.y_true.to_numpy(), predictions.y_pred.to_numpy()
        item = {'model': name, 'augmentation': augmentation,
                'validation_folds': [{key: row[key] for key in ['fold', config['metric'], 'reference_' + config['metric']]}
                                     for row in records.to_dict('records')],
                'primary_score_std_across_folds': float(records[config['metric']].std(ddof=0)),
                'validation_observations': len(predictions)}
        last = records.sort_values('fold').iloc[-1]
        train = splits[int(last.fold) - 1][0]
        transformer = joblib.load(root / f'preprocessor_fold_{int(last.fold)}.joblib')
        model = joblib.load(root / f'{last.checkpoint_id}.pkl')
        train_X = np.asarray(transformer.transform(X.iloc[train]), dtype=np.float32)
        train_p = model.predict_proba(train_X) if config['task'] == 'classification' else None
        item.update(training_scores=score_predictions(y[train], model.predict(train_X), train_p, config['task']),
                    training_rows=len(train), training_scope='Last frozen development fold only; in-sample.')
        if config['task'] == 'classification':
            labels = list(read_json(root / 'labels.json').values())
            p, r, f, support = precision_recall_fscore_support(true, predicted, labels=np.arange(len(labels)), zero_division=0)
            item['per_class'] = [{'label': label, 'precision': float(a), 'recall': float(b), 'f1': float(c), 'rows': int(n)}
                                 for label, a, b, c, n in zip(labels, p, r, f, support)]
            item['confusion_matrix'] = {'label_order': labels, 'rows_are_true': True,
                                        'counts': confusion_matrix(true, predicted, labels=np.arange(len(labels))).tolist()}
            if len(labels) == 2:
                ordered = np.argsort(-predictions.probability_1.to_numpy(), kind='stable')[:max(1, int(np.ceil(len(true) * .1)))]
                item.update(positive_prevalence=float(true.mean()), precision_top_10_percent=float(true[ordered].mean()),
                            recall_top_10_percent=float(true[ordered].sum() / max(1, true.sum())))
        else:
            residual = predicted - true
            item['residuals'] = {'mean_error': float(residual.mean()), 'absolute_error_p50': float(np.quantile(abs(residual), .5)),
                                 'absolute_error_p90': float(np.quantile(abs(residual), .9)),
                                 'absolute_error_p99': float(np.quantile(abs(residual), .99)),
                                 'observed_mean': float(true.mean()), 'predicted_mean': float(predicted.mean())}
            largest = np.argsort(-abs(residual), kind='stable')[:5]
            item['large_error_examples'] = [{'row': int(predictions.iloc[i].row), 'fold': int(predictions.iloc[i].fold),
                                             'observed': float(true[i]), 'predicted': float(predicted[i]),
                                             'inputs': json.loads(frame.iloc[[int(predictions.iloc[i].row)]][config['features'][:6]].to_json(orient='records', date_format='iso'))[0]}
                                            for i in largest]
            numeric = X.iloc[predictions.row].select_dtypes(include='number').reset_index(drop=True)
            associations = numeric.corrwith(pd.Series(abs(residual))).dropna().abs().nlargest(3)
            slices = []
            for column in associations.index:
                groups = pd.qcut(numeric[column], 3, duplicates='drop')
                for label in groups.cat.categories:
                    mask = (groups == label).to_numpy()
                    if mask.any():
                        slices.append({'feature': column, 'interval': str(label), 'rows': int(mask.sum()),
                                       'rmse': float(np.sqrt(np.mean(residual[mask] ** 2))), 'mean_error': float(residual[mask].mean())})
            item['exploratory_error_slices'] = slices
        from sklearn.multioutput import MultiOutputRegressor
        estimator = model.estimators_[0] if isinstance(model, MultiOutputRegressor) else model
        importance = getattr(estimator, 'feature_importances_', None)
        if importance is None and hasattr(estimator, 'coef_'):
            importance = np.abs(np.atleast_2d(estimator.coef_)).mean(axis=0)
        if importance is not None:
            item['top_feature_importances'] = sorted(
                [{'feature': str(key), 'importance': float(value)} for key,value in zip(transformer.get_feature_names_out(), importance)],
                key=lambda entry: entry['importance'], reverse=True)[:8]
            item['importance_scope'] = 'Last fold; impurity importance or absolute standardized coefficient. Association only.'
        diagnostics.append(item)
    result = {'trial': trial['number'], 'diagnostics': diagnostics,
              'caution': 'Selection uses the mean primary score across frozen folds. Diagnostics pool development validation predictions. '
                         'Error slices are exploratory associations, not causes or significance tests. Final rows are inaccessible.'}
    write_json(root.parent / 'diagnostics.json', result)
    return result
