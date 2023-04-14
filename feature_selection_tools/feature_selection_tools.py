import pandas as pd
import numpy as np
import itertools
from IPython.display import clear_output


def metric_decreases_by_pair(X_train, X_test, y_train, y_test, model, metric, verbose=True):
    dic_result = {}
    features = list(X_train.columns)
    for i, feature in enumerate(features):
        if verbose:
            clear_output(wait=True)
            print(f'Analysing ({i + 1} of {len(features)}) {feature}')
        feature_stats = {}
        delta_scores = []
        for feature_pair in features:    
            X_train_inner = X_train[[feature, feature_pair]]
            X_test_inner = X_test[[feature, feature_pair]]
            model.fit(X_train_inner, y_train)
            y_pred = model.predict(X_test_inner)
            score_pair = metric(y_test, y_pred)

            X_train_inner = X_train[[feature]]
            X_test_inner = X_test[[feature]]
            model.fit(X_train_inner, y_train)
            y_pred = model.predict(X_test_inner)
            score_alone = metric(y_test, y_pred)

            delta_score = score_pair - score_alone
            feature_stats[feature_pair] = delta_score
            delta_scores.append(delta_score)
        feature_stats['metric'] = score_alone
        feature_stats['suffered_decrease_mean'] = np.mean(delta_scores)
        feature_stats['suffered_decrease_std'] = np.std(delta_scores)        
        dic_result[feature] = feature_stats
    result = pd.DataFrame(dic_result).T
    result['caused_decrease_mean'] = result.apply(lambda column: column[column.index != column.name].mean(), axis=0)
    result['caused_decrease_std'] = result.apply(lambda column: column[column.index != column.name].std(), axis=0)
    return result

def sum_cross_combinations(data, fixed, combine=None, verbose=False):
    sums = []
    
    if combine is None:
        combine = list(set(data.columns) - set(fixed))
        
    for i in range(1, len(combine) + 1):
        if verbose:
            clear_output(wait=True)
            print(f'Combinations taken {i} by {i}.')
        combinations = list(itertools.combinations(combine, i))
        for combination in combinations:
            full_combination = list(combination) + fixed
            _sum = data.loc[list(full_combination), list(full_combination)].sum().sum()
            sums.append((full_combination, _sum))
    sums.sort(key=lambda x: x[1])
    return sums