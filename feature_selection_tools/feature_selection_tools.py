import pandas as pd
import numpy as np
import itertools
from IPython.display import clear_output
from scipy.stats import t

def isolated_features_analysis(data, evaluator, features, verbose=False):
    stats = {}
    metric_full = evaluator(data, features)
    for feature in features:
        if verbose:
            clear_output(wait=True)
            print(f'Analysing {feature}')
        other_features = list(features)
        other_features.remove(feature)
        metric_without =  evaluator(data, other_features)
        metric_alone = evaluator(data, [feature])
        impact = (metric_full - metric_without) / metric_without
        stats[feature] = {'metric': metric_alone,
                          'impact': impact}
    df_stats = pd.DataFrame(stats).T.sort_values('metric')
    if verbose:
        clear_output(wait=True)
        print(f'Isolated analysis completed!')
    return df_stats

def full_features_analysis(data, evaluator, features, verbose=False):
    df_stats = isolated_features_analysis(data, evaluator, features, verbose)
    impacts = {}
    remaining_features = list(features)
    while remaining_features:
        feature = remaining_features[0]
        if verbose:
            clear_output(wait=True)
            print(f'Analysing {feature} combinations')
        for pair_feature in remaining_features:
            if feature != pair_feature:
                combination = [feature, pair_feature]
                metric = evaluator(data, combination)
                
                impacts[feature] = impacts.get(feature, [])
                impact = (metric - df_stats.loc[pair_feature, 'metric']) / df_stats.loc[pair_feature, 'metric']
                impacts[feature].append(impact)
                
                impacts[pair_feature] = impacts.get(pair_feature, [])
                impact = (metric - df_stats.loc[feature, 'metric']) / df_stats.loc[feature, 'metric']
                impacts[pair_feature].append(impact)
        remaining_features.remove(feature)
    for feature in features:
        df_stats.loc[feature, 'mean'] = np.mean(impacts[feature])
        df_stats.loc[feature, 'std'] = np.std(impacts[feature])
        df_stats.loc[feature, 'min'] = np.min(impacts[feature])
        df_stats.loc[feature, 'max'] = np.max(impacts[feature])
    df_stats.columns = pd.MultiIndex.from_arrays([['isolated analysis', 'isolated analysis', 'pairwise analysis (impact)', 
                                                   'pairwise analysis (impact)', 'pairwise analysis (impact)', 'pairwise analysis (impact)'],
                                                  ['metric', 'impact', 'mean', 'std', 'min', 'max']])
    if verbose:
        clear_output(wait=True)
        print(f'Full analysis completed!')
    return df_stats

def feature_analyse(data, feature, features, evaluator, stop_error=0.01, alpha=0.95, verbose=False):
    alpha_adj = 1 - ((1 - alpha) / 2)
    
    other_features = list(features)
    other_features.remove(feature)
    impacts = []
    for i in range(len(other_features), 0, -1):
        combinations = list(itertools.combinations(other_features, i))
        for combination in combinations:
            combination_full = list(combination)
            combination_full.append(feature)
            
            metric_full = evaluator(data, combination_full)
            metric_without = evaluator(data, list(combination))
            
            impact = (metric_full - metric_without) / metric_without
            impacts.append(impact)
            
            n = len(impacts)
            ts = t.ppf(alpha_adj, n)
            s = np.std(impacts)
            error = ts * s / np.sqrt(n)
            x = np.mean(impacts)
            min_CI = x - error
            max_CI = x + error            
            
            if ((error < stop_error) or ((error / abs(x)) < stop_error)) and n > 1:
                break
        if ((error < stop_error) or ((error / abs(x)) < stop_error)) and n > 1:
            break
            
    stats = {'mean': x,
             'min': min_CI,
             'max': max_CI,
             'error': error,
             'n': n,
             'std': s,
             'alpha': alpha}
    return stats

def features_selector(data, features, evaluator, discart_limit, verbose=False):
    remaining_features = list(features)
    stats = {}
    last_discated_impact = float('-inf')
    while True:
        discart_feature, discart_impact = '', float('-inf')
        for feature in remaining_features: 
            stats[feature] = stats.get(feature, {})
            last_impact = stats[feature].get('mean', 0)
            if verbose:
                clear_output(wait=True)
                print(f'Analysing {feature}. Remaining {len(remaining_features)}')
            
            if (last_impact + np.abs(last_discated_impact)) > discart_impact:
                other_features = list(remaining_features)
                other_features.remove(feature)
                metric_without = evaluator(data, other_features)
                metric_full = evaluator(data, remaining_features)
                impact = (metric_full - metric_without) / metric_without

                stats[feature]['action'] = 'kept'
                stats[feature]['mean'] = impact

                if (impact > discart_limit) and (impact > discart_impact):
                    discart_feature = feature
                    discart_impact = impact                        

        if discart_feature:
            if verbose:
                clear_output(wait=True)
                print(f'Descarting {discart_feature}')
            stats[discart_feature] = feature_analyse(data, discart_feature, remaining_features, evaluator)
            stats[discart_feature]['action'] = 'discarted'
            remaining_features.remove(discart_feature)
            remaining_features = [feature for feature, stats in sorted(stats.items(), key=lambda stat: stat[1]['mean'], reverse=True) if feature in remaining_features]
            last_discated_impact = discart_impact
        else: 
            break
    for feature in remaining_features:
        if verbose:
            clear_output(wait=True)
            print(f'Analysing stats for {feature}')
        stats[feature] = feature_analyse(data, feature, remaining_features, evaluator)
        stats[feature]['action'] = 'kept'
        
    if verbose:
        clear_output(wait=True)
        print(f'Features selection completed!')
    df_stats = pd.DataFrame(stats).T.sort_values('mean')
    df_stats = df_stats[['action', 'mean', 'min', 'max', 'error', 'n', 'std']]
    return df_stats

def evaluate_feature_combinations(data, evaluator, combine, fixed=[], verbose=False):
    combinations = combinations_generator(combine, fixed, verbose)
    results = []
    for k, combination in enumerate(combinations):
        if verbose:
            clear_output(wait=True)
            print(f'Evaluating {k + 1} of {len(combinations)} combinations.')
        results.append({'combination': combination,
                        'n': len(combination),
                        'metric': evaluator(data, combination)})
    if verbose:
        clear_output(wait=True)
        print(f'Features combinations evaluated!')

    results = pd.DataFrame(results)
    results = results.sort_values('metric').reset_index()
    results['var_from_best'] = (results['metric'] - results.loc[0, 'metric']) / results.loc[0, 'metric']
    return results

def combinations_generator(combine, fixed=None, verbose=False):
    combinations_list = []        
    for i in range(len(combine), 0, -1):
        if verbose:
            clear_output(wait=True)
            print(f'Creating combinations with {i} items (total items: {len(combine) + 1}).')
        combinations = list(itertools.combinations(combine, i))
        for combination in combinations:
            if fixed is not None:
                full_combination = list(combination) + fixed
            else:
                full_combination = list(combination)
            combinations_list.append(full_combination)
    if verbose:
        clear_output(wait=True)
        print(f'Sums of cross combinations finished.')
    return combinations_list

def pairwise_score_matrix(X_train, X_test, y_train, y_test, model, score_function, verbose=False):
    features = list(X_train.columns)
    df_result = pd.DataFrame(0, index=features, columns=features)
    while features:
        if verbose:
            clear_output(wait=True)
            print(f'Remmaining {len(features)} features. Analysing "{features[0]}".')
        main_feature = features[0]
        for feature in features:
            if feature == main_feature:
                model.fit(X_train[[main_feature]], y_train)
                y_pred = model.predict(X_test[[main_feature]])
            else:
                model.fit(X_train[[main_feature, feature]], y_train)
                y_pred = model.predict(X_test[[main_feature, feature]])    
            score = score_function(y_test, y_pred)
            df_result.loc[main_feature, feature] = score
            df_result.loc[feature, main_feature] = score
        features.pop(0)
    if verbose:
        clear_output(wait=True)
        print(f'Pairwise score matrix created!')   
    return df_result

def main_diagonal_difference_matrix(matrix):
    features = list(matrix.columns)
    df_result = pd.DataFrame(0, index=features, columns=features)
    for row in features:
        base_score = matrix.loc[row, row]
        for column in features:
            df_result.loc[row, column] = (matrix.loc[row, column] - base_score) if column != row else 0
    return df_result

def main_diagonal_proportion_matrix(matrix):
    features = list(matrix.columns)
    df_result = pd.DataFrame(0, index=features, columns=features)
    for row in features:
        base_score = matrix.loc[row, row]
        for column in features:
            df_result.loc[row, column] = 100 * (matrix.loc[row, column] - base_score) / base_score if column != row else base_score
    return df_result

def symmetrical_elements_proportion_matrix(matrix):
    features = list(matrix.columns)
    df_result = pd.DataFrame(0, index=features, columns=features)
    while features:
        main_feature = features[0]
        for feature in features:
            if feature == main_feature:
                df_result.loc[main_feature, feature] = 1
            else:
                total = matrix.loc[main_feature, feature] + matrix.loc[feature, main_feature]
                df_result.loc[main_feature, feature] = matrix.loc[main_feature, feature] / total
                df_result.loc[feature, main_feature] = matrix.loc[feature, main_feature] / total
        features.pop(0)
    return df_result
    
def worst_potential_improvement(scores, n):
    def n_smallests(x):
        result = x.nlargest(n)
        result = result.round(2).astype(str)  + ' | ' +  result.index
        result.index = range(0, n)
        return result
    
    mddm = main_diagonal_difference_matrix(scores)
    
    index_names = list(mddm.columns)
    for i, name in enumerate(index_names):
        abbreviation = ''
        name_pieces = name.split('_')
        for piece in name_pieces:
            abbreviation += piece[0]
        if len(name_pieces) == 1:
            abbreviation += name[1] + name[-1]
        index_names[i] = abbreviation
        
    mddm.columns = index_names
        
    return mddm.apply(n_smallests, axis=1).T