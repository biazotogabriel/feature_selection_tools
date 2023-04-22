def get_features_samples(features, fixed, n_min, n_max):   
    available = set(features)    
    available.difference_update(fixed)
    available = list(available)
    features_size = random.randint(n_min, n_max)
    without = random.sample(available, features_size-len(fixed))
    _with = without.copy()
    _with[0:0] = fixed
    return without, _with

X = data[features]
y = data['sales']
time_series = data['date']
tscv = Time_series_CV(time_series, offset=timedelta(days=60), k_folds=1, test_size=timedelta(days=60))

begin = tscv.folds[1]['begin']
split = tscv.folds[1]['split']
end = tscv.folds[1]['end']

X_train = X[(time_series >= begin) & (time_series < split)]
X_test = X[(time_series >= split) & (time_series < end)]
y_train = y[(time_series >= begin) & (time_series < split)]
y_test = y[(time_series >= split) & (time_series < end)]

lr = LinearRegression()

features_comb = [['store'],
                 ['promo'],
                 ['state_holiday'],
                 ['school_holiday'],
                 ['store_type'],
                 ['assortment'],
                 ['competition_distance'],
                 ['promo2'],
                 ['competition_time'],
                 ['promo2_distance'],
                 ['year'],
                 ['month'],
                 ['day'],
                 ['week'],
                 ['week_day'],
                 ['month_sin', 'month_cos'],
                 ['day_sin', 'day_cos'],
                 ['week_day_sin', 'week_day_cos'],
                 ['week_sin', 'week_cos']]

score = {}
count = 0
for feature in features_comb:
    delta_scores = []
    mean_score_vars = []
    tests_limit = 25
    delta_mean_min_lim = 0.001
    i = 0
    decrease_mean = 1
    while (i < tests_limit) and (decrease_mean > delta_mean_min_lim): 
        count += 1
        
        samples_without, samples = get_features_samples(features, feature, 10, 10)

        X_train_l = X_train[samples]
        X_test_l = X_test[samples]
        lr.fit(X_train_l, y_train)
        y_pred = lr.predict(X_test_l)
        score_with = mt.mean_absolute_error(np.expm1(y_test), np.expm1(y_pred))

        X_train_l = X_train[samples_without]
        X_test_l = X_test[samples_without]
        lr.fit(X_train_l, y_train)
        y_pred = lr.predict(X_test_l)
        score_without = mt.mean_absolute_error(np.expm1(y_test), np.expm1(y_pred))


        if delta_scores:
            old_mean_score = np.mean(delta_scores)
        else:
            old_mean_score = 0
        delta_score = score_with - score_without
        delta_scores.append(delta_score)
        new_mean_score = np.mean(delta_scores)
        if old_mean_score:
            mean_score_var = (new_mean_score - old_mean_score) / old_mean_score
            mean_score_vars.append(np.abs(mean_score_var))

        if len(mean_score_vars) >= 5:
            decrease_mean = np.mean(mean_score_vars[-5:])
        else: 
            decrease_mean = 1
        i += 1
        
    print(feature)
    score[feature[0]] = np.mean(delta_scores)


