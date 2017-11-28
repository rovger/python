import xgboost
from sklearn import ensemble
from sklearn import model_selection
from shpmtanlsml import shpmt_dtp_data


def shpmt_dtp_grid_search():
    print('=================================================================')
    print('Grid Search\n')
    X_train, y_train, n_samples, n_features, feature_names, target_name = shpmt_dtp_data.load_data('shpmt_dtp_d_i.csv')
    print('Features: %s' % feature_names)
    print('Number of features: %s' % len(feature_names))
    print('Number of training data: %i' % n_samples)
    print('=================================================================')

    param = {
        'subsample': [i/10.0 for i in range(6, 11)],
        'colsample_bytree': [i/10.0 for i in range(6, 11)]
    }
    # model = ensemble.GradientBoostingRegressor(loss='ls', learning_rate=0.1, n_estimators=500,
    #                                            min_samples_split=1000, min_samples_leaf=100, verbose=2)
    model = xgboost.XGBRegressor(n_estimators=500, learning_rate=0.1, max_depth=5, min_child_weight=9, gamma=0.2,
                                 subsample=0.9, colsample_bytree=0.6,
                                 nthread=4, silent=False)
    gsearch = model_selection.GridSearchCV(model, param_grid=param, cv=3, n_jobs=3, verbose=2)
    gsearch.fit(X_train, y_train)

    print("Best score: %0.4f" % gsearch.best_score_)
    print("Best parameters set:")
    best_parameters = gsearch.best_estimator_.get_params()
    for param_name in sorted(param.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))

if __name__ == '__main__':
    shpmt_dtp_grid_search()
