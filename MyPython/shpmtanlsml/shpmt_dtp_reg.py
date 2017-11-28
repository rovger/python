import numpy as np
import math
import matplotlib.pyplot as plt
import xgboost
from sklearn import ensemble
from sklearn import linear_model
from sklearn import model_selection
from sklearn import metrics
from sklearn import tree
from sklearn.externals import joblib
from shpmtanlsml import shpmt_dtp_data
from shpmtanlsml import shpmt_dtp_preproc
from shpmtanlsml import shpmt_dtp_pmml_converter


def shpmt_dtp_linear():
    print('=================================================================')
    print('Linear Regression Models\n')
    X_train, y_train, n_samples, n_features, feature_names, target_name = shpmt_dtp_data.load_data('shpmt_dtp_a_d.csv')
    X_test, y_test, n_samples_test, n_features_test, feature_names_test, target_name_test = shpmt_dtp_data.load_data('shpmt_dtp_a_d_test.csv')
    X_ensemble = np.concatenate((X_train, X_test))
    # One-Hot Encoding
    X_encoded, feature_names = shpmt_dtp_preproc.encode_categorical_features(X_ensemble, feature_names, [0, 1, 2, 3])
    X_train = X_encoded[:n_samples]
    X_test = X_encoded[n_samples:]
    print('Features: %s' % feature_names)
    print('Number of features: %s' % len(feature_names))
    print('Number of training data: %i' % n_samples)
    print('Number of test data: %i' % n_samples_test)
    print('=================================================================')
    print('Linear Regression (Ordinary Least Squares) training...')
    ols = linear_model.LinearRegression(normalize=True).fit(X_train, y_train)
    print('=================================================================')
    print('Ridge Regression training...')
    ridge = linear_model.RidgeCV(normalize=True).fit(X_train, y_train)
    print('=================================================================')
    print('Lasso Regression training...')
    lasso = linear_model.LassoCV(normalize=True, verbose=1).fit(X_train, y_train)
    print('Number of Iteration: ', end=' ')
    print(lasso.n_iter_)
    print('=================================================================')

    model_names = ['Linear Regression - Ordinary Least Squares', 'Ridge Regression', 'Lasso Regression']

    plt.figure('RMSE')
    plt.title('Linear Regression Models')
    plt.xlabel('Test Samples')
    plt.ylabel('Root Mean Square Error')

    for i, model in enumerate((ols, ridge, lasso)):
        print('=================================================================')
        print(model_names[i])
        cv_scores = model_selection.cross_val_score(model, X_train, y_train, cv=3, n_jobs=3, verbose=2)
        y_pred = model.predict(X_test)
        test_score = model.score(X_test, y_test)
        accurate_count = 0
        d = np.zeros((n_samples_test, ), dtype=np.float)
        rmse = np.zeros((n_samples_test, ), dtype=np.float64)
        boundary = 2

        for j in range(0, n_samples_test):
            d[j] = y_test[j] - y_pred[j]
            rmse[j] = math.sqrt(metrics.mean_squared_error(y_test[:(j+1)], y_pred[:(j+1)]))
            # Consider the prediction result is accurate if the true time is no later than prediciton by 2 days
            if d[j] < boundary:
                accurate_count += 1

        accuracy = 100 * accurate_count / n_samples_test
        print('Independent Term: %0.4f' % model.intercept_)
        print('CV Score: ', end='')
        print(cv_scores)
        print('Test Score: ', end='')
        print(test_score)
        print('Accuracy (no later than %i days): %.2f%%' % (boundary, accuracy))
        print('Mean Absolute Error: %.4f' % metrics.mean_absolute_error(y_test, y_pred))
        print('Root Mean Square Error: %.4f' % math.sqrt(metrics.mean_squared_error(y_test, y_pred)))
        print('=================================================================')

        plt.plot(range(0, n_samples_test), rmse, label=model_names[i])
        plt.legend()

    plt.figure('Difference')
    plt.title('Difference')
    plt.xlabel('Test Samples')
    plt.ylabel('Shipment Delivery Time (Day)')
    y_pred = ridge.predict(X_test)
    for i in range(0, n_samples_test):
        diff = y_test[i] - y_pred[i]
        if diff <= 0:
            plt.vlines(i + 1, diff / 2, -diff / 2, colors='b')
        else:
            plt.vlines(i + 1, -diff / 2, diff / 2, colors='r')

    plt.show()


def shpmt_dtp_gbrt(model_file=None):
    print('=================================================================')
    print('Gradient Boosting Regressor\n')
    X_train, y_train, n_samples, n_features, feature_names, target_name = shpmt_dtp_data.load_data('shpmt_dtp_a_d.csv')
    X_test, y_test, n_samples_test, n_features_test, feature_names_test, target_name_test = shpmt_dtp_data.load_data('shpmt_dtp_a_d_test.csv')
    # X_ensemble = np.concatenate((X_train, X_test))
    # One-Hot Encoding
    # X_encoded, feature_names = shpmt_dtp_preproc.encode_categorical_features(X_ensemble, feature_names, [0, 1])
    # X_train = X_encoded[:n_samples]
    # X_test = X_encoded[n_samples:]
    print('Features: %s' % feature_names)
    print('Number of features: %s' % len(feature_names))
    print('Number of training data: %i' % n_samples)
    print('Number of test data: %i' % n_samples_test)
    print('=================================================================')
    if model_file is None:
        print('=================================================================')
        print('Gradient Boosting Regressor training...')
        model = ensemble.GradientBoostingRegressor(loss='ls', learning_rate=0.1, n_estimators=500,
                                                   min_samples_split=1000, min_samples_leaf=100,
                                                   max_depth=10, max_features='sqrt', verbose=2)
        model.fit(X_train, y_train)
        joblib.dump(model, 'GBRT.pkl')
        print('Training completed.')
        print('=================================================================')
    else:
        print('=================================================================')
        print('Gradient Boosting Regressor loading...')
        model = joblib.load(model_file)
        print('Model loaded.')
        print('=================================================================')

    """
    Print structure of trees in JAVA style and export graphs
    
    for i in range(1, 6):
        print('=================================================================')
        print('Tree %i of %i:' % (i, model.n_estimators))
        estimator = model.estimators_[i, 0]
        print_tree(estimator, feature_names)
        tree_dot = tree.export_graphviz(estimator, feature_names=feature_names, out_file=None)
        graph = pydotplus.graph_from_dot_data(tree_dot)
        graph.write_png("Tree_Graph_%i_of_%i.png" % (i, model.n_estimators))
        print('=================================================================')
    """

    cv_scores = model_selection.cross_val_score(model, X_train, y_train, cv=3, n_jobs=3, verbose=2)
    test_score = model.score(X_test, y_test)
    y_pred = model.predict(X_test)
    print('CV Score: ', end='')
    print(cv_scores)
    print('Test Score: ', end='')
    print(test_score)
    print("Mean Absolute Error: %.4f" % metrics.mean_absolute_error(y_test, y_pred))
    print("Root Mean Square Error: %.4f" % math.sqrt(metrics.mean_squared_error(y_test, y_pred)))
    print('=================================================================')

    test_score = np.zeros((model.n_estimators, ), dtype=np.float64)
    for i, y_pred in enumerate(model.staged_predict(X_test)):
        test_score[i] = model.loss_(y_test, y_pred)

    # Plot RMSE by iteration of estimators
    plt.figure('RMSE')
    plt.title('Root Mean Square Error')
    plt.xlabel('Boosting Iterations')
    plt.ylabel('Mean Square Error')
    plt.plot(np.arange(model.n_estimators) + 1, model.train_score_ ** 0.5, 'b-', label='Training Set MSE')
    plt.plot(np.arange(model.n_estimators) + 1, test_score ** 0.5, 'r-', label='Test Set MSE')
    plt.legend()

    """
    # Plot difference on test data
    plt.figure('Difference')
    plt.title('Difference')
    plt.xlabel('Test Samples')
    plt.ylabel('Shipment Delivery Time (Day)')
    for i in range(0, n_samples_test):
        diff = y_test[i] - y_pred[i]
        if diff <= 0:
            plt.vlines(i + 1, diff / 2, -diff / 2, colors='b')
        else:
            plt.vlines(i + 1, -diff / 2, diff / 2, colors='r')
    """

    # Plot deviation statistics on test data
    plt.figure('Deviation Statistics')
    plt.title('Deviation Statistics')
    plt.xlabel('Deviation Days')
    plt.ylabel('Deviation Counts')
    xvalue = [0]*20
    for i in range(0, n_samples_test):
        diff = int(round(y_pred[i] - y_test[i]))
        xvalue[diff+10] += 1

    plt.bar(list(range(-10, 10)), xvalue, 0.4, color="blue")
    plt.show()

    # Plot top 20 importante features
    feature_importance = model.feature_importances_
    sorted_idx = np.argsort(-feature_importance)[:20]

    plt.figure('Feature Importances')
    plt.title('Feature Importances')
    plt.xlabel('Features')
    plt.ylabel('Feature Importance Score')
    plt.bar(np.arange(sorted_idx.shape[0]), feature_importance[sorted_idx], alpha=.5)
    plt.xticks(np.arange(sorted_idx.shape[0]), feature_names[sorted_idx], rotation=30)

    plt.show()


def shpmt_dtp_xgboost(model_name='default_model', train_file=None, test_file=None, cv=False, pkl=False, pmml=False):
    print('=================================================================')
    print('XGBoost Regressor\n')
    X_train, y_train, n_samples, n_features, feature_names, target_name = shpmt_dtp_data.load_data(train_file)
    X_test, y_test, n_samples_test, n_features_test, feature_names_test, target_name_test = shpmt_dtp_data.load_data(test_file)
    print('Features: %s' % feature_names)
    print('Number of features: %s' % len(feature_names))
    print('Number of training data: %i' % n_samples)
    print('Number of test data: %i' % n_samples_test)
    print('=================================================================')

    pkl_file = model_name + ".pkl"
    if pkl:
        print('=================================================================')
        print('XGBoost Regressor loading...')
        model = joblib.load(pkl_file)
        print('Model loaded.')
        print('=================================================================')
    else:
        print('=================================================================')
        print('XGBoost Regressor training...')
        if model_name == 'dtp_a_d_model':
            # Optimized model for dtp_a_d
            model = xgboost.XGBRegressor(n_estimators=500, learning_rate=0.01, max_depth=10, min_child_weight=1, gamma=0,
                                         subsample=1, colsample_bytree=1,
                                         nthread=4, silent=False)
        elif model_name == 'dtp_d_i_model':
            # Optimized model for dtp_d_i
            model = xgboost.XGBRegressor(n_estimators=500, learning_rate=0.1, max_depth=5, min_child_weight=9, gamma=0.2,
                                         subsample=0.9, colsample_bytree=0.6,
                                         nthread=4, silent=False)
        else:
            # Default model
            model = xgboost.XGBRegressor()
        model.fit(X_train, y_train, verbose=True)
        joblib.dump(model, pkl_file)
        print('Training completed.')
        print('=================================================================')

    if cv:
        cv_scores = model_selection.cross_val_score(model, X_train, y_train, cv=3, n_jobs=3, verbose=2)
        print('CV Score: ', end='')
        print(cv_scores)
    test_score = model.score(X_test, y_test)
    y_pred = model.predict(X_test)
    print('Test Score: ', end='')
    print(test_score)
    print("Mean Absolute Error: %.4f" % metrics.mean_absolute_error(y_test, y_pred))
    print("Root Mean Square Error: %.4f" % math.sqrt(metrics.mean_squared_error(y_test, y_pred)))
    print('=================================================================')

    """
    Plot feature importance
    """
    xgboost.plot_importance(model)

    """
    Plot difference on test data
    """
    plt.figure('Difference')
    plt.title('Difference')
    plt.xlabel('Test Samples')
    plt.ylabel('Shipment Delivery Time (Day)')
    for i in range(0, n_samples_test):
        diff = y_test[i] - y_pred[i]
        if diff <= 0:
            plt.vlines(i + 1, diff / 2, -diff / 2, colors='b')
        else:
            plt.vlines(i + 1, -diff / 2, diff / 2, colors='r')

    plt.show()

    for i in range(0, n_samples_test):
        print('%f, %f' % (y_test[i], y_pred[i]))

    if pmml:
        pmml_file = model_name + '.pmml'
        shpmt_dtp_pmml_converter.convert_sklearn_to_pmml(model, pmml_file, feature_names=feature_names, target_name=target_name)


def print_tree(estimator, feature_names, node_id=0, depth=0):
    t = estimator.tree_
    indent = '    ' * depth
    left_child = t.children_left[node_id]
    right_child = t.children_right[node_id]
    if t.n_outputs == 1:
        value = t.value[node_id][0, :]
    else:
        value = t.value[node_id]

    if left_child == -1:  # define in sklearn.tree._tree.TREE_LEAF = -1
        print('%sleaf_val = %.17f;' % (indent, value))
    else:
        print(indent + 'if (feature[Features.%s.ordinal()] < %fF) {' % (feature_names[t.feature[node_id]], t.threshold[node_id]))
        print_tree(estimator, feature_names, node_id=left_child, depth=depth + 1)
        print('%s} else {' % indent)
        print_tree(estimator, feature_names, node_id=right_child, depth=depth + 1)
        print('%s}' % indent)


if __name__ == '__main__':
    # shpmt_dtp_linear()
    # shpmt_dtp_gbrt()
    shpmt_dtp_xgboost(model_name='dtp_d_i_model', train_file='shpmt_dtp_d_i.csv', test_file='shpmt_dtp_d_i_test.csv',
                      cv=True, pkl=False, pmml=True)
