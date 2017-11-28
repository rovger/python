import csv
import numpy as np
import xgboost as xgb
from sklearn2pmml import *
from sklearn import ensemble
import matplotlib.pyplot as plt


def data_analysis(train_file, test_file):
    x_train, y_train, n_samples, n_features, feature_names, target_name = load_data(train_file)
    x_test, y_test, n_samples_test, n_features_test, feature_names_test, target_name_test = load_data(test_file)
    # print('target_name: %s' % target_name)
    '''
    print('-----------------start: train data----------------')
    print('Features: %s' % feature_names)
    print('Number of features: %s' % len(feature_names))
    print('Number of training data: %i' % n_samples)
    print('test data %s' % x_test)
    print('-----------------end: train data----------------')
    '''
    # model = xgb.XGBRegressor()
    # model = xgb.XGBClassifier()
    # model = ensemble.GradientBoostingRegressor()
    model = ensemble.GradientBoostingClassifier()
    model.fit(x_train, y_train)

    pipeline = PMMLPipeline([("classifier", model)])
    pipeline.active_fields = np.array(feature_names)
    pipeline.target_field = target_name
    sklearn2pmml(pipeline, "training_result_nba.pmml", with_repr=True, debug=True)

    # test score
    test_scrore = model.score(x_test, y_test)
    print('test score', test_scrore)

    """
    # test difference
    y_pred = model.predict(x_test)
    plt.figure('Difference')
    plt.title('Difference')
    plt.xlabel('Train/Test data')
    plt.ylabel('NBA Position')
    for i in range(0, n_samples_test):
        diff = y_test[i] - y_train[i]
        if diff != 0:
            plt.vlines(i + 1, diff / 2, -diff / 2, colors='r')
    """

    # features inportance
    feature_importance = model.feature_importances_
    sorted_idx = np.argsort(-feature_importance)
    plt.figure('Feature Importances')
    plt.title('Feature Importances')
    plt.xlabel('Features')
    plt.ylabel('Feature Importance Score')
    plt.bar(np.arange(sorted_idx.shape[0]), feature_importance[sorted_idx], alpha=.5)
    feature_str = ','.join(feature_names)
    plt.xticks(np.arange(sorted_idx.shape[0]), feature_str.split(','), rotation=30)
    plt.show()


def load_data(file_name):
    file_path = 'data\\' + file_name
    with open(file_path) as f:
        data_file = csv.reader(f)
        temp = next(data_file)
        n_samples = int(temp[0])
        n_features = int(temp[1])
        data = np.empty((n_samples, n_features))
        target = np.empty((n_samples, ))
        temp = next(data_file)
        feature_names = temp[:-2]
        target_name = temp[-2]
        for i, d in enumerate(data_file):
            data[i] = np.asarray(d[:-2], dtype=int)
            target[i] = np.asarray(d[-2], dtype=int)
    return data, target, n_samples, n_features, feature_names, target_name


if __name__ == '__main__':
    data_analysis(train_file='train_data_nba.csv', test_file='test_data_nba.csv')