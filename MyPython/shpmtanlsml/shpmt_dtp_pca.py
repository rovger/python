from sklearn import decomposition
from sklearn import model_selection
from sklearn import metrics
from shpmtanlsml import shpmt_dtp_data
from shpmtanlsml import shpmt_dtp_pmml_converter
import math
import xgboost

def decompose(X, n_components=3):
    print('Components: %i\n' % n_components)
    pca = decomposition.PCA(n_components)
    X_decomposed = pca.fit_transform(X, n_components)
    return X_decomposed


def determine_n(X, target=0.95):
    print('=================================================================')
    print('PCA')
    pca = decomposition.PCA()
    pca.fit(X)
    print('PCA Components: %s' % pca.components_)
    print('PCA Variance Ratio: %s' % pca.explained_variance_ratio_)

    divergence_ratio = 0
    for i in range(0, len(pca.explained_variance_ratio_)):
        divergence_ratio += pca.explained_variance_ratio_[i]
        if divergence_ratio >= target:
            n_components = i + 1
            break
    print('PCA Components: %i' % n_components)
    print('PCA Divergence Ratio: %0.2f, Target: %0.2f' % (divergence_ratio, target))
    print('=================================================================')
    return n_components


def shpmt_pca():
    X_train, y_train, n_samples, n_features, feature_names, target_name = shpmt_dtp_data.load_data('shpmt_dtp_a_d.csv')
    X_test, y_test, n_samples_test, n_features_test, feature_names_test, target_name_test = shpmt_dtp_data.load_data(
        'shpmt_dtp_a_d_test.csv')

    determine_num=determine_n(X_train)
    X_train_decomposed=decompose(X_train, determine_num)
    print(X_train_decomposed)
    X_test_decomposed = decompose(X_test, determine_num)
    print(X_test_decomposed)

    print('XGBoost Regressor training...')
    model = xgboost.XGBRegressor(n_estimators=500, learning_rate=0.01, max_depth=10, min_child_weight=1, gamma=0,
                                 subsample=1, colsample_bytree=1,
                                 nthread=4, silent=False)
    model.fit(X_train_decomposed, y_train, verbose=True)

    shpmt_dtp_pmml_converter.convert_sklearn_to_pmml(model)

    cv_scores = model_selection.cross_val_score(model, X_train_decomposed, y_train, cv=3, n_jobs=3, verbose=2)
    test_score = model.score(X_test_decomposed, y_test)
    y_pred = model.predict(X_test_decomposed)
    print('CV Score: ', end='')
    print(cv_scores)
    print('Test Score: ', end='')
    print(test_score)
    print("Mean Absolute Error: %.4f" % metrics.mean_absolute_error(y_test, y_pred))
    print("Root Mean Square Error: %.4f" % math.sqrt(metrics.mean_squared_error(y_test, y_pred)))
    print('=================================================================')

if __name__ == '__main__':
    shpmt_pca()