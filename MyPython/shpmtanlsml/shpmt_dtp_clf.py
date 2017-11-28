import numpy as np
from sklearn import ensemble
from sklearn import naive_bayes
from sklearn import tree
from sklearn.model_selection import cross_val_score
from shpmtanlsml import shpmt_dtp_data
from shpmtanlsml import shpmt_dtp_preproc


def shpmt_dte_clf():
    X, y, n_samples, n_features, feature_names = shpmt_dtp_data.load_data('shpmt_dte_clf.csv')
    X_test, y_test, n_samples_test, n_features_test, feature_names_test = shpmt_dtp_data.load_data('shpmt_dte_clf_test.csv')
    '''
    X_ensemble = np.concatenate((X, X_test))
    # Encode ZIP_FROM
    X_encoded, feature_names, index_end = shpmt_dte_preproc.encode_categorical_features(X_ensemble, feature_names, 0)
    # Encode ZIP_TO
    X_encoded, feature_names, index_end = shpmt_dte_preproc.encode_categorical_features(X_encoded, feature_names, index_end + 1)
    # Encode ISC
    X_encoded, feature_names, index_end = shpmt_dte_preproc.encode_categorical_features(X_encoded, feature_names, index_end + 1)
    X = X_encoded[:n_samples]
    X_test = X_encoded[n_samples:]
    '''
    print('=================================================================')
    print('Desicion Tree\n')
    print('Features: %s' % feature_names)
    print('Number of training data: %i' % len(X))
    print('Number of test data: %i' % len(X_test))
    print('=================================================================')

    print('=================================================================')
    print('Gaussian Naive Bayes Classifier training...')
    gnb = naive_bayes.GaussianNB().fit(X, y)
    print('=================================================================')
    print('Decision Tree Classifier training...')
    dt = tree.DecisionTreeClassifier().fit(X, y)
    print('=================================================================')
    print('Gradient Boosting Classifier training...')
    gb = ensemble.GradientBoostingClassifier().fit(X, y)
    print('=================================================================')

    # Title for the plots
    titles = ['Gaussian Naive Bayes Classifier',
              'Decision Tree Classifier',
              'Gradient Boosting Classifier']

    for i, clf in enumerate((gnb, dt, gb)):
        print('=================================================================')
        print(titles[i])
        scores = cross_val_score(clf, X, y)
        pred = clf.predict(X_test)
        accurate_count = 0
        for j in range(0, n_samples_test):
            if int(pred[j]) == int(y_test[j]):
                accurate_count += 1
        accuracy = 100 * accurate_count / n_samples_test
        print('Scores: ', end='')
        print(scores)
        print('CV Accuracy: %0.2f (+/- %0.2f)' % (scores.mean(), scores.std() * 2))
        print('Prediction Accuracy: %0.2f%%' % accuracy)
        print('=================================================================')

shpmt_dte_clf()
