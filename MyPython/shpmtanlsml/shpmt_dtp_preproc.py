import numpy as np
from sklearn.preprocessing import OneHotEncoder


def encode_categorical_features(X, feature_names, column_indices):
    # Data Preprocessing using OneHotEncoder
    index_end = 0
    X_encoded = X
    for i in range(0, len(column_indices)):
        if i == 0:
            column_index = index_end + column_indices[i]
        else:
            column_index = index_end + column_indices[i] - column_indices[i-1]
        enc = OneHotEncoder(categorical_features=[column_index])
        X_encoded = enc.fit_transform(X_encoded).toarray()
        print('One-Hot n-values: %s' % enc.n_values_)
        print('One-Hot feature-indices: %s' % enc.feature_indices_)
        print('One-Hot active-features: %s' % enc.active_features_)

        feature_name_target = feature_names[column_index]
        feature_names_unchanged_prev = feature_names[:column_index]
        feature_names_unchanged_next = feature_names[column_index + 1:]
        feature_names_expanded = []
        for j in range(0, len(enc.active_features_)):
            feature_names_expanded.append(feature_name_target + "_" + str(enc.active_features_[j]))
        feature_names = np.append(feature_names_unchanged_prev, feature_names_expanded)
        index_end = len(feature_names) - 1
        feature_names = np.append(feature_names, feature_names_unchanged_next)

    return X_encoded, feature_names
