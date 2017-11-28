import csv
import numpy as np


def load_data(fname):
    fpath = 'data\\' + fname
    with open(fpath) as f:
        data_file = csv.reader(f)
        temp = next(data_file)
        n_samples = int(temp[0])
        n_features = int(temp[1])
        data = np.empty((n_samples, n_features))
        target = np.empty((n_samples, ))
        temp = next(data_file)  # Names of features
        feature_names = np.array(temp)[:-1]
        target_name = temp[-1]

        for i, d in enumerate(data_file):
            data[i] = np.asarray(d[:-1], dtype=np.float64)
            target[i] = np.asarray(d[-1], dtype=np.float64)

    return data, target, n_samples, n_features, feature_names, target_name
