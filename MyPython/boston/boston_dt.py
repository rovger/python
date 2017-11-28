import matplotlib.pyplot as plt
import numpy
import sklearn.datasets as datasets
import sklearn.tree as tree


def boston_dt(x=datasets.load_boston()['data'], y=datasets.load_boston()['target']):
    print('Decision Tree Regression\n')
    model = tree.DecisionTreeRegressor()
    num_train = 500
    num_val = 6
    x_train_ones = numpy.ones((num_train, 1))
    x_train = numpy.column_stack((x_train_ones, numpy.array(x[0:num_train])))
    y_train = y[0:num_train]
    x_val_ones = numpy.ones((num_val, 1))
    x_val = numpy.column_stack((x_val_ones, numpy.array(x[num_train:num_train+num_val])))
    y_val = y[num_train:num_train+num_val]
    print('Number of training data: %i' % len(x_train))
    print('Number of validation data: %i' % len(x_val))

    model.fit(x_train, y_train)
    variance = 0

    for i in range(0, num_val):
        x = x_val[i]
        y = y_val[i]
        hypo = model.predict([x])[0]
        print('\n%i.' % (i + 1))
        print('X: %s\nHypothesis: %s \ny: %s\nVariance: %s' % (x, hypo, y, (hypo - y) ** 2))
        variance += (hypo - y_val[i]) ** 2
        plt.scatter(i + 1, hypo, c='b')
        plt.scatter(i + 1, y, c='g')

    mean_variance = variance/num_val
    print('\nMean Variance: %s\n' % mean_variance)
    plt.show()

boston_dt()
