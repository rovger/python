import numpy as np
import matplotlib.pyplot as plt
from sklearn import ensemble, datasets
from sklearn.model_selection import cross_val_score

iris = datasets.load_iris()
X = iris.data
y = iris.target

clf = ensemble.GradientBoostingClassifier()

print('=================================================================')
print('Gradient Boosting Classifier\n')
scores = cross_val_score(clf, X, y)
print("Scores: ", end='')
print(scores)
print('Accuracy: %0.2f (+/- %0.2f)' % (scores.mean(), scores.std() * 2))
print('=================================================================')

clf.fit(X, y)
feature_importance = clf.feature_importances_
# make importances relative to max importance
feature_importance = 100.0 * (feature_importance / feature_importance.max())
sorted_idx = np.argsort(feature_importance)
pos = np.arange(sorted_idx.shape[0]) + .5
plt.subplot(1, 2, 2)
plt.barh(pos, feature_importance[sorted_idx], align='center')
plt.yticks(pos, iris.feature_names)
plt.xlabel('Relative Importance')
plt.title('Variable Importance')
plt.show()
