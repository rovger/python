from sklearn import tree, datasets
from sklearn.model_selection import cross_val_score

iris = datasets.load_iris()
X = iris.data
y = iris.target

clf = tree.DecisionTreeClassifier()

print('=================================================================')
print('Gradient Boosting Classifier\n')
scores = cross_val_score(clf, X, y)
print("Scores: ", end='')
print(scores)
print('Accuracy: %0.2f (+/- %0.2f)' % (scores.mean(), scores.std() * 2))
print('=================================================================')
