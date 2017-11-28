from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split # 注意，这个模块从0.20开始不存在于 sklearn.cross_validation了!
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from numpy import *

# step1: load iris data
iris = load_iris()
X = iris.data
y = iris.target
# print(X)
# print(y)

# step2: split X and y int traning and testing sets
train, test, t_train, t_test = train_test_split(X, y, test_size=0.3, random_state=4)

# step3: train the model on the training set
logreg = LogisticRegression()
logreg.fit(train, t_train)

# step4: make predictions on the testing set
predict = logreg.predict(test)
print(predict)
print('=========================')
print(t_test)
print('=========================')
print(metrics.accuracy_score(predict, t_test))

# demo training resault, select the best K value
k_range = list(range(1, 30))
scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(train, t_train)
    y_pred = knn.predict(test)
    scores.append(metrics.accuracy_score(y_pred, t_test))
print(scores)

# 语法测试
arr1 = ones(5, int8)
arr2 = empty(5, int8)
print('arr1', arr1, 'arr2', arr2)