import numpy as np
import xgboost
from sklearn import datasets
from sklearn2pmml import sklearn2pmml
from sklearn2pmml import PMMLPipeline

iris = datasets.load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names
model = xgboost.XGBClassifier(learning_rate=0.1, n_estimators=10, max_depth=10, silent=False)

iris_pipeline = PMMLPipeline([("classifier", model)])
iris_pipeline.active_fields = np.array(feature_names)
iris_pipeline.fit(X, y)

sklearn2pmml(iris_pipeline, "iris.pmml", with_repr=True, debug=True)
