import xgboost
from sklearn import datasets
from sklearn2pmml import sklearn2pmml
from sklearn2pmml import PMMLPipeline

boston = datasets.load_boston()
X = boston.data
y = boston.target
feature_names = boston.feature_names
model = xgboost.XGBRegressor(learning_rate=0.1, n_estimators=10, max_depth=10, silent=False)

boston_pipeline = PMMLPipeline([("regressor", model)])
boston_pipeline.active_fields = feature_names
boston_pipeline.fit(X, y)

sklearn2pmml(boston_pipeline, "boston.pmml", with_repr=True, debug=True)
