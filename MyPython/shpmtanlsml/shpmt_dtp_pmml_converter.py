from sklearn2pmml import sklearn2pmml
from sklearn2pmml import PMMLPipeline


def convert_sklearn_to_pmml(model, pmml, feature_names=None, target_name=None):
    pipeline = PMMLPipeline([("regressor", model)])
    if feature_names is not None:
        pipeline.active_fields = feature_names
    if target_name is not None:
        pipeline.target_field = target_name
    sklearn2pmml(pipeline, pmml, with_repr=True, debug=True)
