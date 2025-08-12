# small helper showing how to create an xgboost model with params
from models.model_factory import model_factory

def build_xgb():
    return model_factory('xgboost', n_estimators=50, use_label_encoder=False, eval_metric='mlogloss')