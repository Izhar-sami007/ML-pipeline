from models.model_factory import model_factory

def build_rf():
    return model_factory('random_forest', n_estimators=100)