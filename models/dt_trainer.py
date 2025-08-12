from models.model_factory import model_factory

def build_dt():
    return model_factory('decision_tree', random_state=42)