from typing import Any


def model_factory(name: str, **kwargs) -> Any:
    name = name.lower()
    if name == 'xgboost':
        from xgboost import XGBClassifier
        return XGBClassifier(**kwargs)
    elif name in ('random_forest', 'rf'):
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(**kwargs)
    elif name in ('decision_tree', 'dt'):
        from sklearn.tree import DecisionTreeClassifier
        return DecisionTreeClassifier(**kwargs)
    else:
        raise ValueError(f'Unknown model: {name}')