import os
import mlflow
import mlflow.sklearn
from sklearn.metrics import accuracy_score
from pipeline.preprocessing import load_data, prepare
from models.model_factory import model_factory

CANDIDATES = [
    ('xgboost', {'n_estimators': 50, 'use_label_encoder': False, 'eval_metric': 'mlogloss'}),
    ('random_forest', {'n_estimators': 100}),
    ('decision_tree', {}),
]


def evaluate_and_register(data_path: str = 'data/iris.csv'):
    df = load_data(data_path)
    X_train, X_val, X_test, y_train, y_val, y_test = prepare(df)

    mlflow.set_experiment('modular-demo')

    best_score = -1.0
    best_uri = None
    best_name = None

    for name, params in CANDIDATES:
        with mlflow.start_run(run_name=name):
            print(f"Training candidate: {name} with params {params}")
            model = model_factory(name, **params)
            model.fit(X_train, y_train)

            preds_val = model.predict(X_val)
            val_score = accuracy_score(y_val, preds_val)
            mlflow.log_metric('val_accuracy', float(val_score))

            # evaluate on test set and log
            preds_test = model.predict(X_test)
            test_score = accuracy_score(y_test, preds_test)
            mlflow.log_metric('test_accuracy', float(test_score))

            # log model artifact
            artifact_path = f'models/{name}'
            mlflow.sklearn.log_model(model, artifact_path)

            # log parameters
            mlflow.log_params(params)

            print(f"Candidate {name}: val={val_score:.4f}, test={test_score:.4f}")

            if val_score > best_score:
                best_score = val_score
                best_name = name
                best_uri = f'runs:/{mlflow.active_run().info.run_id}/{artifact_path}'

    if best_uri:
        model_name = 'modular-demo-model'
        print(f"Registering best model {best_name} with score {best_score:.4f} -> uri {best_uri}")
        # Note: register_model requires an MLflow Tracking Server with a backend store.
        mv = mlflow.register_model(best_uri, model_name)
        print('Registered', mv.name, 'version', mv.version)

    return best_name, best_score


if __name__ == '__main__':
    evaluate_and_register()