import os
from fastapi import FastAPI, HTTPException
import pandas as pd
from mlflow.pyfunc import load_model

app = FastAPI()

MODEL_NAME = os.getenv('MODEL_NAME', 'modular-demo-model')
MODEL_STAGE = os.getenv('MODEL_STAGE', 'Production')
LOCAL_FALLBACK = os.getenv('LOCAL_FALLBACK', 'models/best_model.pkl')

model = None


def load_from_registry():
    uri = f"models:/{MODEL_NAME}/{MODEL_STAGE}"
    return load_model(uri)


def load_from_local(path):
    return load_model(path)


@app.on_event('startup')
def startup_event():
    global model
    try:
        model = load_from_registry()
        print('Loaded model from registry')
    except Exception as e:
        print('Could not load from registry:', e)
        try:
            model = load_from_local(LOCAL_FALLBACK)
            print('Loaded local fallback model')
        except Exception as e2:
            print('No model available at startup', e2)
            model = None


@app.post('/predict')
def predict(instances: list):
    global model
    if model is None:
        raise HTTPException(status_code=503, detail='Model not loaded')
    df = pd.DataFrame(instances)
    preds = model.predict(df)
    return {'predictions': preds.tolist()}


@app.post('/reload')
def reload_model():
    global model
    try:
        model = load_from_registry()
        return {'status': 'reloaded', 'source': 'registry'}
    except Exception as e:
        return {'status': 'error', 'error': str(e)}