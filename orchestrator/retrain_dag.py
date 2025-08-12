from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta

from model_selection import evaluate_and_register


def trigger_retrain(**kwargs):
    evaluate_and_register()


def print_hello():
    print('Checking drift (placeholder)')


default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2025, 1, 1),
    'retries': 0,
}

with DAG('retrain_pipeline', default_args=default_args, schedule_interval='@daily', catchup=False) as dag:
    t1 = PythonOperator(task_id='check_drift', python_callable=print_hello)
    t2 = PythonOperator(task_id='retrain', python_callable=trigger_retrain)

    t1 >> t2