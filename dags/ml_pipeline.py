from datetime import timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator, PythonVirtualenvOperator
from airflow.utils.dates import days_ago

from src.train_keras import train_and_save
from src.train_forest import train_and_save_tree
from src.evaluate import get_and_save_scores


default_args = {
    'owner': 'Victoria Latynina',
    'depends_on_past': False,
    'start_date': days_ago(1),
    'email': ['v.latynina46@gmail.com'],
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=2)
}

dag = DAG(
        'ml_mnist_pipeline',
        default_args=default_args,
        description='MNIST Machine Learning pipeline',
        schedule_interval=timedelta(days=30))

train_keras = PythonOperator(
                    task_id='train_keras',
                    depends_on_past=False,
                    python_callable=train_and_save,
                    retries=2,
                    dag=dag,
                    )
train_rfc = PythonOperator(
                    task_id='train_RFC',
                    depends_on_past=False,
                    python_callable=train_and_save_tree,
                    retries=2,
                    dag=dag,
                )
count_scores = PythonOperator(
                        task_id='count_scores',
                        depends_on_past=False,
                        python_callable=get_and_save_scores,
                        retries=2,
                        dag=dag,
    )
    
[train_rfc, train_keras] >> count_scores

