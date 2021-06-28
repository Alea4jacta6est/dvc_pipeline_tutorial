from datetime import timedelta
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.utils.dates import days_ago


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

train_keras = BashOperator(
                    task_id='train_keras',
                    depends_on_past=False,
                    bash_command='python src/train_keras.py',
                    retries=2,
                    dag=dag,
                    )
train_rfc = BashOperator(
                    task_id='train_RFC',
                    depends_on_past=False,
                    bash_command='python src/train_forest.py',
                    retries=2,
                    dag=dag,
                )
count_scores = BashOperator(
                        task_id='count_scores',
                        depends_on_past=False,
                        bash_command='python src/evaluate.py',
                        retries=2,
                        dag=dag,
    )
    
[train_rfc, train_keras] >> count_scores

