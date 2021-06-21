import os
import mlflow
import mlflow.sklearn

from src.train_keras import train_and_save
from src.train_forest import train_and_save_tree
from src.evaluate import get_and_save_scores


MLFLOW_TRACKING_URI = "http://0.0.0.0:5000"
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'secrets/mlflow-server-credentials.json'
EXPERIMENTS = ["keras_experiment", "RFC_experiment"]


if __name__ == "__main__":
    current_experiment = EXPERIMENTS[0]
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(current_experiment)
    with mlflow.start_run():
        if current_experiment == EXPERIMENTS[0]:
            batch_size = 128
            num_classes = 10 
            epochs = 5
            model, model_name = train_and_save(num_classes, batch_size=batch_size, epochs=epochs)
            mlflow.log_param("batch_size", batch_size)
            mlflow.log_param("num_classes", num_classes)
            mlflow.log_param("epochs", epochs)
            mlflow.keras.log_model(model, model_name)
            scores = get_and_save_scores(model_name)
        elif current_experiment == EXPERIMENTS[1]:
            n_estimators = 50
            model, model_name = train_and_save_tree(n_estimators)
            mlflow.log_param("n_estimators", n_estimators)
            mlflow.sklearn.log_model(model, model_name)
            scores = get_and_save_scores(model_name)
        else:
            raise ValueError("Wrong experiment name")
        
        mlflow.log_metrics(scores)
