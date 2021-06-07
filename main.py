import mlflow
import mlflow.sklearn

from src.preprocess import get_and_preprocess_data
from src.train_keras import train_and_save
from src.evaluate import get_and_save_scores


MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"

if __name__ == "__main__":
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("1_experiment")
    with mlflow.start_run():
        batch_size = 128
        num_classes = 10 
        epochs = 1
        model = train_and_save()
        scores = get_and_save_scores()

        mlflow.log_param("batch_size", str(batch_size))
        mlflow.log_param("num_classes", str(num_classes))
        mlflow.log_param("epochs", str(epochs))
        mlflow.log_metrics("score", scores)
        mlflow.sklearn.log_model(model, "model")
        client = mlflow.tracking.MlflowClient()
        data = client.get_run(mlflow.active_run().info.run_id).data
    print("Here", data)