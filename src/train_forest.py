import click
import joblib
from sklearn.ensemble import RandomForestClassifier

from preprocess import get_processed_data


@click.command()
@click.argument('n_estimators', type=int, default=50)
def train_and_save_tree(n_estimators):
    """Trains RFC and saves it as a joblib file

    Args:
        n_estimators (int): hyperparameter
    """
    x_train, y_train, _, _ = get_processed_data()
    nsamples, nx, ny, _ = x_train.shape
    x_train = x_train.reshape((nsamples, nx*ny))
    forest = RandomForestClassifier(n_estimators = n_estimators)
    forest = forest.fit(x_train, y_train)
    joblib.dump(forest, f"models/random_forest_{n_estimators}.joblib")
    
if __name__ == "__main__":
    train_and_save_tree()
