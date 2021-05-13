import json
import click
from keras.models import load_model

from preprocess import get_processed_data
 

@click.command()
@click.argument('model_name', type=str, default='models/mnist_model_5.h5')
@click.argument('save_filename', type=str, default='reports/scores.json')
def get_and_save_scores(model_name, save_filename):
    """Gets cached test data, loads model and counts scores

    Args:
        model_name ([type]): [description]
        save_filename ([type]): [description]
    """
    _, _, x_test, y_test = get_processed_data()
    model = load_model(model_name)
    score = model.evaluate(x_test, y_test, verbose=0)
    final_scores = {'Test loss': round(score[0], 2), 
                    'Test accuracy': round(score[1], 2)}
    print(final_scores)
    if save_filename:
        with open(save_filename, "w") as file:
            json.dump(final_scores, file, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    get_and_save_scores()