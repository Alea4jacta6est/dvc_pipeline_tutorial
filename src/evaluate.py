from keras.models import load_model
import click
import json

from preprocess import x_test, y_test
 

@click.command()
@click.argument('model_name', type=str, default='mnist.h5')
@click.argument('save_filename', type=str, default='scores.json')
def get_and_save_scores(model_name, save_filename):
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