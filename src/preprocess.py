import os

import click
import keras
from keras.datasets import mnist

from src.handlers import pickler


def get_and_preprocess_data(save_data_path = "data/data.pickle", num_classes = 10):
    """Get data if not cached in dir, preprocess given data

    Args:
        save_data_path (str): path to save preprocessed data
        num_classes (int): classes number; defaults to 10.

    Returns:
        x_train, y_train, x_test, y_test: data slices
    """
    path = os.getcwd()+"/data/mnist_data"
    (x_train, y_train), (x_test, y_test) = mnist.load_data(path)
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
    
    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    if save_data_path:
        processed_data = {"x_train": x_train, "y_train": y_train, 
                        "x_test":  x_test, "y_test": y_test}
        pickler.save(save_data_path, processed_data)
    return x_train, y_train, x_test, y_test

def get_processed_data(filename="data/data.pickle"):
    data_dict = pickler.read(filename)
    x_train, y_train = data_dict["x_train"], data_dict["y_train"]
    x_test, y_test = data_dict["x_test"], data_dict["y_test"]
    return x_train, y_train, x_test, y_test

if __name__ == "__main__":
    x_train, y_train, x_test, y_test = get_and_preprocess_data()