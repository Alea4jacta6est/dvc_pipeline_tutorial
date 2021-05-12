import click
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D

from preprocess import get_processed_data


def create_model(num_classes):
    input_shape = (28, 28, 1)
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])
    return model

@click.command()
@click.argument('batch_size', type=int, default=128)
@click.argument('num_classes', type=int, default=10)
@click.argument('epochs', type=int, default=10)
def train_and_save(num_classes,  batch_size, epochs, model_name='mnist.h5'):
    x_train, y_train, x_test, y_test = get_processed_data()
    model = create_model(num_classes)
    hist = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_test, y_test))
    print("The model has successfully trained")
    model.save(model_name)
    print("Saving the model as mnist.h5")


if __name__ == "__main__":
    train_and_save()