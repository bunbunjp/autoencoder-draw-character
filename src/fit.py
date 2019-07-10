import numpy as np
from keras import Sequential
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, UpSampling2D, Flatten, Dense, Dropout, Reshape, \
    Activation, LeakyReLU, AveragePooling2D, Conv2DTranspose
from keras.optimizers import Adam
from numpy.lib.npyio import NpzFile
import matplotlib.pyplot as plt
from typing import Tuple


# import keras.backend as K
# K.set_floatx('float16')


def create_model(target: np.ndarray) -> Sequential:
    ae: Sequential = Sequential()
    kernel_size: Tuple[int, int] = (3, 3)

    filter_len: int = 64
    ae.add(Conv2DTranspose(filter_len, kernel_size, padding='same',
                           input_shape=(target.shape[1], target.shape[2], 3)))
    ae.add(LeakyReLU())
    ae.add(BatchNormalization())
    ae.add(MaxPooling2D())

    ae.add(Conv2DTranspose(filter_len * 2, kernel_size, padding='same'))
    ae.add(LeakyReLU())
    ae.add(BatchNormalization())
    ae.add(MaxPooling2D())

    ae.add(Conv2DTranspose(filter_len * 4, kernel_size, padding='same'))
    ae.add(LeakyReLU())
    ae.add(BatchNormalization())
    ae.add(MaxPooling2D())

    ae.add(Conv2DTranspose(filter_len * 4, kernel_size, padding='same'))
    ae.add(LeakyReLU())
    ae.add(BatchNormalization())
    ae.add(MaxPooling2D())

    ae.add(Conv2DTranspose(filter_len * 4, kernel_size, padding='same'))
    ae.add(LeakyReLU())
    ae.add(BatchNormalization())
    # ae.add(MaxPooling2D())

    ae.add(Conv2DTranspose(filter_len * 4, kernel_size, padding='same'))
    ae.add(Activation('relu'))
    ae.add(BatchNormalization())
    ae.add(Dropout(0.25))
    # ae.add(UpSampling2D())

    ae.add(Conv2DTranspose(filter_len * 4, kernel_size, padding='same'))
    ae.add(Activation('relu'))
    ae.add(BatchNormalization())
    ae.add(Dropout(0.25))
    ae.add(UpSampling2D())

    ae.add(Conv2DTranspose(filter_len * 4, kernel_size, padding='same'))
    ae.add(Activation('relu'))
    ae.add(BatchNormalization())
    ae.add(Dropout(0.25))
    ae.add(UpSampling2D())

    ae.add(Conv2DTranspose(filter_len * 2, kernel_size, padding='same'))
    ae.add(Activation('relu'))
    ae.add(BatchNormalization())
    ae.add(Dropout(0.25))
    ae.add(UpSampling2D())

    ae.add(Conv2DTranspose(filter_len, kernel_size, padding='same'))
    ae.add(Activation('relu'))
    ae.add(BatchNormalization())
    ae.add(Dropout(0.25))
    ae.add(UpSampling2D())

    ae.add(Conv2DTranspose(3, kernel_size, padding='same'))
    ae.add(Activation('tanh'))

    # binary_crossentropy, mse
    ae.compile(loss='mse', optimizer=Adam(lr=0.0002), metrics=['accuracy'])
    ae.summary()
    return ae


if __name__ == '__main__':
    data: NpzFile = np.load('data.npz')
    x: np.ndarray = data['x']
    y: np.ndarray = data['y']

    epoch: int = 400
    model: Sequential = create_model(target=x)
    stack = model.fit(x=x, y=y, epochs=epoch, verbose=1, batch_size=64,
                      validation_split=0.2)

    model.save(filepath='weight.h5')
    plt.plot(range(epoch), stack.history['loss'], label="loss")
    plt.plot(range(epoch), stack.history['val_loss'], label="val_loss")
    plt.show()
