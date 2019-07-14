import os
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from keras import Sequential
from keras.callbacks import ModelCheckpoint
from keras.layers import MaxPooling2D, BatchNormalization, UpSampling2D, Dropout, Activation, LeakyReLU, Conv2D, \
    Conv2DTranspose
from keras.optimizers import Adam
from numpy.lib.npyio import NpzFile


def create_model(target: np.ndarray) -> Sequential:
    ae: Sequential = Sequential()
    kernel_size: Tuple[int, int] = (3, 3)

    filter_len: int = 64
    # Encoder
    ae.add(Conv2D(filter_len, (1, 1), padding='same',
                  input_shape=(target.shape[1], target.shape[2], 3)))
    ae.add(BatchNormalization())
    ae.add(Activation('relu'))

    ae.add(Conv2D(filter_len * 2, kernel_size, padding='same'))
    ae.add(BatchNormalization())
    ae.add(LeakyReLU())
    ae.add(MaxPooling2D())

    ae.add(Conv2D(filter_len * 4, kernel_size, padding='same'))
    ae.add(BatchNormalization())
    ae.add(LeakyReLU())
    ae.add(MaxPooling2D())

    # Middle
    ae.add(Conv2D(filters=filter_len * 4, kernel_size=kernel_size, padding='same'))
    ae.add(Activation('relu'))

    ae.add(Conv2D(filters=filter_len * 4, kernel_size=kernel_size, padding='same'))
    ae.add(Activation('relu'))

    ae.add(Conv2D(filters=filter_len * 4, kernel_size=kernel_size, padding='same'))
    ae.add(Activation('relu'))

    ae.add(Conv2DTranspose(filter_len * 4, kernel_size, padding='same'))
    ae.add(Activation('relu'))

    # Decoder
    ae.add(Conv2DTranspose(filter_len * 2, kernel_size, padding='same'))
    ae.add(BatchNormalization())
    ae.add(Activation('relu'))
    ae.add(UpSampling2D())

    ae.add(Conv2DTranspose(filter_len * 2, kernel_size, padding='same'))
    ae.add(BatchNormalization())
    ae.add(Activation('relu'))
    ae.add(UpSampling2D())

    ae.add(Conv2D(filter_len * 2, kernel_size, padding='same'))
    ae.add(Activation('relu'))
    ae.add(Conv2D(filter_len * 1, kernel_size, padding='same'))
    ae.add(Activation('relu'))
    ae.add(Dropout(0.25))
    ae.add(Conv2D(int(filter_len / 2), kernel_size, padding='same'))
    ae.add(Activation('relu'))
    ae.add(Dropout(0.25))

    ae.add(Conv2D(3, kernel_size, padding='same'))
    ae.add(Activation('tanh'))

    # binary_crossentropy, mse
    ae.compile(loss='mae',
               optimizer=Adam(lr=0.0001), metrics=['accuracy'])
    ae.summary()
    return ae


if __name__ == '__main__':
    data: NpzFile = np.load('data.npz')
    x: np.ndarray = data['x']
    y: np.ndarray = data['y']

    epoch: int = 300
    model: Sequential = create_model(target=x)
    chkpt: str = os.path.join('check_point', 'weight_{epoch:02d}-{val_loss:.2f}.h5')
    cp_cb: ModelCheckpoint = ModelCheckpoint(filepath=chkpt, monitor='val_loss', verbose=1, save_best_only=True,
                                             mode='auto')

    stack = model.fit(x=x, y=y, epochs=epoch, verbose=1, batch_size=32,
                      validation_split=0.12,
                      callbacks=[cp_cb])
    # callbacks=[EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='auto')])
    model.save(filepath='weight.h5')
    plt.plot(stack.history['loss'], label="loss")
    plt.plot(stack.history['val_loss'], label="val_loss")
    plt.show()
