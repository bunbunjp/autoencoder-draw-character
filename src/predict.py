from keras import Sequential
from numpy.lib.npyio import NpzFile
from typing import List

from src.fit import create_model
import numpy as np
import matplotlib.pyplot as plt


def inverse(x: np.ndarray) -> np.ndarray:
    r: np.ndarray = np.zeros(shape=(x.shape[0], x.shape[1], x.shape[2]),
                             dtype=int)
    r[:] = x * 255
    r[:] = np.where(r[:] < 0, 0, r[:])
    r[:] = np.where(r[:] > 255, 255, r[:])
    return r


if __name__ == '__main__':
    data: NpzFile = np.load('/home/huna/data.npz')
    x: np.ndarray = data['x']
    y: np.ndarray = data['y']

    model: Sequential = create_model(target=x)
    model.load_weights(filepath='weight.h5')

    targets: List[int] = [
        1,
        2,
        10,
        100,
        101,
        102,
        110,
        1000,
        1001,
    ]
    for idx in targets:
        p: np.ndarray = model.predict(x=x[idx:idx+1, :, :, :])
        inv: np.ndarray = inverse(x=p[0])

        plt.subplot(2, 2, 1)
        plt.imshow(inverse(x=x[idx, :, :, :]))
        plt.title('line draw')

        plt.subplot(2, 2, 2)
        plt.imshow(inverse(x=y[idx, :, :, :]))
        plt.title('origin')

        plt.subplot(2, 2, 3)
        plt.imshow(inv)
        plt.title('predict')

        plt.suptitle(str(idx))
        plt.savefig('image/{0}.png'.format(idx))

        plt.show()