import glob
from time import sleep
from typing import List
import numpy as np
from keras import Sequential
from numpy import float16
import cv2
import matplotlib.pyplot as plt

from src.create_dataset import TARGET_SIZE, make_contour_image
from src.fit import create_model
from src.predict import inverse


def main():
    targets: List[str] = glob.glob('test_image/*')
    length: int = len(targets)
    x: np.ndarray = np.zeros(shape=(length, TARGET_SIZE[0], TARGET_SIZE[1], 3),
                             dtype=float16)
    for idx, f in enumerate(targets):
        print(f)
        color, line = make_contour_image(f)
        # color = cv2.resize(color, dsize=TARGET_SIZE)
        line = cv2.resize(line, dsize=TARGET_SIZE)
        x[idx, :, :, :] = line.astype('float16') / 255

    model: Sequential = create_model(target=x)
    model.load_weights(filepath='weight.h5')
    y: np.ndarray = model.predict(x=x)
    plt.figure(figsize=(4, 9))
    for idx in range(length):
        plt.subplot(length, 2, 1 + 2 * idx)
        plt.imshow(inverse(x[idx, :, :, :]))

        plt.subplot(length, 2, 2 + 2 * idx)
        plt.imshow(inverse(y[idx, :, :, :]))
    plt.show()

    return


if __name__ == '__main__':
    main()
