import glob
from time import sleep
from typing import List, Tuple

import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import tqdm

SAVE_DIR: str = 'line-draw'


def make_contour_image(path: str, size: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
    neiborhood24 = np.array([[1, 1, 1, 1, 1],
                             [1, 1, 1, 1, 1],
                             [1, 1, 1, 1, 1],
                             [1, 1, 1, 1, 1],
                             [1, 1, 1, 1, 1]],
                            np.uint8)
    # グレースケールで画像を読み込む.
    gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    # cv2.imwrite("gray.jpg", gray)

    # 白い部分を膨張させる.
    dilated = cv2.dilate(gray, neiborhood24, iterations=1)
    # cv2.imwrite("dilated.jpg", dilated)

    # 差をとる.
    diff = cv2.absdiff(dilated, gray)
    # cv2.imshow("diff.jpg", diff)

    # 白黒反転
    contour = 255 - diff
    color: np.ndarray = np.zeros(shape=(size[0], size[1], 3), dtype=int)
    color[:] = 255
    origin: np.ndarray = np.asarray(cv2.imread(path, cv2.IMREAD_COLOR))[:, :, ::-1].copy()
    color[:origin.shape[0], :origin.shape[1], :] = origin

    line: np.ndarray = np.zeros_like(a=color, dtype=int)
    line[:] = 255
    line[:origin.shape[0], :origin.shape[1], 0] = contour
    line[:origin.shape[0], :origin.shape[1], 1] = contour
    line[:origin.shape[0], :origin.shape[1], 2] = contour

    # file_name: str = os.path.basename(path)
    # dir_name: str = os.path.dirname(path)
    # cv2.imwrite(os.path.join(SAVE_DIR, file_name), contour)
    return color, line


if __name__ == '__main__':
    max_size: Tuple[int, int] = (160, 160)
    target_size: Tuple[int, int] = (128, 128)
    if not os.path.exists(SAVE_DIR):
        os.mkdir(SAVE_DIR)
    targets: List[str] = glob.glob('source-images/*.png')
    max_width: int = 0
    max_height: int = 0
    # length: int = len(targets)
    length: int = 6000
    x: np.ndarray = np.zeros(shape=(length, target_size[0], target_size[1], 3),
                             dtype=float)
    y: np.ndarray = np.zeros(shape=(length, target_size[0], target_size[1], 3),
                             dtype=float)
    for idx, f in enumerate(tqdm.tqdm(targets[:length])):
        color, line = make_contour_image(f, size=(target_size[0], max_size[1]))
        color = np.resize(new_shape=target_size, a=color)
        line = np.resize(new_shape=target_size, a=line)
        # plt.imshow(color)
        # plt.show()
        # sleep(0.1)
        # plt.imshow(line)
        # plt.show()
        # sleep(0.1)
        # exit()

        y[idx, :, :, :] = color.astype('float32') / 255
        x[idx, :, :, :] = line.astype('float32') / 255

    np.savez('data.npz', x=x, y=y)

