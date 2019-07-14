import glob
import os
from typing import List, Tuple

import cv2
import numpy as np
import tqdm
from numpy import uint8, float16

SAVE_DIR: str = 'line-draw'
MAX_SIZE: Tuple[int, int] = (160, 160)
TARGET_SIZE: Tuple[int, int] = (128, 128)


def make_contour_image(path: str) -> Tuple[np.ndarray, np.ndarray]:
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
    color: np.ndarray = np.zeros(shape=(MAX_SIZE[0], MAX_SIZE[1], 3), dtype=uint8)
    color[:] = 255
    origin: np.ndarray = np.asarray(cv2.imread(path, cv2.IMREAD_COLOR))[:, :, ::-1].copy()
    color[:origin.shape[0], :origin.shape[1], :] = origin
    line: np.ndarray = np.zeros_like(a=color, dtype=uint8)
    line[:] = 255
    line[:origin.shape[0], :origin.shape[1], 0] = contour
    line[:origin.shape[0], :origin.shape[1], 1] = contour
    line[:origin.shape[0], :origin.shape[1], 2] = contour

    # file_name: str = os.path.basename(path)
    # dir_name: str = os.path.dirname(path)
    # cv2.imwrite(os.path.join(SAVE_DIR, file_name), contour)
    return color, line


if __name__ == '__main__':
    if not os.path.exists(SAVE_DIR):
        os.mkdir(SAVE_DIR)
    targets: List[str] = glob.glob('source-images/*.png')
    max_width: int = 0
    max_height: int = 0
    # length: int = len(targets)
    length: int = 12000
    x: np.ndarray = np.zeros(shape=(length, TARGET_SIZE[0], TARGET_SIZE[1], 3),
                             dtype=float16)
    y: np.ndarray = np.zeros(shape=(length, TARGET_SIZE[0], TARGET_SIZE[1], 3),
                             dtype=float16)
    for idx, f in enumerate(tqdm.tqdm(targets[:length])):
        color, line = make_contour_image(f)
        color = cv2.resize(color, dsize=TARGET_SIZE)
        line = cv2.resize(line, dsize=TARGET_SIZE)
        # plt.imshow(color)
        # plt.show()
        # sleep(0.1)
        # plt.imshow(line)
        # plt.show()
        # sleep(0.1)
        # exit()

        y[idx, :, :, :] = color.astype('float16') / 255
        x[idx, :, :, :] = line.astype('float16') / 255

    np.savez('data.npz', x=x, y=y)



