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
    learn_length: int = 11000
    test_length: int = 3100
    full_order: np.ndarray = np.arange(start=0, stop=learn_length + test_length, step=1, dtype=int)
    np.random.seed(len(targets))
    np.random.shuffle(full_order)
    print('full_order[:learn_length]', full_order[:learn_length].shape)
    learn_order: np.ndarray = full_order[:learn_length]
    test_order: np.ndarray = full_order[learn_length:learn_length + test_length]
    max_width: int = 0
    max_height: int = 0
    x: np.ndarray = np.zeros(shape=(learn_order.shape[0], TARGET_SIZE[0], TARGET_SIZE[1], 3),
                             dtype=float16)
    y: np.ndarray = np.zeros(shape=(learn_order.shape[0], TARGET_SIZE[0], TARGET_SIZE[1], 3),
                             dtype=float16)
    print('create learn data ...')
    for counter in tqdm.trange(learn_order.shape[0]):
        idx: int = learn_order[counter]
        f: str = targets[idx]
        color, line = make_contour_image(f)
        color = cv2.resize(color, dsize=TARGET_SIZE)
        line = cv2.resize(line, dsize=TARGET_SIZE)
        y[counter, :, :, :] = color.astype('float16') / 255
        x[counter, :, :, :] = line.astype('float16') / 255

    print('create test data ... ')
    validate_x: np.ndarray = np.zeros(shape=(test_order.shape[0], TARGET_SIZE[0], TARGET_SIZE[1], 3),
                                      dtype=float16)
    validate_y: np.ndarray = np.zeros(shape=(test_order.shape[0], TARGET_SIZE[0], TARGET_SIZE[1], 3),
                                      dtype=float16)
    for counter in tqdm.trange(test_order.shape[0]):
        idx: int = test_order[counter]
        f: str = targets[idx]
        color, line = make_contour_image(f)
        color = cv2.resize(color, dsize=TARGET_SIZE)
        line = cv2.resize(line, dsize=TARGET_SIZE)
        validate_x[counter, :, :, :] = color.astype('float16') / 255
        validate_y[counter, :, :, :] = line.astype('float16') / 255

    np.savez('data.npz', x=x, y=y, validate_x=validate_x, validate_y=validate_y)
