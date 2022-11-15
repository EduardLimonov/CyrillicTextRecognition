from __future__ import annotations
from typing import *
import numpy as np
from image_preprocessing.PicHandler import PicHandler, view_image


class BitMatrix:
    matrix: np.ndarray

    def __init__(self, img: Union[PicHandler, BitMatrix, np.ndarray]):
        # img: 0 -- пиксель черный, 255 -- белый
        if type(img) is PicHandler:
            self.matrix = img.get_copy()
        elif type(img) is BitMatrix:
            self.matrix = img.matrix
        elif type(img) is np.ndarray:
            self.matrix = img

        self.__crop_borders()

    def __crop_borders(self) -> None:
        x_min = np.concatenate((np.where(~np.all(self.matrix, axis=0))[0], [0]))[0]
        x_max = np.concatenate(([self.matrix.shape[1]], np.where(~np.all(self.matrix, axis=0))[0]))[-1]
        y_min = np.concatenate((np.where(~np.all(self.matrix, axis=1))[0], [0]))[0]
        y_max = np.concatenate(([self.matrix.shape[0]], np.where(~np.all(self.matrix, axis=1))[0]))[-1]

        self.matrix = self.matrix[y_min: y_max, x_min: x_max]

    @staticmethod
    def count_black(_matrix: np.ndarray) -> int:
        return (_matrix == 0).sum()


if __name__ == '__main__':
    fname = '../../resources/01_270.png'
    ph = PicHandler(fname)
    ph.show()
    ph.apply_adaptive_bin_filter()
    import time
    start = time.time()

    matrix = BitMatrix(ph)
    print(time.time() - start)
    view_image(matrix.matrix)
