from __future__ import annotations
from PicHandler import *
from WriteHelper import WriteHelper
from SegAnalyzer import SegAnalyzer
import cv2


def get_crop_index(img: np.ndarray, trange, axis) -> int:
    idx = 0
    for idx in trange:
        if axis == 0:
            if not img[idx].all():    return idx
        else:
            if not img[:, idx].all():    return idx

    return idx


class BitMatrix:
    matrix: np.ndarray

    def __init__(self, img: Union[PicHandler, BitMatrix, np.ndarray]):
        if type(img) is PicHandler:
            self.matrix = img.img.copy()
        elif type(img) is BitMatrix:
            self.matrix = img.matrix
        elif type(img) is np.ndarray:
            self.matrix = img

        self.__crop_borders()

    def __crop_borders(self) -> None:
        x_min = get_crop_index(self.matrix, range(self.matrix.shape[1] - 1), 1)
        x_max = get_crop_index(self.matrix, range(self.matrix.shape[1] - 1, 0, -1), 1)
        y_min = get_crop_index(self.matrix, range(self.matrix.shape[0] - 1), 0)
        y_max = get_crop_index(self.matrix, range(self.matrix.shape[0] - 1, 0, -1), 0)

        self.matrix = self.matrix[y_min: y_max, x_min: x_max]

    @staticmethod
    def count_black(matrix: np.ndarray) -> int:
        return (matrix == 0).sum()


if __name__ == '__main__':
    fname = '01_270.png'
    # fname = '51.jpg'
    ph = PicHandler(fname)
    ph.show()
    ph.apply_adaptive_bin_filter()
    matrix = BitMatrix(ph)
    view_image(matrix.matrix)
