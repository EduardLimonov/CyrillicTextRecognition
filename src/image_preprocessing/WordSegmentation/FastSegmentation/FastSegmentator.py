from __future__ import annotations
from typing import *
import cv2 as cv
from utils.TextBlock import TextBlock
from utils.algs import postprocess_segmentation
from utils.geometry import Rect
import numpy as np
from image_preprocessing.WordSegmentation.FastSegmentation.Contour import Contour
from image_preprocessing.WordSegmentation.SegAnalyzer import SegAnalyzer


PARSING_MODE = 'PARSING_MODE'
FAST_MODE = 1
DEFAULT_MODE = 0


class FastSegmentator:
    """
    Поиск слов с помощью функции поиска контуров из opencv
    """

    @staticmethod
    def parse(img: np.ndarray, sens: Union[int, float], **kwargs) -> List[TextBlock]:
        if PARSING_MODE in kwargs.keys():
            m = kwargs[PARSING_MODE]
        else:
            m = DEFAULT_MODE

        contours, hier = cv.findContours(img, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
        contours = [Contour(cont.reshape((-1, 2))) for cont in contours]

        uf = lambda a, b: a.union(b)
        if m == DEFAULT_MODE:
            criterion = lambda a, b, dist: SegAnalyzer.unite_segments(a.union_bb, b.union_bb, sens, dist)
            distance = lambda a, b: Contour.distance(a, b, sens)
            cs = contours
        else:
            criterion = lambda a, b, dist: SegAnalyzer.unite_segments(a, b, sens, dist)
            distance = Rect.distance
            cs = [c.union_bb for c in contours]

        contours = postprocess_segmentation(
            cs, criterion, float(max(kwargs.values())), union_function=uf, distance_metric=distance
        )

        if m == DEFAULT_MODE:
            rects = [contour.union_bb for contour in contours]
        else:
            rects = contours

        return [TextBlock(rect, img[rect.top(): rect.bottom(), rect.left(): rect.right()]) for rect in rects]
