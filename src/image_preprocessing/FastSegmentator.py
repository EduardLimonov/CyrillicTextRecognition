from __future__ import annotations
from typing import *
import cv2 as cv
from utils.TextBlock import TextBlock
from utils.algs import postprocess_segmentation
from utils.geometry import Rect, Point
from SegAnalyzer import SegAnalyzer
import numpy as np


class FastSegmentator:
    @staticmethod
    def find_rect(component: Iterable[np.ndarray]) -> Rect:
        '''min_val, max_val = np.min(component), np.max(component)
        min_x = np.amin(component, where=[False, True], initial=max_val)
        max_x = np.amax(component, where=[False, True], initial=min_val)
        min_y = np.amin(component, where=[True, False], initial=max_val)
        max_y = np.amax(component, where=[True, False], initial=min_val)'''

        x, y ,w, h = cv.boundingRect(component)
        return Rect(Point(x, y), Point(x + w, y + h))

    @staticmethod
    def parse(img: np.ndarray, sens: Union[int, float]) -> List[TextBlock]:
        res = []
        contours, hier = cv.findContours(img, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE)
        rects = [FastSegmentator.find_rect(contour) for contour in contours]

        def _criterion(a: Rect, b: Rect) -> bool:
            return SegAnalyzer.unite_segments(a, b, sens, rects)

        rects = postprocess_segmentation(
            rects, _criterion
        )

        return [TextBlock(rect, img[rect.top(): rect.bottom(), rect.left(): rect.right()]) for rect in rects]
