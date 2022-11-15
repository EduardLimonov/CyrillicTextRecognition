from __future__ import annotations
from typing import *
import numpy as np
from utils.geometry import Rect, Point
import cv2 as cv


class Contour:
    cont_points: np.ndarray
    _bounding_rects: List[Rect]
    union_bb: Rect

    def __init__(self, contour: np.ndarray):
        self.cont_points = contour
        self.union_bb = Contour.find_rect(contour)
        self._bounding_rects = [self.union_bb]

    def union(self, other: Contour) -> Contour:
        self.cont_points = np.vstack([self.cont_points, other.cont_points])
        self._bounding_rects += other._bounding_rects
        self.union_bb = self.union_bb.union(other.union_bb)

        return self

    @staticmethod
    def distance(a: Contour, b: Contour, rude_sens: float) -> float:
        d = Rect.distance(a.union_bb, b.union_bb)
        if d > rude_sens:
            # грубая оценка расстояния
            return d
        # точная оценка расстояния
        return Contour._cont_distance(a.cont_points, b.cont_points)

    @staticmethod
    def _cont_distance(a: np.ndarray, b: np.ndarray):
        def _min_squared_distance_to_vector(xy, to):
            return np.min(np.square(xy[0] - to[:, 0]) + np.square(xy[1] - to[:, 1]))

        min_squared_distances = np.apply_along_axis(lambda _axy: _min_squared_distance_to_vector(_axy, b), 1, a)
        res = min_squared_distances.min()
        return np.sqrt(res)

    @staticmethod
    def find_rect(component: np.ndarray) -> Rect:
        x, y, w, h = cv.boundingRect(component)
        return Rect(Point(x, y), Point(x + w, y + h))
