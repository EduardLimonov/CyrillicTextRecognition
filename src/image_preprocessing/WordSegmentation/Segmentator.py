from __future__ import annotations
from typing import *
from typing import List

import numpy as np
from utils.TextBlock import TextBlock
from itertools import product
from utils.algs import get_connect_components, postprocess_segmentation
from utils.geometry import Point, Rect
from SegAnalyzer import SegAnalyzer
from image_preprocessing.PicHandler import PicHandler


class Segmentator:
    """
    Поиск слов путем поиска компонент связности на графе закрашенных пикселей
    """
    @staticmethod
    def get_pixel_area(x, y, w, h, sensitivity: int = 1):
        diap = range(-sensitivity, sensitivity + 1)
        for dx, dy in product(diap, diap):
            if (dx, dy) != (0, 0):
                nx, ny = x + dx, y + dy
                if 0 <= nx < w and 0 <= ny < h:
                    yield nx, ny

    @staticmethod
    def find_borders(component: Iterable[Tuple[int, int]]) -> Tuple[int, int, int, int]:
        min_x = min(component, key=lambda t: t[0])[0]
        max_x = max(component, key=lambda t: t[0])[0]

        min_y = min(component, key=lambda t: t[1])[1]
        max_y = max(component, key=lambda t: t[1])[1]

        return min_x, min_y, max_x, max_y

    @staticmethod
    def find_rect(component: Iterable[Tuple[int, int]]) -> Rect:
        min_x, min_y, max_x, max_y = Segmentator.find_borders(component)
        return Rect(Point(min_x, min_y), Point(max_x, max_y))

    @staticmethod
    def parse_image(img: np.ndarray, sensitivity: int = 1, **params) -> List[TextBlock]:
        # данный метод качественно находит все отдельные компоненты связности img: 1, если пиксел закрашен,
        # иначе 0 sensitivity -- чувствительность к компонентам: какого кол-ва пустого места достаточно,
        # чтобы пикселы не считались смежными
        h, w = img.shape

        adj_lists: Dict[Tuple[int, int], List[Tuple[int, int]]] = {
            (_x, _y):
                [(nx, ny) for nx, ny in Segmentator.get_pixel_area(_x, _y, w, h) if img[ny, nx]]
            for _y, _x in product(range(h), range(w))
                if img[_y, _x]
        }  # adj_lists[i] -- список -- позиции закрашенных пикселов, которые смежны с закрашенным пикселом i

        components_pointed: List[Set[Tuple[int, int]]] = get_connect_components(adj_lists)

        zones = [Segmentator.find_rect(component) for component in components_pointed]
        zones = Segmentator._unite_components(zones, sensitivity, img, params)  # объединенные компоненты связности

        res = []
        for i in range(len(zones)):
            zone = zones[i]
            cont = PicHandler.crop(img, zone, make_copy=True)
            res.append(TextBlock(zone, cont))

        return res

    @staticmethod
    def _unite_components(zones: List[Rect], sensitivity: float, doc: np.ndarray, params: Optional[Dict]) -> List[Rect]:
        analyzer = SegAnalyzer(doc, params)

        def check_unite_rects(a: Rect, b: Rect, dist: float) -> bool:
            return analyzer.unite_segments(a, b, sensitivity, dist)

        return postprocess_segmentation(zones, check_unite_rects, max(params.values()))

    @staticmethod
    def parse_image_trivial(img: np.ndarray, **params) -> List[TextBlock]:
        sa = SegAnalyzer(img, params)
        zones = sa.trivial_parse()
        return [TextBlock(zone, PicHandler.crop(img, zone)) for zone in zones]
