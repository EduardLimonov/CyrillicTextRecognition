from __future__ import annotations
from typing import *
from typing import List

import numpy as np
from utils.TextBlock import TextBlock
from itertools import product
from utils.algs import get_connect_components, postprocess_segmentation
from utils.geometry import Point, Rect
from SegAnalyzer import SegAnalyzer
from PicHandler import PicHandler


class Segmentator:
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
        zones = Segmentator.unite_components(zones, sensitivity, img, params)  # объединенные компоненты связности

        res = []
        for i in range(len(zones)):
            zone = zones[i]
            cont = PicHandler.crop(img, zone, make_copy=True)
            res.append(TextBlock(zone, cont))

        return res

    @staticmethod
    def unite_components(zones: List[Rect], sensitivity: float, doc: np.ndarray, params: Optional[Dict]) -> List[Rect]:
        analyzer = SegAnalyzer(doc, params)

        def check_unite_rects(a: Rect, b: Rect) -> bool:
            return analyzer.unite_segments(a, b, sensitivity, zones)

        return postprocess_segmentation(zones, check_unite_rects)

    @staticmethod
    def parse_image_trivial(img: np.ndarray, **params) -> List[TextBlock]:
        sa = SegAnalyzer(img, params)
        zones = sa.trivial_parse()
        return [TextBlock(zone, PicHandler.crop(img, zone)) for zone in zones]

    @staticmethod
    def parse_hybrid(img: np.ndarray, sensitivity: float, threshold: float) -> List[TextBlock]:

        near = lambda a, b: 0 if not Rect.intersects(a, b) \
            else a.intersection(b).area() / min(a.area(), b.area())

        trivial_zones = SegAnalyzer(img).trivial_parse()
        blocks = Segmentator.parse_image(img, int(sensitivity))
        zones = [tb.zone for tb in blocks]

        res = []
        for tzone in trivial_zones:
            rs = [z for z in zones if near(z, tzone) >= threshold]
            if len(rs):
                res.append(Rect.union_of_rects(rs))

        out_of_res = []
        for z in zones:
            not_intersects = True
            for r in res:
                if z.intersects(z, r):
                    not_intersects = False
            if not_intersects:
                out_of_res.append(z)

        res += out_of_res

        return [TextBlock(zone, PicHandler.crop(img, zone)) for zone in res]


if __name__ == "__main__":
    import time
    from PicHandler import *

    f1_scan = '../test/test1.png'
    f1_photo = '../test/hand1.jpg'
    f2_scan = '../test/handwritten2.jpg'
    f2_photo = '../test/hand2.jpg'

    ph = PicHandler(f1_scan)
    h, w = ph.img.shape
    new_shape = h // 2, w // 2
    ph.resize(new_shape)

    ph.apply_adaptive_bin_filter(w=0.15)
    ph.apply_filter(GAUSSIAN_FILTER, 3)
    ph.apply_adaptive_bin_filter(w=0.15)
    ph.show()

    #tbs = Segmentator.parse_hybrid(ph.make_zero_one(), 9, 0.01)
    #tbs = Segmentator.parse_image(ph.make_zero_one(), 9)
    #tbs = Segmentator.parse_image_trivial(ph.make_zero_one())

    start_time = time.time()

    #ph = PicHandler(fname)
    #ph.apply_filter(MEDIAN_FILTER, 5)
    #ph.apply_adaptive_bin_filter()

    #tbs = Segmentator.parse_image_trivial(ph.make_zero_one())
    tbs = Segmentator.parse_image(ph.make_zero_one(), 10)
    print(time.time() - start_time)
    p = PicHandler(ph.img, make_copy=True)
    for tb in tbs:
        p.draw_rect(tb.zone, 120)
    p.show()

