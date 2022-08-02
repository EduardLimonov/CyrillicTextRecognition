from __future__ import annotations
from typing import *
import numpy as np
from utils.TextBlock import TextBlock
from itertools import product
from utils.algs import get_connect_components, unite
from utils.geometry import Point, Rect
from SegAnalyzer import SegAnalyzer


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
    def find_borders(component: Set[Tuple[int, int]]) -> Tuple[int, int, int, int]:
        def find_max_by_order(s: Set[Tuple[int, int]], axis: Union[0, 1], order_function: Callable[int, int, bool]) -> int:
            # order_function(a, b) == True => a -- max по отношению порядка order_function
            res = None
            for point in s:
                if isinstance(res, type(None)):
                    res = point[axis]
                    continue

                t = point[axis]
                if order_function(t, res):
                    res = t

            return res

        min_x = find_max_by_order(component, 0, lambda x1, x2: x1 < x2)
        max_x = find_max_by_order(component, 0, lambda x1, x2: x1 > x2)

        min_y = find_max_by_order(component, 1, lambda x1, x2: x1 < x2)
        max_y = find_max_by_order(component, 1, lambda x1, x2: x1 > x2)

        return min_x, min_y, max_x, max_y

    @staticmethod
    def find_rect(component: Set[Tuple[int, int]]) -> Rect:
        min_x, min_y, max_x, max_y = Segmentator.find_borders(component)
        return Rect(Point(min_x, min_y), Point(max_x, max_y))

    @staticmethod
    def parse_image(img: np.ndarray, sensitivity: int = 1) -> List[TextBlock]:
        # данный метод качественно находит все отдельные компоненты связности img: 1, если пиксел закрашен,
        # иначе 0 sensitivity -- чувствительность к компонентам: какого кол-ва пустого места достаточно,
        # чтобы пикселы не считались смежными
        h, w = img.shape
        # отображение координат точек в номера
        coord_to_num = lambda x, y: y * w + x
        num_to_coord = lambda num: (num % w, num // w)

        adj_lists: Dict[int, List[int]] = {
            coord_to_num(_x, _y):
                [coord_to_num(nx, ny) for nx, ny in Segmentator.get_pixel_area(_x, _y, w, h) if img[ny, nx]]
            for _y, _x in product(range(h), range(w))
                if img[_y, _x]
        }  # adj_lists[i] -- список -- номера закрашенных пикселов, которые смежны с закрашенным пикселом i

        components = get_connect_components(adj_lists)
        components_pointed = [{num_to_coord(num) for num in comp} for comp in components]

        zones = [Segmentator.find_rect(component) for component in components_pointed]
        components, zones = Segmentator.unite_components(components, zones, sensitivity,
                                                  img, coord_to_num, num_to_coord)  # объединенные компоненты связности
        #components_pointed = [{num_to_coord(num) for num in comp} for comp in components]  # то же, но с коордитами пикселов
        #zones = [Segmentator.find_rect(component) for component in components_pointed]

        res = []
        for i in range(len(components)):
            zone = zones[i]  # Segmentator.find_rect(component)
            cont = PicHandler.crop(img, zone, make_copy=True)
            #cont = PicHandler.draw_pixels(zone, component)
            res.append(TextBlock(zone, cont))

        return res

    @staticmethod
    def unite_components(components: List[Set[int]], zones: List[Rect], sensitivity: float, doc: np.ndarray,
                         coord_to_num: Callable, num_to_coord: Callable) -> Tuple[List[Set[int]], List[Rect]]:

        analyzer = SegAnalyzer(doc)

        def check_unite_rects(a: Rect, b: Rect) -> bool:
            return Rect.distance(a, b) <= sensitivity and analyzer.unite_segments(a, b)

        while True:
            changed = False
            to_unite: List[Tuple[int, int]] = []

            for ic1 in range(len(components)):
                for ic2 in range(ic1 + 1, len(components)):
                    if check_unite_rects(zones[ic1], zones[ic2]):
                        to_unite.append((ic1, ic2))
                        changed = True

            components = unite(components, to_unite)
            components_pointed = [{num_to_coord(num) for num in comp} for comp in components]
            zones = [Segmentator.find_rect(component) for component in components_pointed]

            if not changed:
                break

        return components, zones

    @staticmethod
    def parse_image_trivial(img: np.ndarray) -> List[TextBlock]:
        sa = SegAnalyzer(img)
        zones = sa.trivial_parse()
        return [TextBlock(zone, PicHandler.crop(img, zone)) for zone in zones]

    @staticmethod
    def parse_hybrid(img: np.ndarray, sensivity: float, threshold: float) -> List[TextBlock]:

        near = lambda a, b: 0 if not Rect.intersects(a, b) \
            else a.intersection(b).area() / min(a.area(), b.area())

        trivial_zones = SegAnalyzer(img).trivial_parse()
        blocks = Segmentator.parse_image(img, sensivity)
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
    #new_shape = h // 2.5, w // 2.5
    new_shape = h // 2, w // 2
    ph.resize(new_shape)

    ph.apply_adaptive_bin_filter(w=0.15)
    ph.apply_filter(GAUSSIAN_FILTER, 3)
    ph.apply_adaptive_bin_filter(w=0.15)
    ph.show()

    #tbs = Segmentator.parse_hybrid(ph.make_zero_one(), 9, 0.01)
    #tbs = Segmentator.parse_image(ph.make_zero_one(), 9)
    #tbs = Segmentator.parse_image_trivial(ph.make_zero_one())

    for tbs in (Segmentator.parse_image(ph.make_zero_one(), 3), Segmentator.parse_image(ph.make_zero_one(), 6),
                Segmentator.parse_image(ph.make_zero_one(), 24)):
        p = PicHandler(ph.img, make_copy=True)
        for tb in tbs:
            p.draw_rect(tb.zone, 120)
        p.show()

    '''

    for fname in ('46.jpg', '51.jpg'):
        ph = PicHandler(fname)
        ph.show()
        #ph.apply_filter(MEDIAN_FILTER, 5)
        #ph.apply_adaptive_bin_filter(1)
        ph.apply_fixed_bin_filter(240)
        ph.show()'''

    exit(0)
    fname = 'ex2small.jpg'
    start_time = time.time()

    ph = PicHandler(fname)
    ph.apply_filter(MEDIAN_FILTER, 5)
    ph.apply_adaptive_bin_filter()

    print(time.time() - start_time)

    #tbs = Segmentator.parse_image_trivial(ph.make_zero_one())
    print(ph.make_zero_one().shape)
    tbs = Segmentator.parse_image(ph.make_zero_one(), 15)
    print(time.time() - start_time)
    ph.show()
    for tb in tbs:
        tb.view()
