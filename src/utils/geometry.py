from __future__ import annotations
from typing import *
from math import sqrt


def len_of_intersection(a1: int, a2: int, b1: int, b2: int) -> int:
    # длина пересечения двух отрезков на прямой

    if not(b1 <= a1 < b2 or b1 < a2 <= b2 or a1 <= b1 < a2 or a1 < b2 <= a2):
        # пересечения нет или это одна точка
        return 0

    start, stop = max(a1, b1), min(a2, b2)
    # найдем границы наибольшего отрезка, лежащего в пересечении a и b
    for b in (a1, a2, b1, b2):
        if a1 <= b <= a2 and b1 <= b <= b2:
            if b < start:
                start = b
            if b > stop:
                stop = b
    return stop - start


def sign(x: float) -> float:
    return 1 if x > 0 else (-1 if x < 0 else 0)


def distance(x1, y1, x2, y2) -> float:
    dx, dy = x2 - x1, y2 - y1
    return sqrt(dx * dx + dy * dy)


class Point:
    x: int
    y: int

    def __init__(self, x: int, y: int):
        self.x, self.y = x, y

    def __str__(self) -> str:
        return '(x: %d, y: %d)' % (self.x, self.y)

    def __add__(self, other: Point) -> Point:
        return Point(self.x + other.x, self.y + other.y)

    def __sub__(self, other: Point) -> Point:
        return Point(self.x - other.x, self.y - other.y)

    @staticmethod
    def distance(a: Point, b: Point) -> float:
        dp = a - b
        return sqrt(dp.x * dp.x + dp.y * dp.y)


class Rect:
    top_left: Point
    bottom_right: Point

    def __init__(self, tl: Point, br: Point):
        self.top_left, self.bottom_right = tl, br

    def __str__(self) -> str:
        return 'top-left: %s; right-bottom: %s' % (self.top_left, self.bottom_right)

    def __contains__(self, item: Point) -> bool:
        return self.top_left.x <= item.x <= self.bottom_right.x and \
               self.top_left.y <= item.y <= self.bottom_right.y

    def shape(self) -> Tuple[int, int]:
        return self.w(), self.h()

    def center(self) -> Point:
        return self.top_left + Point(int(self.w()/2), int(self.h()/2))

    def w(self) -> int:
        return self.bottom_right.x - self.top_left.x

    def h(self) -> int:
        return self.bottom_right.y - self.top_left.y

    @staticmethod
    def intersects(a: Rect, b: Rect) -> bool:
        return a.top_left in b or \
               a.bottom_right in b or \
               Point(a.top_left.x, a.bottom_right.y) in b or \
               Point(a.bottom_right.x, a.top_left.y) in b \
               or \
               b.top_left in a or \
               b.bottom_right in a or \
               Point(b.top_left.x, b.bottom_right.y) in a or \
               Point(b.bottom_right.x, b.top_left.y) in a

    @staticmethod
    def distance(a: Rect, b: Rect) -> float:
        if Rect.intersects(a, b):
            return 0

        ac, bc = a.center(), b.center()
        dc = ac - bc
        # non_inters = dc.x > a.w() + b.w() and dc.y > a.h() + b.h()  # ни одна из проекций на оси x, y не пересекается
        ix, iy = a.intersects_x(b), a.intersects_y(b)

        if not ix and not iy:
            # кратчайшее расстояние -- между вершинами
            x1, y1 = ac.x - a.w() / 2 * sign(dc.x), ac.y - a.h() / 2 * sign(dc.y)
            x2, y2 = bc.x + b.w() / 2 * sign(dc.x), bc.y + b.h() / 2 * sign(dc.y)
            return distance(x1, y1, x2, y2)

        elif ix:
            # достаточно найти расстояние по y
            y1 = ac.y - a.h() / 2 * sign(dc.y)
            y2 = bc.y + b.h() / 2 * sign(dc.y)
            return abs(y1 - y2)

        elif iy:
            # достаточно найти расстояние по x
            x1 = ac.x - a.w() / 2 * sign(dc.x)
            x2 = bc.x + b.w() / 2 * sign(dc.x)
            return abs(x1 - x2)

    def left(self) -> int:
        return self.top_left.x

    def right(self) -> int:
        return self.bottom_right.x

    def top(self) -> int:
        return self.top_left.y

    def bottom(self) -> int:
        return self.bottom_right.y

    def is_on_left(self, other: Rect) -> bool:
        return self.right() >= other.left()

    def is_on_right(self, other: Rect) -> bool:
        return other.is_on_left(self)

    def is_on_top(self, other: Rect) -> bool:
        return self.bottom() <= other.top()

    def is_on_bottom(self, other: Rect) -> bool:
        return other.is_on_top(self)

    def intersects_x(self, other: Rect) -> bool:
        return self.left() <= other.left() <= self.right() or \
               self.left() <= other.right() <= self.right() or \
               other.left() <= self.left() <= other.right() or \
               other.left() <= self.right() <= other.right()

    def intersects_y(self, other: Rect) -> bool:
        return self.top() <= other.top() <= self.bottom() or \
               self.top() <= other.bottom() <= self.bottom() or \
               other.top() <= self.top() <= other.bottom() or \
               other.top() <= self.bottom() <= other.bottom()

    def area(self) -> int:
        return self.w() * self.h()

    def distance_to_y(self, y: int) -> int:
        return min(abs(self.top() - y), abs(self.bottom() - y))

    def scale_x(self, scale_k: float) -> None:
        dx = int(self.w() * (1 - scale_k) / 2)
        self.top_left.x += dx
        self.bottom_right.x -= dx

    def scale_y(self, scale_k: float) -> None:
        dy = int(self.h() * (1 - scale_k) / 2)
        self.top_left.y -= dy
        self.bottom_right.y += dy

    def scale(self, scale_k: float) -> None:
        self.scale_x(scale_k)
        self.scale_y(scale_k)

    def intersection(self, other: Rect) -> Optional[Rect]:
        if not Rect.intersects(self, other):
            return None
        else:
            l = max(self.left(), other.left())
            t = max(self.top(), other.top())
            b = min(self.bottom(), other.bottom())
            r = min(self.right(), other.right())
            return Rect(Point(l, t), Point(r, b))

    def union(self, other: Rect) -> Rect:
        l = min(self.left(), other.left())
        t = min(self.top(), other.top())
        b = max(self.bottom(), other.bottom())
        r = max(self.right(), other.right())
        return Rect(Point(l, t), Point(r, b))

    @staticmethod
    def union_of_rects(rects: List[Rect]) -> Rect:
        res = rects[0]
        for i in range(1, len(rects)):
            res = res.union(rects[i])

        return res



if __name__ == '__main__':
    print(len_of_intersection(1, 4, -2, 3))
