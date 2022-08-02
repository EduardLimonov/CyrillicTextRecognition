from __future__ import annotations
from typing import *
import numpy as np
from utils.geometry import Rect, Point, len_of_intersection


class SegAnalyzer:
    # Экземпляру этого класса нужно скормить бинаризованный документ (1 -- пиксел закрашен, 0 -- незакрашен).
    # Потом этот экземпляр будет указывать точнее, следует ли объединять две компоненты связности в одну, или нет

    strings_on_doc: List[Tuple[int, int]]  # для каждой строки текста: y-координата начала и конца
    words_in_strings: List[List[Tuple[int, int]]]  # начало и конец слова в каждой строке

    TEXT_THRESH = 0.2  # порог количества закрашенных пикселов в строке (доля от пикового значения в строке)
    WORDS_HORIZ_THRESH = 0.02 #0.1
    MIN_LINE_DIST = 5  # минимальный межстрочный интервал в пикселах
    MIN_LINE_HEIGHT = 8
    MIN_WORDS_DIST = 10
    DIACRITICS_PROP = 0.1  # максимальная высота диакритического знака относительно высоты символа
    MAX_LINE_H = 30
    MIN_OUTSTR_DIST = 5
    MAX_OUTSTR_PROP = 1.8

    def __init__(self, doc: np.ndarray):
        self.strings_on_doc = []
        self._analyze(doc)

    @staticmethod
    def find_peaks_ids(arr: np.ndarray, thresh: float) -> List[int]:
        mb = arr >= thresh

        def is_peak(i) -> bool:
            if i == 0 or i == len(arr) - 1:
                return False
            return mb[i] and arr[i - 1] <= arr[i] and arr[i] >= arr[i + 1]

        return [i for i in range(len(arr)) if is_peak(i)]

    @staticmethod
    def find_border_idx(arr: np.ndarray, peak_idx, next_idx: Callable[int, int], thresh: float) -> int:
        # находит границы строки текста (надстрочная часть символа не входит в строку по критерию thresh);
        # изменение индекса определяется функцией next_idx
        thresh_val = arr[peak_idx] * thresh  # порог суммы, ниже которого строка пикселов считается пустой строкой между
        # строк текста
        idx = peak_idx
        while idx < len(arr) and arr[idx] >= thresh_val and 0 < idx < len(arr):
            idx = next_idx(idx)

        return idx

    def _analyze(self, doc: np.ndarray) -> None:
        # заполняет список строк self.strings_on_doc
        sums = doc.sum(axis=1)  # построчные суммы; в тех строках пикселов, где большие суммы, идет строка

        peaks_ids = SegAnalyzer.find_peaks_ids(sums, thresh=sums.max() * 0.1)
        for peak_idx in peaks_ids:
            border_low = SegAnalyzer.find_border_idx(sums, peak_idx, lambda i: i - 1, SegAnalyzer.TEXT_THRESH)
            border_up = SegAnalyzer.find_border_idx(sums, peak_idx, lambda i: i + 1, SegAnalyzer.TEXT_THRESH)

            self.strings_on_doc.append((border_low, border_up))

        self._unite_lines()

        self.words_in_strings = []
        self._find_words(doc)

    def _find_words(self, doc: np.ndarray) -> None:
        # для каждой строки (линии со словами) определяет, где проходят вертикальные границы между словами
        for top, bot in self.strings_on_doc:
            string: np.ndarray = doc[top: bot]
            vert_sums = string.sum(axis=0)

            peaks_ids = SegAnalyzer.find_peaks_ids(vert_sums, vert_sums.max() * 0.2)
            words_in_string = []
            for peak_idx in peaks_ids:
                border_left = SegAnalyzer.find_border_idx(vert_sums, peak_idx, lambda i: i - 1,
                                                          SegAnalyzer.WORDS_HORIZ_THRESH)
                border_right = SegAnalyzer.find_border_idx(vert_sums, peak_idx, lambda i: i + 1,
                                                           SegAnalyzer.WORDS_HORIZ_THRESH)
                words_in_string.append((border_left, border_right))

            res = []
            for w in words_in_string:
                if w not in res:
                    res.append(w)

            SegAnalyzer._unite_words(res)
            self.words_in_strings.append(res)

    @staticmethod
    def _unite_words(words_in_string: List[Tuple[int, int]]) -> None:
        # если мы нечаянно выделили слишном малые интервалы между словами (разделили одно слово)
        while True:
            changed = False

            i = 0
            while i < len(words_in_string):
                j = i + 1
                while j < len(words_in_string):
                    si = words_in_string[i]
                    sj = words_in_string[j]

                    ti = Rect(Point(si[0], 0), Point(si[1], 2))
                    tj = Rect(Point(sj[0], 0), Point(sj[1], 2))

                    if Rect.distance(ti, tj) <= SegAnalyzer.MIN_WORDS_DIST:
                        # объединяем области
                        words_in_string.pop(j)
                        words_in_string[i] = si[0], sj[1]
                        changed = True
                    else:
                        break

                i += 1

            if not changed:
                return

    def _unite_lines(self):
        # если мы нечаянно выделили слишном маленькие строки или слишком малые интервалы
        while True:
            changed = False

            i = 0
            while i < len(self.strings_on_doc):
                j = i + 1
                while j < len(self.strings_on_doc):
                    si = self.strings_on_doc[i]
                    sj = self.strings_on_doc[j]
                    if si[1] - si[0] < SegAnalyzer.MIN_LINE_HEIGHT or sj[0] - si[1] < SegAnalyzer.MIN_LINE_DIST:
                        # объединяем области
                        self.strings_on_doc.pop(j)
                        self.strings_on_doc[i] = si[0], sj[1]
                        changed = True
                    else:
                        break

                i += 1

            if not changed:
                return


    def unite_segments(self, a: Rect, b: Rect) -> bool:
        # следует ли объединять два фрагмента a, b в один (слитны ли они)
        if self.diacritic_check(a, b):
            # один из блоков является диакритическим знаком какой-то буквы другого блока
            return True
        elif self.out_of_line_check(a, b):
            # один из блоков является надстрочным или подстрочным фрагментом для символа из другого
            return True
        elif self.in_the_same_line(a, b) and self.are_near_words(a, b):
            # оба блока находятся в одной линии; между ними нет слова
            return True

        return False

    def diacritic_check(self, a: Rect, b: Rect) -> bool:
        # один из блоков является диакритическим знаком для какого-то символа из другого блока
        if not a.intersects_x(b):
            return False

        a_is_diacr = a.area() < b.area()  # a -- диакритический знак (иначе b)
        diacr, word = (a, b) if a_is_diacr else (b, a)

        line_of_diacr = self.find_line(diacr)
        if line_of_diacr == -1:
            # word не лежит ни в какой строке => это не слово, а фигня какая-то!
            return False

        # смотрим отношение высот диакритического знака и слова
        if diacr.h() / word.h() <= SegAnalyzer.DIACRITICS_PROP and diacr.is_on_top(word) and \
                self.is_above_line(line_of_diacr, diacr):
            # выполнено соотношение для высот; диакритика полностью над словом; диакритика лежит над линией
            return True
        else:
            return False

    def out_of_line_check(self, a: Rect, b: Rect) -> bool:
        line_of_a, line_of_b = self.find_line(a), self.find_line(b)
        if line_of_a == -1 and line_of_b == -1 or not a.intersects_x(b):
            # фигня какая-то, ни a, ни b не стоит ни в какой строке
            return False

        def is_outstr(a: Rect, b: Rect, line_of_b: int) -> bool:
            # а -- надстрочный или подстрочный фрагмент для b
            top_bord, bot_bord = self.neigh_borders(line_of_b)
            if a.distance_to_y(top_bord) >= SegAnalyzer.MIN_OUTSTR_DIST and \
                    a.distance_to_y(bot_bord) >= SegAnalyzer.MIN_OUTSTR_DIST and \
                    (a.center().y <= b.bottom() or a.center().y >= b.top()):  # a -- надстрочный или подстрочный фрагм.
                return True

            return False

        # если и a, и b стоят в какой-то линии, то фиг с ним, все объединяем при выполнении остальных критериев
        # (возможно, имеем подстрочный символ для "ц" или "щ")
        if line_of_a != -1 and line_of_b != -1:
            return is_outstr(a, b, line_of_a) or is_outstr(b, a, line_of_b)

        instr, outstr = (a, b) if line_of_a != -1 else (b, a)
        return is_outstr(instr, outstr, max(line_of_a, line_of_b))

    def is_above_line(self, line_i: int, zone: Rect) -> bool:
        # прямоугольник zone лежит целиком над строкой слов line_i, в междустрочном интервале над ней
        top_border = self.strings_on_doc[line_i][0]
        bot_border_prev = 0 if line_i == 0 else self.strings_on_doc[line_i - 1][1]
        return bot_border_prev <= zone.top() and zone.bottom() <= top_border

    def is_under_line(self, line_i: int, zone: Rect) -> bool:
        # прямоугольник zone лежит целиком под строкой слов line_i, в междустрочном интервале под ней
        bot_border = self.strings_on_doc[line_i][1]
        top_border_next = 999999 if line_i == len(self.strings_on_doc) - 1 \
            else self.strings_on_doc[line_i + 1][0]

        return bot_border <= zone.top() and zone.bottom() <= top_border_next

    def neigh_borders(self, line_i: int) -> Tuple[int, int]:
        top_bord = 0 if line_i == 0 else self.strings_on_doc[line_i - 1][1]  # нижняя граница строки выше
        bot_bord = 999999 if line_i == len(self.strings_on_doc) - 1 \
            else self.strings_on_doc[line_i + 1][0]
        return top_bord, bot_bord

    def in_the_same_line(self, a: Rect, b: Rect) -> bool:
        la, lb = self.find_line(a), self.find_line(b)
        return la == lb and la != -1

    def find_line(self, segm: Rect) -> int:
        # находит линию, в границах которой лежит центр сегмента segm
        for i in range(len(self.strings_on_doc)):
            t, b = self.strings_on_doc[i]
            proj = len_of_intersection(t, b, segm.top(), segm.bottom())
            if proj >= 0.5 * segm.h() / (1 + SegAnalyzer.MAX_OUTSTR_PROP):  # хотя бы 50% строчной части символа лежит в
                # линии
                return i

        return -1

    def trivial_parse(self) -> List[Rect]:
        # выделяет в документе слова тривиальным образом, подсчитывая суммы пикселов по строкам и по вертикалям строк
        res = []
        for i in range(len(self.strings_on_doc)):
            topy, boty = self.strings_on_doc[i]
            for word in self.words_in_strings[i]:
                wx_left, wx_right = word
                res.append(Rect(Point(wx_left, topy), Point(wx_right, boty)))

        return res

    def find_word(self, line_i, zone: Rect) -> int:
        xc = zone.center().x
        for i in range(len(self.words_in_strings[line_i])):
            l, r = self.words_in_strings[line_i][i]
            if l <= xc <= r:
                return i

        return -1

    def are_near_words(self, a: Rect, b: Rect) -> bool:
        # a и b в одной строке. Метод проверяет, что они являются одним словом
        # TODO: мб проверять, что они соседние слова?..
        line = self.find_line(a)
        wa, wb = self.find_word(line, a), self.find_word(line, b)
        return wa == wb and wa != -1
