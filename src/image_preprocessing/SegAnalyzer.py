from __future__ import annotations
from typing import *
import numpy as np
from utils.geometry import Rect, Point
from utils.algs import postprocess_segmentation

MIN_LINE_DIST = 'MIN_LINE_DIST'  # минимальный межстрочный интервал в пикселах
MIN_LINE_HEIGHT = 'MIN_LINE_HEIGHT'
MIN_WORDS_DIST = 'MIN_WORDS_DIST'
MIN_HOR_SUM = 'MIN_HOR_SUM'
MIN_VERT_SUM = 'MIN_VERT_SUM'

DEFAULT_PARAMS = {
    MIN_LINE_DIST: 1,
    MIN_LINE_HEIGHT: 12,
    MIN_WORDS_DIST: 8,
    MIN_HOR_SUM: 10,
    MIN_VERT_SUM: 3
}


class SegAnalyzer:
    # Экземпляру этого класса нужно скормить бинарный документ (1 -- пиксел закрашен, 0 -- незакрашен).
    # Объект определяет расположение строк и слов (на основе подсчитанных сумм пикселей по горизонтали и вертикали)

    strings_on_doc: List[Tuple[int, int]]  # для каждой строки текста: y-координата начала и конца
    words_in_strings: List[List[Tuple[int, int]]]  # начало и конец слова в каждой строке
    img: np.ndarray
    params: Dict

    DIACRITICS_PROP = 0.1  # максимальная высота диакритического знака относительно высоты символа
    MIN_OUTSTR_DIST = 5
    MAX_OUTSTR_PROP = 1.8

    def __init__(self, doc: np.ndarray, params: Dict = None):
        if params is None:
            params = DEFAULT_PARAMS
        self.img = doc
        self.params = params
        self._analyze()

    @staticmethod
    def find_areas(sums: np.ndarray, thresh: float, min_val: int) -> List[Tuple[int, int]]:
        """
        Находит области строк или слов в строке

        :param min_val: число, значение сумм больше которого означает, что в данной позиции находится локальный максимум
        :param sums: массив сумм закрашенных пикселей
        :param thresh: критерий определения края зоны
        :return: множество областей локальных максимумов, выделенных в sums
        """

        res = []
        start = None
        line_beg = False
        thresh = thresh * np.mean(sums)

        for l in range(len(sums)):
            if sums[l] > thresh or sums[l] > min_val:
                if not line_beg:
                    line_beg = True
                    start = l
            else:
                if line_beg:
                    line_beg = False
                    res.append((start, l))

        if line_beg:
            res.append((start, len(sums) - 1))
        return res

    def _analyze(self) -> None:
        # заполняет список строк self.strings_on_doc
        doc = self.img
        params = self.params
        sums = doc.sum(axis=1)  # построчные суммы; в тех строках пикселов, где большие суммы, идет строка
        self.strings_on_doc = SegAnalyzer.find_areas(sums, thresh=0.01, min_val=params[MIN_HOR_SUM])
        SegAnalyzer._unite_lines(self.strings_on_doc, params)

        self.words_in_strings = []

        for top, bot in self.strings_on_doc:
            string: np.ndarray = doc[top: bot]
            vert_sums = string.sum(axis=0)

            words = SegAnalyzer.find_areas(vert_sums, thresh=0.01, min_val=params[MIN_VERT_SUM])

            SegAnalyzer._unite_words(words, params, bot - top)
            self.words_in_strings.append(words)

    @staticmethod
    def _unite_pseudo_words(words: List[Rect], params: Dict) -> None:
        while True:
            changed = False

            i = 0
            while i < len(words):
                j = i + 1
                while j < len(words):
                    ti = words[i]
                    tj = words[j]

                    if SegAnalyzer.unite_segments(ti, tj, params[MIN_WORDS_DIST], words):
                        # Rect.distance(ti, tj) <= params[MIN_WORDS_DIST]:
                        # объединяем области
                        words[i] = words[i].union(words[j])
                        words.pop(j)
                        changed = True

                    else:
                        j += 1

                i += 1

            if not changed:
                return

    @staticmethod
    def _unite_words(words_in_string: List[Tuple[int, int]], params: Dict, line_h: int,
                     rects_in_line: Optional[List[Rect]] = None) -> None:
        # если мы нечаянно выделили слишком малые интервалы между словами (разделили одно слово)
        unite_rects = not (rects_in_line is None)

        if not unite_rects:
            rects_in_line = [Rect(Point(l, 0), Point(r, line_h)) for l, r in words_in_string]

        SegAnalyzer._unite_pseudo_words(rects_in_line, params)

        if unite_rects:
            return
        else:
            words_in_string.clear()
            for r in rects_in_line:
                words_in_string.append((r.left(), r.right()))

    @staticmethod
    def _unite_lines(strings_on_doc: List[Tuple[int, int]], params: Dict):
        # если мы нечаянно выделили слишком маленькие строки или слишком малые интервалы
        while True:
            changed = False

            i = 0
            while i < len(strings_on_doc):
                j = i + 1
                while j < len(strings_on_doc):
                    si = strings_on_doc[i]
                    sj = strings_on_doc[j]
                    if si[1] - si[0] < params[MIN_LINE_HEIGHT] or sj[0] - si[1] < params[MIN_LINE_DIST]:
                        # объединяем области
                        strings_on_doc.pop(j)
                        strings_on_doc[i] = si[0], sj[1]
                        changed = True
                    else:
                        break

                i += 1

            if not changed:
                return

    @staticmethod
    def unite_segments(a: Rect, b: Rect, sens: float, words: List[Rect]) -> bool:
        # следует ли объединять два фрагмента a, b в один (слитны ли они)
        if SegAnalyzer._is_diacritic(a, b, sens * 3) or SegAnalyzer._is_diacritic(b, a, sens * 3):
            return True
        elif SegAnalyzer.is_punctuation(a, b) or SegAnalyzer.is_punctuation(b, a):
            return False
        elif Rect.distance(a, b) <= sens:
            # фрагменты близко находятся
            return True
        else:
            wa, wb = SegAnalyzer.find_word(words, a), SegAnalyzer.find_word(words, b)
            if wa == wb and wa != -1:
                return True

        return False

    def trivial_parse(self) -> List[Rect]:
        # выделяет в документе слова тривиальным образом, подсчитывая суммы пикселов по строкам и по вертикалям строк,
        # затем уточняет границы
        res = []
        for i in range(len(self.strings_on_doc)):
            topy, boty = self.strings_on_doc[i]
            words_in_line = []
            for wx_left, wx_right in self.words_in_strings[i]:
                zone = Rect(Point(wx_left, topy), Point(wx_right, boty))
                zone = self.specify_borders(zone, step=1)
                words_in_line.append(zone)

            SegAnalyzer._unite_words(self.words_in_strings[i], self.params, boty - topy, words_in_line)
            res += words_in_line

        res = postprocess_segmentation(
            res,
            criterion=lambda a, b, sens=self.params[MIN_WORDS_DIST]:
            SegAnalyzer.unite_segments(a, b, sens, res)
        )
        if len(res) > 1:
            res = self._check_line_errors(res)

        return res

    def _check_line_errors(self, zones: List[Rect]) -> List[Rect]:
        # возможно, мы некорректно выделили границы строк, и два слова в соседних строках было выделено как одно
        res = []
        for zone in zones:
            zone_words = SegAnalyzer(self.img[zone.top(): zone.bottom() + 1, zone.left(): zone.right() + 1],
                                     self.params).trivial_parse()
            if len(zone_words) > 1:
                # мы нашли больше одного слова в сегменте => была ошибка разметки линий
                for r in zone_words:
                    r.move(zone.left(), zone.top())
                res += zone_words
            else:
                res.append(zone)
        return res

    @staticmethod
    def find_word(words: List[Rect], zone: Rect, iou_thresh: float = 0.5) -> int:
        for i in range(len(words)):
            if words[i].iou_val(zone) >= iou_thresh:
                return i

        return -1

    def specify_borders(self, zone: Rect, step: int = 5) -> Rect:
        while True:
            t = self.__specify(self.img[:, zone.left(): zone.right()], zone.top(), out_is_upper=True, step=step)
            b = self.__specify(self.img[:, zone.left(): zone.right()], zone.bottom(), out_is_upper=False, step=step)
            l = self.__specify(self.img[t: b].transpose(), zone.left(), out_is_upper=True, step=step)
            r = self.__specify(self.img[t: b].transpose(), zone.right(), out_is_upper=False, step=step)
            new = Rect(Point(l, t), Point(r, b))

            if new == zone:
                break
            else:
                zone = new
        return new

    @staticmethod
    def __specify(arr_norm: np.ndarray, tbord: int, out_is_upper: bool, step: int = 1) -> int:
        # arr_norm -- нормализованная матрица: она повернута так, что интересующая нас ось -- вертикальная

        def bound_tbord(tbord) -> int:
            if tbord < 0:
                tbord = 0
            elif tbord >= arr_norm.shape[0]:
                tbord = arr_norm.shape[0] - 1

            return tbord

        tbord = bound_tbord(tbord)

        while np.count_nonzero(arr_norm[tbord]):
            # есть пересечение с линией на границе зоны, значит границу надо отодвинуть наружу
            tbord -= step if out_is_upper else -step
            if tbord < 0 or tbord >= arr_norm.shape[0]:
                tbord = bound_tbord(tbord)
                break

        while not np.count_nonzero(arr_norm[tbord]) and tbord not in (0, arr_norm.shape[0]):
            # нет пересечения с линией символа => границу надо сузить
            tbord += 1 if out_is_upper else -1
            if tbord < 0 or tbord >= arr_norm.shape[0]:
                tbord = bound_tbord(tbord)
                break

        return tbord

    @staticmethod
    def _is_diacritic(_prev_word: Rect, _segment: Rect, sens: Union[int, float]) -> bool:
        # _segment -- диакритический знак _prev_word
        return _segment.bottom() <= _prev_word.center().y and _segment.intersects_x(_prev_word) and \
               _segment.h() * 3 < _prev_word.h() and Rect.distance(_prev_word, _segment) <= sens

    @staticmethod
    def _is_point_comma(_prev_word: Rect, _segment: Rect) -> bool:
        # _segment -- точка или запятая: его площадь намного меньше предыдущего слова и он лежит правее этого слова
        epsilon = min(_prev_word.h() / 4, _prev_word.w() / 8)
        # epsilon: строго говоря, бывают такие ситуации, когда точка или запятая налезает на предыдущее слово,
        # epsilon -- максимальный допустимый сдвиг пунктуации влево
        return _segment.area() < 0.05 * _prev_word.area() and _segment.left() >= _prev_word.right() - epsilon

    @staticmethod
    def _is_dash(_prev_word: Rect, _segment: Rect) -> bool:
        # _segment -- тире или дефис
        return _prev_word.intersects_y(_segment) and _segment.is_on_right(_prev_word) \
               and _segment.w() > _segment.h() and _segment.h() * 3 < _prev_word.h()

    @staticmethod
    def is_punctuation(prev_word: Rect, segment: Rect) -> bool:
        return SegAnalyzer._is_point_comma(prev_word, segment) or SegAnalyzer._is_dash(prev_word, segment)
