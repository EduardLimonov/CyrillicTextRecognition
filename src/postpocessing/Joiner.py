from __future__ import annotations
from typing import *
from Selector import *


MAX_LEFT_INDENT = 'max_left_indent'  # максимальный отступ левой границы первого символа от левой границы изображения
MAX_RIGHT_INDENT = 'max_right_indent'  # максимальный отступ правой границы последнего символа от правой границы
                                        # изображения
LIMIT_MOVING_LEFT = 'limit_moving_left'  # максимальное смещение влево левой границы следующего символа относительно
                                         # правой границы предыдущего
LIMIT_MOVING_RIGHT = 'limit_moving_right'  # максимальное смещение вправо левой границы следующего символа относительно
                                           # правой границы предыдущего

MLI_k, MRI_k, LML_k, LMR_k = 100, 100, 100, 100

import pickle
from asrtoolkit import cer
import numpy as np
file = open('D:\\projects\\datasets\\ga_test', 'rb')
DATA = pickle.load(file)
NO_ANSWER_PENALTY = 10


def estimate(vector: List[float], data: List[Tuple[List[Tuple[int, int, str, float]], int, str]] = DATA):
    score = 0
    for sample in data:
        symbols, imgwidth, result = sample
        params = Joiner.read_from_vector(vector)
        words = Joiner.join_components(symbols, params, imgwidth, 5)
        cers = [cer(word, result) for word, k in words]

        if len(words):
            score += np.max(cers)
        else:
            score += cer('', result)
    return score


class Joiner:

    @staticmethod
    def join_components(results: List[Tuple[int, int, str, float]], params: Dict, imgwidth: int,
                        max_num_of_variants: int) -> List[Tuple[str, float]]:

        def get_first_symbols() -> List[int]:
            return [i for i in range(len(results)) if results[i][0] <= params[MAX_LEFT_INDENT]]

        def get_next_symbol(done: List[int]) -> List:
            actual_right_border = results[done[-1]][1]
            return [i for i in range(len(results)) if i not in done and
                    params[LIMIT_MOVING_LEFT] <= results[i][0] - actual_right_border <= params[LIMIT_MOVING_RIGHT]]

        def may_be_end(done: List[int]) -> bool:
            return imgwidth - results[done[-1]][1] <= params[MAX_RIGHT_INDENT]

        def step_to(done) -> List[int]:
            # возвращает список возможных окончаний пройденного маршрута done
            mbroutes = [done + [var] for var in get_next_symbol(done)]
            res = []
            for route in mbroutes:
                res += step_to(route)
            if may_be_end(done) or len(mbroutes) == 0:
                res += [done]

            return res

        first_steps = get_first_symbols()  # представление дерева вариантов распознанных слов
        res = []

        for step in first_steps:
            res += step_to([step])

        words = []
        for route in res:
            word = ''
            confidence = 0
            for i in route:
                word += results[i][2]
                confidence += results[i][3]

            words.append((word, confidence / len(word)))

        words.sort(reverse=True, key=lambda x: x[1])
        if len(words) < max_num_of_variants:
            return words
        else:
            return words[: max_num_of_variants]

    @staticmethod
    def read_from_vector(vector: List[float]) -> Dict:
        return {
            MAX_LEFT_INDENT: vector[0] * MLI_k,
            MAX_RIGHT_INDENT: vector[1] * MRI_k,
            LIMIT_MOVING_LEFT: vector[2] * LML_k,
            LIMIT_MOVING_RIGHT: vector[3] * LMR_k
        }


if __name__ == '__main__':
    print(Joiner.read_from_vector([0.882, -1.582, -1.383,    1.892]))
    exit()

    import time
    start = time.time()
    res = Selector(estimate, 4, pc = 0.5, pm=0.05, size_of_population=10).make_selection(50)
    print(res[0].estimation)
    print(res[0])
    print(res[1])
    print('time: %f' % (time.time() - start))
