from __future__ import annotations
from typing import *
from Decision import Decision
import random
import numpy as np


class Selector:
    pm: float  # вероятность успешной мутации
    pc: float  # вероятность успешной рекомбинации
    func: Callable[List[float], float]  # фитнесс функция
    population: List[Decision]  # популяция

    def __init__(self, func: Callable[List[float], float], dim_of_vector: int,
                 size_of_population: int = 1000, pc: float = 0.5, pm: float = 0.01):
        # size_of_population -- четное число
        self.pm, self.pc = pm, pc
        self.func = func

        self.init_population(size_of_population, dim_of_vector)

    def init_population(self, size_of_population: int, dim_of_vector: int) -> None:
        SIGMA_DIAP = (0.05, 0.1)

        def make_random_sample() -> Tuple[List[float], List[float]]:
            return [Decision.DIAP[0] + random.random() * (Decision.DIAP[1] - Decision.DIAP[0])
                    for _ in range(dim_of_vector)], \
                   [SIGMA_DIAP[0] + random.random() * (SIGMA_DIAP[1] - SIGMA_DIAP[0])
                    for _ in range(dim_of_vector)]


        self.population = sorted(
            [Decision(*make_random_sample(), self.func)
             for _ in range(size_of_population)])

    @staticmethod
    def estimate(func: Callable[List[float], float], x: List[float]) -> float:
        return func(x)

    @staticmethod
    def find_pairs(l: int) -> List[Tuple[int, int]]:
        items = [i for i in range(l)]
        res = []
        while len(items):
            r1, r2 = random.randint(0, len(items) - 1), random.randint(0, len(items) - 2)
            a1 = items.pop(r1)
            a2 = items.pop(r2)
            res.append((a1, a2))
        return res

    def make_selection_round(self) -> None:
        pairs = Selector.find_pairs(len(self.population))
        new_pop = []

        for ia, ib in pairs:
            a, b = self.population[ia], self.population[ib]
            n = Decision.random_recombinate(a, b, self.pc)
            if not (n is None):
                for new in n:
                    new.random_mutate(1)
                    new_pop.append(new)

        t = self.population + new_pop
        for c in t:
            c.update_estimation(lambda x: Selector.estimate(self.func, x))

        t.sort()
        self.population = t[: len(self.population)]

    def make_selection(self, n_iters: int = 1000, n_to_stop: int = 20) -> Tuple[Decision, int]:
        best = self.population[0]
        cnt_best = 0
        i = 0
        for i in range(n_iters):
            self.make_selection_round()
            t = self.population[0]

            if t < best:
                best = t
                cnt_best = 0
            else:
                cnt_best += 1

            if cnt_best > n_to_stop:
                break

            print(i, best.estimation)

        return best, i

    def get_points(self) -> List[List[float]]:
        return [p.variables.copy() for p in self.population]
