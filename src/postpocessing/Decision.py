from __future__ import annotations
from typing import *
import random
import numpy as np


lambda_mu = 2


class Decision:
    DIAP: Tuple[float, float] = (-1, 1)

    variables: List[float]  # x_i
    sigmas: List[float]  # sigma_i
    estimation: Optional[float]  # оценка данного решения

    def __init__(self, vars: List[float], sigmas: List[float],
                 estimator: Union[None, Callable[List[float], float]] = None):
        self.variables = vars
        self.sigmas = sigmas
        if estimator is not None:
            self.estimation = estimator(self.variables)
        else:
            self.estimation = None

    def update_estimation(self, estimator: Callable[List[float], float]) -> None:
        self.estimation = estimator(self.variables)

    @staticmethod
    def random_recombinate(a: Decision, b: Decision, pc: float = 0.5) -> Optional[List[Decision]]:
        r = random.random()
        if r <= pc:
            return Decision._recombinate(a, b)
        else:
            return None

    def __getitem__(self, item: int) -> float:
        return self.variables[item]

    @staticmethod
    def _recombinate(a: Decision, b: Decision) -> List[Decision]:
        res = []
        for _ in range(2 * lambda_mu):
            hv, hs = [], []
            for i in range(len(a.variables)):
                v, sigma = random.choice([(a.variables[i], a.sigmas[i]), (b.variables[i], a.sigmas[i])])
                hv.append(v), hs.append(sigma)
            res.append(Decision(hv, hs))
        return res

    def random_mutate(self, pm: float = 0.001) -> None:
        r = random.random()
        if r <= pm:
            self._mutate()

    def _mutate(self) -> None:
        self.variables += np.random.normal(0, self.sigmas, size=(len(self.variables, )))

    def __lt__(self, other: Decision) -> bool:
        return self.estimation < other.estimation
