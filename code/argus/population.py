from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Population:
    p: np.ndarray
    p_true: np.ndarray
    v: np.ndarray
    r: np.ndarray
    p_control: np.ndarray

    @property
    def n(self) -> int:
        return self.p.size


def synthetic(n: int = 5000, n_control: int = 50000, dead_frac: float = 0.3,
              pareto_alpha: float = 1.2, v_scale: float = 1.0, p_mean: float = 0.01,
              witness_horizon: float = 4.0, seed: int = 0) -> Population:
    rng = np.random.default_rng(seed)
    v = v_scale * (rng.pareto(pareto_alpha, n) + 1.0)
    alive_p = np.clip(rng.gamma(2.0, p_mean / 2.0, n), 1e-5, 0.4)
    dead = rng.random(n) < dead_frac
    p_true = np.where(dead, 0.0, alive_p)

    p_monitor = np.full(n, max(p_true.mean(), 1e-5))
    r = np.where(dead, 0.0, 1.0 - (1.0 - alive_p) ** witness_horizon)
    p_control = np.clip(rng.gamma(2.0, p_mean / 2.0, n_control), 1e-5, 0.4)
    return Population(p=p_monitor, p_true=p_true, v=v, r=r, p_control=p_control)
