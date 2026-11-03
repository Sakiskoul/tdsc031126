from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Statistic(Protocol):


    def statistic(self, x: np.ndarray, *control: np.ndarray) -> np.ndarray: ...


@dataclass
class ValueWeightedMonitor:


    p: np.ndarray
    v: np.ndarray

    def __post_init__(self):
        self.p = np.asarray(self.p, float)
        self.v = np.asarray(self.v, float)
        if np.any(self.p > 0.5) or np.any(self.p <= 0):
            raise ValueError("null probabilities must lie in (0, 1/2]")

    @property
    def sigma(self) -> float:
        return float(np.sqrt(np.sum(self.v**2 * self.p * (1 - self.p))))

    def statistic(self, x: np.ndarray) -> np.ndarray:

        return (np.asarray(x, float) - self.p) @ self.v

    def threshold_for_value(self, v0: float) -> float:

        return (1.0 - self.p.max()) * v0 / 2.0

    def chebyshev_error(self, v0: float) -> float:

        return 4.0 * self.sigma**2 / ((1.0 - self.p.max()) * v0) ** 2

    def calibrate(self, alpha: float, n_sim: int = 20000, rng=None) -> float:

        rng = np.random.default_rng(rng)
        x = rng.random((n_sim, self.p.size)) < self.p
        return float(np.quantile(self.statistic(x), 1.0 - alpha))


@dataclass
class ControlCalibratedMonitor:


    p0: np.ndarray
    v: np.ndarray
    p0_control: np.ndarray

    def __post_init__(self):
        self.p0 = np.asarray(self.p0, float)
        self.v = np.asarray(self.v, float)
        self.p0_control = np.asarray(self.p0_control, float)
        self.lam_c0 = float(self.p0_control.sum())
        self.f_v0 = float(self.p0 @ self.v)

    def theta_hat(self, xc: np.ndarray) -> np.ndarray:
        return np.asarray(xc, float).sum(axis=-1) / self.lam_c0

    def statistic(self, x: np.ndarray, xc: np.ndarray) -> np.ndarray:
        x = np.asarray(x, float)
        return x @ self.v - self.theta_hat(xc) * self.f_v0

    def variance_bound(self, theta: float) -> float:
        p = theta * self.p0
        return float(np.sum(self.v**2 * p * (1 - p)) + theta * self.f_v0**2 / self.lam_c0)

    def threshold_for_value(self, v0: float, eta: float) -> float:
        return (1.0 - (1.0 + eta) * self.p0.max()) * v0 / 2.0


@dataclass
class Cusum:


    drift: float = 0.5
    h: float = 5.0
    g: float = 0.0
    history: list = field(default_factory=list)

    def update(self, z: float) -> bool:
        self.g = max(0.0, self.g + z - self.drift)
        self.history.append(self.g)
        return self.g > self.h

    def reset(self) -> None:
        self.g = 0.0
