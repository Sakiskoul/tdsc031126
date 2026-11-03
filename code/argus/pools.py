from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass
class PoolMonitor:
    lam_w: float
    alpha: float = 1e-3

    @property
    def threshold(self) -> int:

        return int(stats.poisson.ppf(1.0 - self.alpha, self.lam_w))

    def alarm(self, count: int) -> bool:
        return count > self.threshold

    def miss_probability(self, kappa: float) -> float:

        return float(stats.poisson.cdf(self.threshold, self.lam_w + kappa))


def drain_until_insolvent(balance: float, denomination: float, kappa: float) -> float:

    if kappa <= 0:
        return np.inf
    return balance / (denomination * kappa)
