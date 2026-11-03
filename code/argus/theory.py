from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar


def bernoulli_kl(x: np.ndarray, p: np.ndarray) -> np.ndarray:

    x = np.asarray(x, float)
    p = np.asarray(p, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        a = np.where(x > 0, x * np.log(x / p), 0.0)
        b = np.where(x < 1, (1 - x) * np.log((1 - x) / (1 - p)), 0.0)
    return a + b


def attacked_prob(p: np.ndarray, q: np.ndarray) -> np.ndarray:

    return p + q * (1.0 - p)


def kl_exact(p: np.ndarray, q: np.ndarray) -> float:

    return float(np.sum(bernoulli_kl(attacked_prob(p, q), p)))


def kl_bound(p: np.ndarray, q: np.ndarray) -> float:

    return float(np.sum(q**2 * (1.0 - p) / (2.0 * p)))


def chi2_budget(p: np.ndarray, q: np.ndarray) -> float:

    return float(np.sum(q**2 / p))


def is_covert(p: np.ndarray, q: np.ndarray, eps: float) -> bool:

    return chi2_budget(p, q) <= 4.0 * eps**2 * (1 + 1e-12)


def energy(p: np.ndarray, v: np.ndarray) -> float:

    return float(np.sum(p * v**2))


def flow(p: np.ndarray, v: np.ndarray) -> float:

    return float(np.sum(p * v))


def sigma_v2(p: np.ndarray, v: np.ndarray) -> float:

    return float(np.sum(v**2 * p * (1.0 - p)))


def srl_count_capacity(p: np.ndarray, Q: np.ndarray, eps: float) -> float:

    Lam = float(np.sum(p))
    P = p / Lam
    chi2 = float(np.sum(Q**2 / P) - 1.0)
    return 2.0 * eps * np.sqrt(Lam / (1.0 + chi2))


def l2_covert_value(p: np.ndarray, v: np.ndarray, eps: float) -> float:

    return 2.0 * eps * np.sqrt(energy(p, v))


def tilted_attack(p: np.ndarray, v: np.ndarray, eps: float) -> np.ndarray:

    q = 2.0 * eps * p * v / np.sqrt(energy(p, v))
    if np.any(q > 1):
        raise ValueError("tilted attack infeasible, some q* exceeds one")
    return q


def detectable_value(p: np.ndarray, v: np.ndarray, delta: float) -> float:

    return 2.0 * np.sqrt(sigma_v2(p, v)) / ((1.0 - p.max()) * np.sqrt(delta))


def count_blindness(p: np.ndarray, dagger: int) -> float:

    lam_minus = float(np.sum(p) - p[dagger])
    return 1.0 - np.sqrt(2.0 / np.pi) / np.sqrt(0.25 + lam_minus)


def _g(nu: float, p, v, r) -> float:
    return float(np.sqrt(np.sum(p * np.clip(v - nu * r, 0.0, None) ** 2)))


def witness_dual(p, v, r, eps: float, beta: float):


    p, v, r = (np.asarray(x, float) for x in (p, v, r))
    phi = lambda nu: 2.0 * eps * _g(nu, p, v, r) + nu * beta
    pos = r > 0
    hi = float(np.max(v[pos] / r[pos])) if np.any(pos) else 0.0
    if hi <= 0.0:
        nu = 0.0
    else:
        res = minimize_scalar(phi, bounds=(0.0, hi), method="bounded",
                              options={"xatol": 1e-12 * max(hi, 1.0), "maxiter": 2000})
        nu = float(res.x)
        if phi(0.0) <= phi(nu):
            nu = 0.0
    gs = _g(nu, p, v, r)
    if gs <= 0:
        return phi(nu), nu, np.zeros_like(p)
    q = 2.0 * eps * p * np.clip(v - nu * r, 0.0, None) / gs
    return phi(nu), nu, q


def borrowed_cover_bound(p, v, r, eps: float, beta: float) -> float:

    p, v, r = (np.asarray(x, float) for x in (p, v, r))
    dead = r == 0
    first = 2.0 * eps * np.sqrt(np.sum(p[dead] * v[dead] ** 2))
    alive = ~dead
    second = beta * float(np.max(v[alive] / r[alive])) if np.any(alive) else 0.0
    return first + second


def mimic_attack(p0: np.ndarray, theta0: float, eta: float) -> np.ndarray:

    return (1.0 + eta - theta0) * p0 / (1.0 - theta0 * p0)


def control_detectable_value(p0, v, theta: float, lam_c0: float, eta: float, delta: float) -> float:

    p = theta * p0
    var = sigma_v2(p, v) + theta * flow(p0, v) ** 2 / lam_c0
    return 2.0 * np.sqrt(var / delta) / (1.0 - (1.0 + eta) * p0.max())


def poisson_kl(lam: float, kappa: float) -> float:

    return float((lam + kappa) * np.log1p(kappa / lam) - kappa)


def pool_covert_rate(lam_w: float, eps: float) -> float:

    return 2.0 * eps * np.sqrt(lam_w)
