from __future__ import annotations

import cvxpy as cp
import numpy as np


def exact_kl_attack(p, v, kl_budget: float, cap=1.0, r=None, beta=None, solver=None):


    p = np.asarray(p, float)
    v = np.asarray(v, float)
    n = p.size
    q = cp.Variable(n)
    x = p + cp.multiply(q, 1.0 - p)
    kl = cp.sum(cp.rel_entr(x, p) + cp.rel_entr(1.0 - x, 1.0 - p))
    cons = [kl <= kl_budget, q >= 0, q <= cap]
    if r is not None and beta is not None:
        cons.append(np.asarray(r, float) @ q <= beta)
    prob = cp.Problem(cp.Maximize(v @ q), cons)
    prob.solve(solver=solver)
    if q.value is None:
        raise RuntimeError(f"solver failed: {prob.status}")
    qv = np.clip(np.asarray(q.value).ravel(), 0.0, cap)
    return float(v @ qv), qv


def chi2_attack(p, v, eps: float, cap=1.0, r=None, beta=None, solver=None):

    p = np.asarray(p, float)
    v = np.asarray(v, float)
    q = cp.Variable(p.size)
    cons = [cp.sum(cp.multiply(1.0 / p, cp.square(q))) <= 4 * eps**2, q >= 0, q <= cap]
    if r is not None and beta is not None:
        cons.append(np.asarray(r, float) @ q <= beta)
    prob = cp.Problem(cp.Maximize(v @ q), cons)
    prob.solve(solver=solver)
    qv = np.clip(np.asarray(q.value).ravel(), 0.0, cap)
    return float(v @ qv), qv


def greedy_top(v, k: int) -> np.ndarray:

    q = np.zeros(len(v))
    q[np.argsort(v)[::-1][:k]] = 1.0
    return q


def uniform_exposed(n: int, k_expected: float) -> np.ndarray:

    return np.full(n, min(1.0, k_expected / n))


def sample_theft(q: np.ndarray, rng) -> np.ndarray:

    return rng.random(q.size) < q
