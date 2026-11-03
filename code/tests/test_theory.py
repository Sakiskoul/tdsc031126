import itertools
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from argus import theory as T
from argus.monitor import ControlCalibratedMonitor, ValueWeightedMonitor
from argus.redteam import chi2_attack, exact_kl_attack

RNG = np.random.default_rng(20261016)


def product_pmf(probs):

    n = len(probs)
    outs = np.array(list(itertools.product([0, 1], repeat=n)), dtype=float)
    pmf = np.prod(np.where(outs == 1, probs, 1 - probs), axis=1)
    return outs, pmf


def tv_exact(p, q):
    outs, f0 = product_pmf(p)
    _, f1 = product_pmf(T.attacked_prob(p, q))
    return 0.5 * np.abs(f0 - f1).sum()


def small_instance(n=12, seed=0):
    rng = np.random.default_rng(seed)
    p = rng.uniform(0.02, 0.3, n)
    v = rng.pareto(1.3, n) + 1.0
    return p, v


@pytest.mark.parametrize("seed", range(20))
def test_lemma1_kl_bound(seed):
    rng = np.random.default_rng(seed)
    p = rng.uniform(1e-4, 0.25, 200)
    q = rng.uniform(0, 1, 200) * rng.random(200)
    assert T.kl_exact(p, q) <= T.kl_bound(p, q) + 1e-12
    assert T.kl_bound(p, q) <= T.chi2_budget(p, q) + 1e-12


@pytest.mark.parametrize("seed", range(10))
def test_thm1_covert_by_exact_tv(seed):
    p, _ = small_instance(seed=seed)
    eps = 0.2
    q = np.random.default_rng(seed + 100).random(p.size) * p
    q *= 2 * eps / np.sqrt(T.chi2_budget(p, q))
    assert T.is_covert(p, q, eps)
    assert tv_exact(p, q) <= eps + 1e-12


def test_thm1_identity_count_form():
    p, _ = small_instance(n=50, seed=3)
    q = np.random.default_rng(7).random(50) * p
    K = q.sum()
    Lam = p.sum()
    Q, P = q / K, p / Lam
    chi2 = np.sum(Q**2 / P) - 1
    assert np.isclose(T.chi2_budget(p, q), K**2 / Lam * (1 + chi2))


@pytest.mark.parametrize("scale", [1.0, 3.0, 10.0])
def test_thm1_converse_monte_carlo(scale):
    rng = np.random.default_rng(int(scale * 10))
    n = 4000
    p = rng.uniform(0.005, 0.05, n)
    q = np.minimum(p, scale * 0.002 * p / p.mean())
    C = np.sum(q**2 * (1 - p) / p)
    w = q / p
    trials = 4000
    x0 = rng.random((trials, n)) < p
    x1 = rng.random((trials, n)) < T.attacked_prob(p, q)
    t0 = (x0 - p) @ w
    t1 = (x1 - p) @ w
    err = np.mean(t0 > C / 2) + np.mean(t1 <= C / 2)
    assert err <= 20 / C + 0.03


@pytest.mark.parametrize("seed", range(10))
def test_thm2_l2_law_matches_convex_optimum(seed):
    rng = np.random.default_rng(seed)
    n = 300
    p = rng.uniform(0.001, 0.05, n)
    v = rng.pareto(1.2, n) + 1.0
    eps = 0.05
    closed = T.l2_covert_value(p, v, eps)
    q_star = T.tilted_attack(p, v, eps)
    assert np.isclose(q_star @ v, closed, rtol=1e-10)
    assert np.isclose(T.chi2_budget(p, q_star), 4 * eps**2, rtol=1e-10)
    opt, _ = chi2_attack(p, v, eps, cap=1.0)
    assert np.isclose(opt, closed, rtol=1e-4)


def test_thm2_whale_example_and_energy_range():

    p = np.full(10_001, 0.01)
    v = np.ones(10_001)
    v[-1] = 1e4
    eps = 0.1
    assert np.isclose(T.l2_covert_value(p[:-1], v[:-1], eps), 2.0)
    assert np.isclose(T.l2_covert_value(p, v, eps), 200.0, rtol=1e-4)
    q = T.tilted_attack(p, v, eps)
    assert np.isclose(q[-1], 0.02, rtol=1e-3) and np.isclose(q[0], 2e-6, rtol=1e-3)
    assert q[-1] * v[-1] / (q @ v) > 0.999

    for seed in range(20):
        rng = np.random.default_rng(seed)
        pp = rng.uniform(0.001, 0.25, 200)
        vv = rng.pareto(1.1, 200) + 1.0
        e, f = np.sqrt(T.energy(pp, vv)), T.flow(pp, vv)
        assert f / np.sqrt(pp.sum()) <= e * (1 + 1e-12) and e <= np.sqrt(vv.max() * f) * (1 + 1e-12)
    vv = np.full(50, 3.0)
    e, f = np.sqrt(T.energy(pp[:50], vv)), T.flow(pp[:50], vv)
    assert np.isclose(f / np.sqrt(pp[:50].sum()), e) and np.isclose(np.sqrt(3.0 * f), e)


def test_thm2_exact_kl_close_to_l2_in_small_signal_regime():
    rng = np.random.default_rng(5)
    n = 400
    p = rng.uniform(0.005, 0.05, n)
    v = rng.pareto(1.5, n) + 1.0
    eps = 0.05
    closed = T.l2_covert_value(p, v, eps)

    exact, _ = exact_kl_attack(p, v, kl_budget=2 * eps**2)
    assert closed <= exact * (1 + 1e-3)
    assert exact <= closed * 1.10


def test_thm3_value_monitor_uniform_over_attacks():
    rng = np.random.default_rng(11)
    n = 3000
    p = rng.uniform(0.002, 0.03, n)
    v = rng.pareto(1.3, n) + 1.0
    mon = ValueWeightedMonitor(p, v)
    delta = 0.05
    v0 = T.detectable_value(p, v, delta)
    tau = mon.threshold_for_value(v0)
    trials = 3000
    x0 = rng.random((trials, n)) < p
    assert np.mean(mon.statistic(x0) > tau) <= delta

    order = np.argsort(v)[::-1]
    sets = []
    for pick in (order, rng.permutation(n), order[::-1]):
        s, tot = [], 0.0
        for a in pick:
            s.append(a)
            tot += v[a]
            if tot >= v0:
                break
        sets.append(np.array(s))
    for s in sets:
        x1 = rng.random((trials, n)) < p
        x1[:, s] = True
        assert np.mean(mon.statistic(x1) <= tau) <= delta


def test_prop1_count_blindness_exact():
    rng = np.random.default_rng(2)
    n = 400
    p = rng.uniform(0.01, 0.2, n)
    dagger = 0

    pmf_w = np.array([1.0])
    for a in range(1, n):
        pmf_w = np.convolve(pmf_w, [1 - p[a], p[a]])
    null = np.convolve(pmf_w, [1 - p[dagger], p[dagger]])
    theft = np.concatenate([[0.0], pmf_w])
    tv = 0.5 * np.abs(null - theft).sum()
    assert 1 - tv >= T.count_blindness(p, dagger) - 1e-12


@pytest.mark.parametrize("seed", range(8))
def test_witness_dual_matches_primal(seed):
    rng = np.random.default_rng(seed)
    n = 200
    p = rng.uniform(0.001, 0.05, n)
    v = rng.pareto(1.3, n) + 1.0
    r = np.where(rng.random(n) < 0.3, 0.0, np.clip(3 * p + rng.uniform(0, 0.05, n), 0, 1))
    eps, beta = 0.05, 0.02
    dual, nu, q = T.witness_dual(p, v, r, eps, beta)
    primal, q_cvx = chi2_attack(p, v, eps, cap=10.0, r=r, beta=beta)
    assert np.isclose(dual, primal, rtol=2e-3)
    assert T.chi2_budget(p, q) <= 4 * eps**2 * (1 + 1e-6)
    assert r @ q <= beta * (1 + 1e-4)
    assert np.isclose(q @ v, dual, rtol=1e-6)
    assert dual <= T.borrowed_cover_bound(p, v, r, eps, beta) * (1 + 1e-9)


def test_cor2_no_witness_tolerance_leaves_only_borrowed_cover():
    rng = np.random.default_rng(4)
    n = 300
    p = rng.uniform(0.001, 0.05, n)
    v = rng.pareto(1.3, n) + 1.0
    r = np.where(rng.random(n) < 0.4, 0.0, 2 * p)
    eps = 0.05
    dead = r == 0
    val, _, q = T.witness_dual(p, v, r, eps, beta=1e-12)
    assert np.isclose(val, 2 * eps * np.sqrt(np.sum(p[dead] * v[dead] ** 2)), rtol=1e-3)
    assert np.all(q[~dead] < 1e-6)


def test_uncert_mimic_reproduces_null_exactly():
    rng = np.random.default_rng(9)
    p0 = rng.uniform(0.001, 0.2, 1000)
    eta, theta0 = 0.08, 1.0
    q = T.mimic_attack(p0, theta0, eta)
    assert np.allclose(T.attacked_prob(theta0 * p0, q), (1 + eta) * p0)


def test_control_restores_detection():
    rng = np.random.default_rng(13)
    n, nc = 3000, 60000
    p0 = rng.uniform(0.002, 0.03, n)
    v = rng.pareto(1.3, n) + 1.0
    p0c = rng.uniform(0.002, 0.03, nc)
    eta, delta = 0.10, 0.05
    mon = ControlCalibratedMonitor(p0, v, p0c)
    v0 = T.control_detectable_value(p0, v, theta=1 + eta, lam_c0=p0c.sum(), eta=eta, delta=delta)
    tau = mon.threshold_for_value(v0, eta)
    trials = 2000
    for theta in (1 - eta, 1.0, 1 + eta):
        x = rng.random((trials, n)) < theta * p0
        xc = rng.random((trials, nc)) < theta * p0c
        assert np.mean(mon.statistic(x, xc) > tau) <= delta

    q = T.mimic_attack(p0, 1.0, eta)
    stolen = q @ v
    x = rng.random((trials, n)) < T.attacked_prob(p0, q)
    xc = rng.random((trials, nc)) < p0c
    detected = np.mean(mon.statistic(x, xc) > tau)
    if stolen >= 1.5 * v0:
        assert detected >= 1 - delta


@pytest.mark.parametrize("lam", [10.0, 100.0, 1000.0])
def test_anon_poisson_kl_bound(lam):
    for kappa in np.linspace(0.1, lam, 30):
        assert T.poisson_kl(lam, kappa) <= kappa**2 / lam + 1e-12
    kappa = T.pool_covert_rate(lam, 0.1)
    tv = 0.5 * np.abs(stats.poisson.pmf(np.arange(int(5 * lam) + 50), lam)
                      - stats.poisson.pmf(np.arange(int(5 * lam) + 50), lam + kappa)).sum()
    assert tv <= 0.1 + 1e-9


def test_lemma1_pointwise_half_chi2():
    for p in np.concatenate([np.logspace(-6, -1, 60), np.linspace(0.1, 0.25, 40)]):
        x = np.linspace(p, 1.0, 4000)
        assert np.all(T.bernoulli_kl(x, np.full_like(x, p)) <= (x - p) ** 2 / (2 * p * (1 - p)) + 1e-12)


def _exact_tests(p, v, V0):

    outs, pmf0 = product_pmf(p)
    tau = (1 - p.max()) * V0 / 2
    stat = (outs - p) @ v
    fa = float(pmf0[stat > tau].sum())
    worst = 0.0
    n = len(p)
    for S in itertools.product([0, 1], repeat=n):
        S = np.array(S, bool)
        if v[S].sum() < V0:
            continue
        q = np.where(S, 1.0, p)
        _, pmfS = product_pmf(q)
        worst = max(worst, float(pmfS[stat <= tau].sum()))
    return outs, pmf0, stat, tau, fa, worst


def test_loss_uniform_bound_exact_for_lotteries():
    rng = np.random.default_rng(11)
    p = rng.uniform(0.01, 0.2, 7)
    v = rng.pareto(1.5, 7) + 1.0
    V0 = 0.6 * v.sum()
    outs, pmf0, stat, tau, fa, worst = _exact_tests(p, v, V0)
    delta = max(fa, worst)
    sets = [np.array(S, bool) for S in itertools.product([0, 1], repeat=7)]
    for trial in range(30):

        w = rng.dirichlet(np.ones(len(sets)) * 0.05)
        idle = rng.uniform(0.5, 0.99)
        pmf = idle * pmf0
        big, unalarmed = 0.0, 0.0
        for S, ws in zip(sets, w):
            q = np.where(S, 1.0, p)
            _, pS = product_pmf(q)
            pmf = pmf + (1 - idle) * ws * pS
            if v[S].sum() >= V0:
                big += (1 - idle) * ws
                unalarmed += (1 - idle) * ws * pS[stat <= tau].sum()
        eps = 0.5 * np.abs(pmf - pmf0).sum()
        assert unalarmed <= delta + 1e-12
        assert big <= (delta + eps) / (1 - delta) + 1e-12


def test_single_output_lottery_beats_l2_budget_but_rarely_pays():

    p, v, eps = 0.01, 100.0, 0.1
    q = eps / (1 - p)
    assert abs((p + q * (1 - p)) - p - eps) < 1e-12
    assert q * v > 5 * 2 * eps * np.sqrt(p * v * v)
    assert 1 - q > 0.89


def test_tight_tilted_loss_concentrates_in_diffuse_population():
    rng = np.random.default_rng(12)
    n, eps, eta = 40000, 0.1, 0.3
    p = rng.uniform(0.002, 0.02, n)
    v = rng.uniform(0.5, 2.0, n)
    E = float(p @ (v * v))
    kappa = float(p @ v**3) / E**1.5
    q = T.tilted_attack(p, v, eps)
    V = np.array([v[rng.random(n) < q].sum() for _ in range(3000)])
    emp = np.mean(V <= (1 - eta) * 2 * eps * np.sqrt(E))
    assert emp <= np.exp(-eta**2 * eps / kappa) + 0.01


def test_budget_truncation_tail_and_coupling():
    from scipy.stats import binom
    for mu in (6, 12, 30, 60):
        assert binom.sf(mu, 4000, (mu / 2) / 4000) <= np.exp(-mu / 6)

    p = np.array([0.05, 0.1, 0.08, 0.12, 0.07])
    q = np.array([0.3, 0.4, 0.2, 0.35, 0.25])
    mu = 2
    outs, pq = product_pmf(T.attacked_prob(p, q))

    sets = [np.array(S, bool) for S in itertools.product([0, 1], repeat=5)]
    pmf_t = np.zeros(len(outs))
    p_over = 0.0
    for S in sets:
        w = float(np.prod(np.where(S, q, 1 - q)))
        if S.sum() > mu:
            p_over += w
            _, pS = product_pmf(p)
        else:
            _, pS = product_pmf(np.where(S, 1.0, p))
        pmf_t += w * pS
    assert 0.5 * np.abs(pmf_t - pq).sum() <= p_over + 1e-12


def test_drift_attack_equals_drifted_null():
    rng = np.random.default_rng(13)
    p0 = rng.uniform(0.001, 0.05, 50)
    theta, delta = 1.0, 0.2
    q = delta * p0 / (1 - theta * p0)
    assert np.allclose(T.attacked_prob(theta * p0, q), (theta + delta) * p0)


def test_drift_test_meets_false_alarm_and_miss_levels():
    rng = np.random.default_rng(14)
    n, nc = 3000, 20000
    p0 = rng.uniform(0.002, 0.02, n)
    v = rng.pareto(2.5, n) + 1.0
    pc = rng.uniform(0.002, 0.02, nc)
    theta, Delta, alpha, beta = 1.0, 0.1, 0.05, 0.05
    F0, lam_c = float(p0 @ v), float(pc.sum())
    B = Delta * F0
    thA = theta + Delta
    sig2 = float((v * v * thA * p0 * (1 - thA * p0)).sum() + theta * F0**2 / lam_c)
    sig = np.sqrt(sig2)
    thr = B + sig / np.sqrt(alpha)
    pbar = (theta + Delta) * p0.max()
    Vneed = (2 * B + sig * (alpha**-0.5 + beta**-0.5)) / (1 - pbar)
    order = np.argsort(-v)
    k = int(np.searchsorted(np.cumsum(v[order]), Vneed)) + 1
    S = np.zeros(n, bool)
    S[order[:k]] = True
    fa = miss = 0
    for _ in range(2000):
        xc = rng.random(nc) < theta * pc
        th_hat = xc.sum() / lam_c
        x = rng.random(n) < thA * p0
        fa += (v @ (x - th_hat * p0)) > thr
        xs = x | S
        miss += (v @ (xs - th_hat * p0)) <= thr
    assert fa / 2000 <= alpha + 0.01
    assert miss / 2000 <= beta + 0.01


def test_horizon_loss_with_depletion():
    rng = np.random.default_rng(15)
    n, H, eps = 2000, 26, 0.1
    p = rng.uniform(0.0005, 0.003, n)
    v = rng.pareto(2.0, n) + 1.0
    E = float(p @ (v * v))
    q = 2 * (eps / np.sqrt(H)) * p * v / np.sqrt(E)
    bound = (1 - H * (p.max() + q.max())) * 2 * eps * np.sqrt(H * E)
    assert bound > 0
    losses = []
    for _ in range(400):
        alive = np.ones(n, bool)
        loss = 0.0
        for _t in range(H):
            stolen = alive & (rng.random(n) < q)
            loss += v[stolen].sum()
            moved = alive & ~stolen & (rng.random(n) < p)
            alive &= ~(stolen | moved)
        losses.append(loss)
    assert np.mean(losses) >= bound * 0.97

    p1, H1 = 0.01, 10000
    q1 = 2 * (eps / np.sqrt(H1)) * p1 * 1.0 / np.sqrt(p1)
    assert 1 - H1 * (p1 + q1) < 0


def test_monitors_realise_statistic_interface():

    from argus.monitor import Statistic
    p = np.full(4, 0.01)
    v = np.ones(4)
    assert isinstance(ValueWeightedMonitor(p, v), Statistic)
    assert isinstance(ControlCalibratedMonitor(p, v, np.full(10, 0.01)), Statistic)
