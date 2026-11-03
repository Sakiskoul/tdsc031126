import numpy as np


def test_shared_global_shrinkage_covariance():
    rng = np.random.default_rng(20261102)
    lam = np.array([5., 20., 3.])
    mu = lam * np.array([.8, 1.2, 1.6])
    n = np.array([25., 50., 100.])
    F = np.array([10., 100., 500.])
    prior_events, prior_mean = 20., 1.1
    counts = rng.poisson(mu, size=(200000, 3))
    global_theta = (counts.sum(axis=1) + prior_events * prior_mean) / (lam.sum() + prior_events)
    group_theta = (counts + n * global_theta[:, None]) / (lam + n)
    predicted_value = group_theta @ F
    b = np.sum(F*n/(lam+n))/(lam.sum()+prior_events)
    coefficients = F/(lam+n)+b
    analytic_variance = np.sum(coefficients**2*mu)
    empirical_variance = predicted_value.var(ddof=1)
    assert abs(empirical_variance/analytic_variance-1) < .02

    independent_group_variance = np.sum(F**2 * group_theta.var(axis=0, ddof=1))
    assert independent_group_variance < .8*empirical_variance
