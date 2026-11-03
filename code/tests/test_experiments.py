import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
import common as C


def synthetic_panel(weeks=200, seed=0):
    rng = np.random.default_rng(seed)
    wk = pd.date_range("2015-01-05", periods=weeks, freq="W-MON")
    theta = np.exp(np.cumsum(rng.normal(0, 0.05, weeks)))
    theta /= theta.mean()
    strata = [("exposed", "p2pkh", "e2013_16", "v1", "a2_5", 0.004, 20000),
              ("exposed", "p2pk", "e2009_10", "v10", "a10_", 0.001, 5000),
              ("control", "p2pkh", "e2013_16", "v1", "a2_5", 0.003, 80000),
              ("control", "p2sh", "e2017_20", "v0_1", "a1_2", 0.006, 50000)]
    rows = []
    for pop, fam, era, vb, band, lam, n in strata:
        for t, w in enumerate(wk):
            e = rng.poisson(n * lam * theta[t])
            rows.append({"pop": pop, "fam": fam, "era": era, "vband": vb, "band": band, "wk": w,
                         "n": float(n), "sv": n * 1.0, "sv2": n * 1.0, "e_n": float(e),
                         "e_sv": float(e), "e_sv2": float(e)})
    return pd.DataFrame(rows), pd.Series(theta, index=wk), {s[:5]: s[5] for s in strata}


def test_null_model_recovers_hazards_and_regime():
    panel, theta, lam = synthetic_panel()
    m = C.NullModel(shrink=0.0).fit(panel)
    for key, true in lam.items():
        est = float(m.lam.loc[key])
        assert abs(est / true - 1) < 0.05, (key, est, true)
    corr = np.corrcoef(np.log(m.theta_train.values), np.log(theta.values))[0, 1]
    assert corr > 0.9


def test_control_theta_tracks_regime():
    panel, theta, _ = synthetic_panel()
    m = C.NullModel(shrink=0.0).fit(panel)
    th = C.control_theta(panel, m, prior_events=1.0)["theta"]
    corr = np.corrcoef(np.log(th.values[10:]), np.log(theta.values[10:]))[0, 1]
    assert corr > 0.8


def test_detector_calibration_meets_target_arl():
    rng = np.random.default_rng(3)
    z_train = rng.standard_t(df=4, size=600)
    det = C.calibrate(C.RobustDetector(), z_train, target_arl=52, rng=rng, n_rep=200)
    z_test = rng.standard_t(df=4, size=52 * 60)
    rate = det.run(z_test).sum() / len(z_test)
    assert 0.3 / 52 < rate < 2.5 / 52


def test_detector_catches_persistent_shift():
    rng = np.random.default_rng(4)
    det = C.calibrate(C.RobustDetector(), rng.standard_normal(600), target_arl=104, rng=rng, n_rep=200)
    hits = [det.run(rng.standard_normal(52) + 1.0).any() for _ in range(200)]
    assert np.mean(hits) > 0.9


def test_estimate_phi_rejects_duplicate_columns_and_recovers_inflation():

    rng = np.random.default_rng(7)
    w = pd.DataFrame({"S": rng.normal(0, 3, 400), "var": np.ones(400),
                      "Sn": rng.normal(0, 2, 400), "var_n": np.ones(400)})
    dup = w.rename(columns={"Sn": "S", "var_n": "var"})[["S", "var"]]
    try:
        C.estimate_phi(dup)
        raised = False
    except ValueError:
        raised = True
    assert raised
    phi = C.estimate_phi(w[["Sn", "var_n"]].rename(columns={"Sn": "S", "var_n": "var"}))
    assert 3.0 < phi < 5.0


def test_feasible_theft_respects_key_budget_and_stock():
    import rq56
    plan = np.array([50.0, 50.0, 50.0, 50.0])
    uv = np.array([1.0, 1.0, 2.0, 2.0])
    add_v, add_n = rq56.feasible_theft(plan, uv, mu=30, stock0=100.0)
    assert np.all(add_n <= 30 + 1e-9)
    assert abs(add_v.sum() - 100.0) < 1e-9
    assert np.allclose(add_v, [30.0, 30.0, 40.0, 0.0])


def _edetect_weeks(rng, T, r_ratio=1.0, theft=None):


    nA, nC = 3000, 30000
    pA = rng.uniform(0.0005, 0.003, nA)
    pC = rng.uniform(0.0005, 0.003, nC)
    v = rng.pareto(1.3, nA) + 0.01
    LA, LC, F = pA.sum(), pC.sum(), float(pA @ v)
    theta = np.exp(rng.normal(0, 1.2, T))
    rows = []
    for t in range(T):
        nA_t = rng.poisson(theta[t] * r_ratio * LA)
        nC_t = rng.poisson(theta[t] * LC)
        idx = rng.choice(nA, size=nA_t, p=pA / LA) if nA_t else np.array([], int)
        rows.append((float(v[idx].sum()), float(nA_t), float(nC_t)))
    w = pd.DataFrame(rows, columns=["obs_v", "obs_n", "ctl_n"])
    w["hF"], w["lam0_A"], w["lam0_C"] = F, LA, LC
    return w


def test_edetector_run_length_survives_wild_regimes_and_heavy_tails():
    rng = np.random.default_rng(31)
    w = _edetect_weeks(rng, 6000)
    d = C.EDetector(c=50.0)
    alarms = d.run(d.y(w)).sum()

    assert alarms <= 6000 / 50 * 1.3


def test_edetector_needs_its_drift_margin():
    rng = np.random.default_rng(32)
    w = _edetect_weeks(rng, 3000, r_ratio=2.0)
    loose = C.EDetector(c=50.0, drift=0.0).run(C.EDetector(drift=0.0).y(w)).sum()
    tight = C.EDetector(c=50.0, drift=1.0)
    covered = tight.run(tight.y(w)).sum()
    assert loose > 3000 / 50 * 2
    assert covered <= 3000 / 50 * 1.3


def test_edetector_catches_a_sustained_theft():
    rng = np.random.default_rng(33)
    w = _edetect_weeks(rng, 300)
    d = C.EDetector(c=104.0)
    add_n = np.zeros(300)
    add_v = np.zeros(300)
    add_n[100:] = 3.0
    add_v[100:] = 3.0 * 40.0
    a = d.run(d.y(w, add_v, add_n))
    assert a[:100].sum() <= 2 and a[100:160].any()


def _band_frame(rng, T=400):


    rows = []
    sd = {"v0_1": 30.0, "v100": 10000.0}
    for t in range(T):
        wk = pd.Timestamp("2014-01-06") + pd.Timedelta(weeks=t)
        for b, s in sd.items():
            S = rng.normal(0, s) if b == "v0_1" else s * rng.standard_t(3)
            rows.append((wk, b, S, s * s, s * s, 10.0 * s, 1e6))
    wb = pd.DataFrame(rows, columns=["wk", "vband", "S", "var", "energy", "exp_v", "stock_v"]).set_index("wk")
    return wb


def test_band_shares_sum_to_one_and_whale_law_hits_largest_band():
    rng = np.random.default_rng(41)
    wb = _band_frame(rng, 20)
    for law in ("tilted", "mimic", "uniform", "whale"):
        sh = C.band_shares(wb, law)
        assert np.allclose(sh.sum(axis=1), 1.0)
    assert (C.band_shares(wb, "whale")["v100"] == 1.0).all()


def test_banded_detector_catches_small_output_theft_hidden_in_whale_noise():
    rng = np.random.default_rng(42)
    wb = _band_frame(rng)
    train = wb[wb.index < pd.Timestamp("2018-01-01")]
    bd = C.calibrate_bands(train, 52.0, rng)
    test = wb[wb.index >= pd.Timestamp("2018-01-01")]
    weeks = test.index.unique()
    add = pd.DataFrame(0.0, index=weeks, columns=C.BANDS)
    add.loc[weeks[20:], "v0_1"] = 150.0
    banded = bd.run_z(bd.z_frame(test, add))
    agg_sigma = np.sqrt(30.0 ** 2 + 10000.0 ** 2)
    assert banded[20:40].any()
    assert 150.0 / agg_sigma < 0.02


def test_glr_ewma_calibration_and_power():


    rng = np.random.default_rng(7)

    z_train = rng.standard_normal(5000)
    for det in (C.GLRDetector(), C.EWMADetector()):
        det = C.calibrate(det, z_train, target_arl=104, rng=rng, n_rep=200)
        Z = rng.standard_normal((50, 416))
        many = det.run_many(Z)
        assert (many[3] == det.run(Z[3])).all()
        arl = Z.size / max(many.sum(), 1)
        assert 60 < arl < 200, arl
        shifted = det.run_many(Z[:, :52] + 0.5)
        assert shifted.any(axis=1).mean() > 0.8


def test_glr_restarts_after_alarm():
    det = C.GLRDetector(h=5.0, z_max=50.0, window=10)
    z = np.r_[3.0, 3.0, np.zeros(20)]
    a = det.run(z)

    assert a[1] and a.sum() == 1


def test_focus_matches_brute_force_glr():

    rng = np.random.default_rng(3)
    z = np.r_[rng.standard_normal(30), rng.standard_normal(30) + 0.8]
    det = C.FOCuSDetector(h=6.0, z_max=1e9, cap=1e9)
    a = det.run(z)

    ref, start = np.zeros(len(z), dtype=bool), 0
    for t in range(len(z)):
        s = np.cumsum(z[start:t + 1][::-1])
        g = np.max(np.maximum(s, 0) ** 2 / (2 * np.arange(1, len(s) + 1)))
        if g > 6.0:
            ref[t], start = True, t + 1
    assert (a == ref).all() and a[30:].any()


def test_newma_calibration_and_power():
    rng = np.random.default_rng(5)
    z_train = rng.standard_normal(5000)
    det = C.calibrate(C.NEWMADetector(ref=z_train), z_train, target_arl=104, rng=rng, n_rep=200)
    Z = rng.standard_normal((40, 416))
    arl = Z.size / max(det.run_many(Z).sum(), 1)
    assert 50 < arl < 250, arl
    assert det.run_many(Z[:, :52] + 1.0).any(axis=1).mean() > 0.8


def test_conformal_martingale_validity_and_power():
    rng = np.random.default_rng(9)
    det = C.ConformalDetector(c=104.0)
    p = det.pvalues(rng.standard_normal(300), rng.standard_normal(2000))

    assert abs(p.mean() - 0.5) < 0.03

    assert det.run(p).sum() <= 2000 / 104 * 1.5 + 2
    shifted = det.pvalues(rng.standard_normal(300), rng.standard_normal(52) + 1.5)
    assert det.run(shifted).any()


def test_ecod_flags_outlier_at_target_rate():
    rng = np.random.default_rng(13)
    train = rng.standard_normal((400, 8))
    det = C.ECODDetector().fit(train, target_arl=104)
    assert det.run(train).mean() < 0.02

    x = np.full((1, 8), 3.0)
    assert det.run(x)[0]
    assert not det.run(np.zeros((1, 8)))[0]
