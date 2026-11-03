import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
import common as C


def test_focus_matches_changepoint_online():
    cpo = pytest.importorskip("changepoint_online")
    rng = np.random.default_rng(21)
    z = np.r_[rng.standard_normal(80), rng.standard_normal(60) + 0.7, rng.standard_normal(80)]
    det = C.FOCuSDetector(h=5.0, z_max=3.5, cap=4.0)
    ours = det.run(z)


    ref = np.zeros(len(z), dtype=bool)
    f = cpo.Focus(cpo.Gaussian(loc=0.0), side="right")
    for t, x in enumerate(z):
        f.update(float(min(x, det.cap)))
        if f.statistic() > det.h or x > det.z_max:
            ref[t] = True
            f = cpo.Focus(cpo.Gaussian(loc=0.0), side="right")
    assert ours.sum() >= 3
    assert (ours == ref).all()


@pytest.mark.skipif(not os.environ.get("NEWMA_REPO"), reason="set NEWMA_REPO to a lightonai/newma checkout")
def test_newma_matches_authors_code():
    sys.path.insert(0, os.environ["NEWMA_REPO"])
    from onlinecp import algos
    from onlinecp.utils import feature_functions as feat
    big, small = algos.select_optimal_parameters(13)
    assert np.isclose(big, C.NEWMADetector.big_lam) and np.isclose(small, C.NEWMADetector.lam)
    rng = np.random.default_rng(34)
    ref = rng.standard_normal(300)
    z = np.r_[rng.standard_normal(100), rng.standard_normal(60) + 1.0]
    det = C.NEWMADetector(ref=ref)
    W = det.frequencies()

    r = np.minimum(ref, det.cap)
    _, sigmasq = feat.generate_frequencies(det.m, 1, data=r[:, None])
    assert np.isclose(np.median((r[:, None] - r[None, :]) ** 2), sigmasq)
    x = np.minimum(z, det.cap)
    code = algos.NEWMA(x[0:1], forget_factor=big, forget_factor2=small,
                       feat_func=lambda v: feat.fourier_feat(np.atleast_1d(v), W[:, None]),
                       thresholding_method="fixed", fixed_threshold=1e9)
    stat = []
    for v in x:
        code.update(np.atleast_1d(v))
        stat.append(code.statistic)

    assert np.allclose(det.statistics(z, init=x[0]), stat, rtol=1e-10, atol=1e-12)


def test_ecod_uses_pyod_one_week_at_a_time():

    pytest.importorskip("pyod")
    from pyod.models.ecod import ECOD
    rng = np.random.default_rng(8)
    train = rng.standard_normal((300, 8))
    test = rng.standard_normal((50, 8))
    det = C.ECODDetector().fit(train, target_arl=104)
    ref = ECOD(contamination=1 / 104).fit(train)
    per_week = np.array([ref.decision_function(x[None, :])[0] for x in test])
    assert np.allclose(det.score(test), per_week)
    assert (det.run(test) == (per_week > ref.threshold_)).all()


def test_ecod_is_causal():

    pytest.importorskip("pyod")
    det = C.ECODDetector().fit(np.linspace(-1.0, 1.0, 300)[:, None], target_arl=104)
    first = np.array([[8.0]])
    alone = det.score(first)[0]
    with_future = det.score(np.r_[first, np.full((100, 1), 10.0)])[0]
    assert np.isclose(alone, with_future)
    assert det.run(first)[0]
