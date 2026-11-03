import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
import common as C


def first(mask: np.ndarray) -> int:
    return int(np.argmax(mask)) if mask.any() else -1


def series(seed: int, T: int = 52) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    base = rng.standard_normal(T)
    shift = base + np.concatenate([np.zeros(T // 2), np.full(T - T // 2, 1.5)])
    spike = base.copy()
    spike[30] = 20.0
    return [base, shift, spike]


def check(det, inputs, inclusive=False, kappas=(0.5, 1.0, 2.0)):
    for x in inputs:
        s = det.scores(x)
        assert s.shape == (len(x),)
        assert first(det.run(x)) == first(s >= 1.0 if inclusive else s > 1.0)

    for kappa in kappas:
        for x in inputs:
            s = det.scores(x)
            assert np.all(np.diff(np.maximum.accumulate(s)) >= 0)
            assert first((s >= kappa) if inclusive else (s > kappa)) >= -1


def test_robust_cusum_scores_match_run():
    for seed in range(20):
        check(C.RobustDetector(h=4.0, z_max=6.0), series(seed))


def test_scaled_thresholds_match_kappa_cut():
    det = C.RobustDetector(h=4.0, z_max=6.0)
    for seed in range(20):
        for x in series(seed):
            s = det.scores(x)
            for kappa in (0.5, 2.0):
                scaled = C.RobustDetector(h=4.0 * kappa, z_max=6.0 * kappa)
                assert first(scaled.run(x)) == first(s > kappa)


def test_focus_and_glr_scores_match_run():
    for seed in range(10):
        check(C.FOCuSDetector(h=6.0, z_max=6.0), series(seed))
        check(C.GLRDetector(h=6.0, z_max=6.0, window=13), series(seed))


def test_newma_scores_match_run():
    ref = np.random.default_rng(7).standard_normal(300)
    for seed in range(10):
        check(C.NEWMADetector(h=0.4, z_max=6.0, ref=ref), series(seed))


def test_martingale_scores_match_run():
    rng = np.random.default_rng(3)
    for seed in range(10):
        r = np.random.default_rng(seed)
        p = np.clip(r.uniform(size=52) ** np.where(np.arange(52) > 20, 4.0, 1.0), 1e-6, 1)
        check(C.ConformalDetector(c=20.0), [p, r.uniform(size=52)], inclusive=True)
        y = np.maximum(r.standard_normal(52) * 0.5 + np.where(np.arange(52) > 20, 0.8, 0.0), -1.0)
        check(C.EDetector(c=20.0), [y, np.maximum(rng.standard_normal(52) * 0.3, -1.0)], inclusive=True)


def test_ecod_scores_match_run():
    rng = np.random.default_rng(11)
    det = C.ECODDetector().fit(rng.standard_normal((300, 4)), target_arl=52.0)
    X = rng.standard_normal((52, 4))
    X[40] += 6.0
    s = det.scores(X)
    assert first(det.run(X)) == first(s > 1.0)
