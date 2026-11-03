import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'experiments'))
from control_variance_quick import control_innovation, fit_multiplier


def test_control_innovation_covariance_monte_carlo():
    rng = np.random.default_rng(20261102)
    lam, value, theta = np.array([5., 2., 1.]), np.array([1., 5., 20.]), 1.2
    counts = rng.poisson(theta*lam, size=(200000, 3))
    F, L = float(lam@value), float(lam.sum())
    theta_hat = (counts.sum(axis=1)+20.*1.1)/(L+20.)
    residual = counts@value-theta_hat*F
    a = F/(L+20.)
    analytic = theta*np.sum(lam*(value-a)**2)
    assert abs(residual.var(ddof=1)/analytic-1) < .02


def test_feature_is_causal_and_ignores_exposed_events():
    class FixedModel:
        def hazard(self, panel):
            return np.full(len(panel), .01)
    weeks = pd.date_range('2018-01-01', periods=40, freq='W-MON')
    panel = pd.DataFrame({'wk': weeks, 'pop': 'control', 'n': 1000., 'sv': 5000.,
                          'sv2': 40000., 'e_n': 10.+np.sin(np.arange(40)),
                          'e_sv': 50.+20.*np.cos(np.arange(40))})
    exposed = panel.assign(pop='exposed', e_n=1e8, e_sv=1e12)
    whole = control_innovation(pd.concat([panel, exposed]), FixedModel())
    prefix = control_innovation(panel.iloc[:30], FixedModel())
    pd.testing.assert_frame_equal(whole.iloc[:30], prefix)
    changed = panel.copy()
    changed.loc[30:, ['e_n', 'e_sv']] *= 1e6
    pd.testing.assert_frame_equal(control_innovation(changed, FixedModel()).iloc[:30], prefix)

    controls = pd.DataFrame({'ratio': [1., 2., 3., 1e8]},
                            index=pd.to_datetime(['2018-01-01', '2018-01-08', '2019-01-07', '2020-01-06']))
    multiplier, norm = fit_multiplier(controls)
    assert norm == 2.
    assert multiplier.iloc[-1] == 4.
