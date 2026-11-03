import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
import replay as R

T0 = pd.Timestamp("2021-01-04")


def test_thief_budget_and_key_cap():
    H = R.HORIZON
    V = np.zeros((4, H))
    V[0] = 10.0
    V[1] = 2.0
    V[2] = 3.0
    V[3] = 1.0
    order = np.array([0, 1, 2, 3])
    plan = np.full(H, 3.0)
    wk = R.thief(order, V, plan, mu=10)


    assert list(wk) == [5, 0, 1, 1]
    taken = np.bincount(wk, weights=V[np.arange(4), wk], minlength=H)
    assert (np.cumsum(taken) <= np.cumsum(plan) + 1e-9).all()

    V2 = np.ones((50, H))
    wk2 = R.thief(np.arange(50), V2, np.full(H, 100.0), mu=7)
    assert np.bincount(wk2[wk2 >= 0]).max() <= 7


def test_trajectories_follow_changes_and_events():
    pool = pd.DataFrame({"sid": 0, "law": "uniform", "id": [11, 22], "key": [0.1, 0.2],
                         "v0": [100_000_000, 5_000_000_000], "fam": "p2pkh", "era": "e2013_16"})
    chg = pd.DataFrame({"sid": 0, "id": [11, 11, 22],
                        "ts": [T0 + pd.Timedelta(days=3), T0 + pd.Timedelta(days=5), T0 + pd.Timedelta(days=20)],
                        "v_new": [300_000_000, 200_000_000, 0]})
    ev = pd.DataFrame({"sid": 0, "id": [22, 22], "wk": [T0 + pd.Timedelta(weeks=2), T0 + pd.Timedelta(weeks=5)]})
    ids, row, V, E = R.trajectories(pool, chg, ev, T0)
    a, b = row.loc[11], row.loc[22]

    assert V[a, 0] == 1.0 and V[a, 1] == 2.0 and V[a, -1] == 2.0

    assert V[b, 2] == 50.0 and V[b, 3] == 0.0

    assert E[b, 2] and not E[b, 5]


def test_deltas_move_whole_value_and_remove_unit():
    H = R.HORIZON
    ids = pd.DataFrame({"id": [1], "v0": [0], "fam": ["p2pkh"], "era": ["e2013_16"]})
    V = np.full((1, H), 5.0)
    E = np.zeros((1, H), dtype=bool)
    E[0, 3] = True
    wk = pd.date_range(T0, periods=H, freq="W-MON")
    d = R.deltas(ids, np.array([0]), np.array([1]), V, E, wk)
    ev = d.groupby("wk")[["de_n", "de_sv", "de_sv2"]].sum()
    assert ev.loc[wk[1], "de_n"] == 1.0 and ev.loc[wk[1], "de_sv"] == 5.0
    assert ev.loc[wk[3], "de_n"] == -1.0 and ev.loc[wk[3], "de_sv2"] == -25.0
    risk = d.groupby("wk")["dn"].sum()
    assert wk[1] not in risk[risk != 0].index
    assert (risk.loc[wk[2:]] == -1.0).all()
    assert set(d["vband"]) == {"v1"}


def test_vband_matches_builder_edges():
    v = np.array([0.005, 0.01, 0.5, 1.0, 99.9, 100.0])
    assert list(R.vband(v)) == ["v0_001", "v0_01", "v0_1", "v1", "v10", "v100"]


def test_attack_start_preserves_the_deployed_refit_calendar():


    assert R.refit_anchor(pd.Timestamp("2020-02-03")) == pd.Timestamp("2020-01-06")
    assert R.refit_anchor(pd.Timestamp("2020-03-30")) == pd.Timestamp("2020-01-06")
    assert R.refit_anchor(pd.Timestamp("2020-04-06")) == pd.Timestamp("2020-04-06")
    assert R.refit_anchor(pd.Timestamp("2020-04-13")) == pd.Timestamp("2020-04-06")


def test_pool_quota_rejects_a_hidden_better_target():
    import pytest


    short = pd.DataFrame({"__filename": ["A", "A", "B", "B"],
                          "law": "tilted", "draw": 0, "key": [0.1, 0.2, 0.3, 0.4]})
    with pytest.raises(ValueError, match="insufficient"):
        R.check_pool_quota(short, k=3, kb=2)
    complete = pd.DataFrame({"__filename": ["A"] * 3 + ["B"] * 3,
                             "law": "tilted", "draw": 0, "key": [0.1, 0.3, 0.5, 0.2, 0.4, 0.6]})
    R.check_pool_quota(complete, k=3, kb=3)


def test_single_movement_rule_cancels_natural_spends_after_a_theft():
    row = pd.Series([0, 1], index=[11, 22])
    events = pd.DataFrame({"id": [11, 11, 22, 33],
                           "wk": [T0, T0 + pd.Timedelta(weeks=2), T0 + pd.Timedelta(weeks=2),
                                  T0 + pd.Timedelta(weeks=3)],
                           "value": [3.0, 200.0, 50.0, 20.0]})
    natural = R.natural_max_after_theft(events, row, np.array([0]), np.array([1]), T0)
    assert natural[0] == 3.0
    assert natural[2] == 50.0
    assert natural[3] == 20.0
