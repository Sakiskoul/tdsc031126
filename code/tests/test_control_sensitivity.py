import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
from control_sensitivity import remove_recognised_exposure


def test_known_reveal_truncates_weekly_control_and_cancels_its_later_spend(tmp_path):
    weeks = pd.date_range("2020-01-06", "2020-02-24", freq="W-MON")
    records = []
    for pop, fam in (("control", "p2pkh"), ("control", "p2sh"), ("exposed", "p2pkh")):
        for w in weeks:
            risk = float(pd.Timestamp("2020-01-13") <= w <= pd.Timestamp("2020-02-10"))
            event = float(w == pd.Timestamp("2020-02-10"))
            records.append({"pop": pop, "fam": fam, "era": "e2017_20", "vband": "v1", "band": "a1_2",
                            "wk": w, "n": risk, "sv": 5 * risk, "sv2": 25 * risk,
                            "e_n": event, "e_sv": 5 * event, "e_sv2": 25 * event})
    panel = pd.DataFrame(records)
    original = panel.copy(deep=True)
    pd.DataFrame({"a": [pd.Timestamp("2020-01-06")], "b": [pd.Timestamp("2020-02-10")],
                  "t_alt": [pd.Timestamp("2020-01-22")], "sat": [500_000_000], "typ": [2],
                  "event": [True], "hit": [True]}).to_parquet(tmp_path / "0.parquet", index=False)
    screened, info = remove_recognised_exposure(panel, tmp_path)
    control = screened[screened["pop"] == "control"].set_index("wk")
    assert set(control["fam"]) == {"p2pkh"}
    assert list(control.index[control["n"] > 0]) == [pd.Timestamp("2020-01-13"), pd.Timestamp("2020-01-20")]
    assert control["e_n"].sum() == 0
    assert info["known_reveal_events_removed"] == 1
    assert info["exposed_panel_unchanged"]
    pd.testing.assert_frame_equal(panel, original)
    all_families, _ = remove_recognised_exposure(panel, tmp_path, simple_only=False)
    retained = all_families[all_families["pop"] == "control"]
    assert set(retained["fam"]) == {"p2pkh", "p2sh"}
    assert retained[retained["fam"] == "p2sh"]["e_n"].sum() == 1
    assert retained[retained["fam"] == "p2pkh"]["e_n"].sum() == 0
