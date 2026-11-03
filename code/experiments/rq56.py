from __future__ import annotations

import numpy as np
import pandas as pd

import common as C
import rq

LAWS = ["tilted", "mimic", "uniform", "whale"]
INTENSITIES = [0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0]


def whale_value(s: "rq.Setup") -> float:

    p = s.panel
    e = p[(p["pop"] == "exposed") & (p["vband"] == "v100") & (p["n"] > 0)]
    e = e[(e["wk"] >= s.evalw[0]) & (e["wk"] < s.evalw[1])]
    return float((e["sv"].sum() / e["n"].sum()) if len(e) else 0.0)


def unit_value(s: "rq.Setup", w: pd.DataFrame, law: str) -> pd.Series:

    if law == "tilted":
        return (w["energy"] / w["exp_v"].clip(lower=1e-12))
    if law == "mimic":
        return (w["exp_v"] / w["exp_n"].clip(lower=1e-12))
    if law == "uniform":
        return (w["stock_v"] / w["stock_n"].clip(lower=1))
    return pd.Series(whale_value(s), index=w.index)


class Bench:
    def __init__(self, s: "rq.Setup"):
        self.s = s
        self.dets = rq.detectors(s)
        self.frames = {}
        for name, (col, var_col, calibrated, det) in self.dets.items():
            w = s.w_cal if calibrated else s.w_unc
            if col == "Sn":
                w = rq._count_stats(w)
            self.frames[name] = w
        self.sigma = np.sqrt(s.phi * s.w_cal["var"])


        self.z_hist = (s.w_cal["S"] / self.sigma.clip(lower=1e-12)).loc[s.train[0]:]
        self.ecod_shares = {law: C.band_shares(s.w_band, law) for law in LAWS}
        self.thr = None
        if "banded" in self.dets:
            wb = s.w_band
            self.bandS = wb.pivot_table(index=wb.index, columns="vband", values="S", aggfunc="sum").reindex(columns=C.BANDS).fillna(0.0)
            self.bandV = wb.pivot_table(index=wb.index, columns="vband", values="var", aggfunc="sum").reindex(columns=C.BANDS).fillna(0.0)
            self.shares = {law: C.band_shares(wb, law).reindex(columns=C.BANDS).fillna(0.0) for law in LAWS}

    def set_threshold_rule(self, big: pd.DataFrame) -> None:
        b = big[big["pop"] == "exposed"].copy()
        b["wk"] = pd.to_datetime(b["wk"])
        self.thr = rq.threshold_rule_level(self.s, big)
        self.natural_max = (b.groupby("wk")["sat"].max() / 1e8).reindex(self.s.w_cal.index).fillna(0.0)

    def alarms(self, det_name: str, win: pd.Index, add_v: np.ndarray, add_n: np.ndarray,
               unit_v: np.ndarray, law: str | None = None) -> np.ndarray:
        if det_name == "threshold":
            return (np.maximum(self.natural_max.loc[win].values, np.where(add_n > 0, unit_v, 0.0))
                    >= self.thr)
        col, var_col, _, det = self.dets[det_name]
        w = self.frames[det_name].loc[win]
        if col == "E":
            return det.run(det.y(w, add_v, add_n))
        if col == "CTM":
            hist = self.z_hist[self.z_hist.index < win[0]].values
            z = (w["S"].values + add_v) / np.sqrt(self.s.phi * w["var"].clip(lower=1e-12).values)
            return det.run(det.pvalues(hist, z))
        if col == "ECOD":
            sh = self.ecod_shares[law] if law is not None else None
            return det.run(C.ecod_features(w, self.s.w_band, self.s.phi, add_v, add_n, sh))
        if col == "B":
            S = self.bandS.reindex(win).fillna(0.0)
            if law is not None:
                S = S + self.shares[law].reindex(win).fillna(0.0).values * np.asarray(add_v)[:, None]
            V = self.bandV.reindex(win).fillna(0.0)
            out = np.zeros(len(win), dtype=bool)
            for b, d in det.dets.items():
                z = (S[b] / np.sqrt(d.phi * V[b].clip(lower=1e-12))).values
                out |= d.run(z)
            return out
        if col == "Sn":
            stat = w["Sn"].values + add_n
        else:
            stat = w["S"].values + add_v
        z = stat / np.sqrt(det.phi * w[var_col].clip(lower=1e-12).values)
        return det.run(z)


def whale_stock(s: "rq.Setup") -> pd.Series:

    p = s.panel
    e = p[(p["pop"] == "exposed") & (p["vband"] == "v100")]
    return (e.groupby("wk")["sv"].sum() / 1.0).reindex(s.w_cal.index).fillna(0.0)


def feasible_theft(add_v: np.ndarray, uv: np.ndarray, mu: float, stock0: float) -> tuple[np.ndarray, np.ndarray]:


    uv = np.clip(uv, 1e-12, None)
    add_v = np.minimum(add_v, mu * uv)
    cum = np.minimum(np.cumsum(add_v), max(stock0, 0.0))
    add_v = np.diff(cum, prepend=0.0)
    return add_v, add_v / uv


def block_indices(n: int, block: int, n_boot: int, rng) -> np.ndarray:

    k = int(np.ceil(n / block))
    starts = rng.integers(0, n, size=(n_boot, k))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]) % n
    return idx.reshape(n_boot, -1)[:, :n]


def rq5_sweep(s: "rq.Setup", big: pd.DataFrame, horizon: int = 52, stride: int = 4,
              mu: float = 1000.0) -> dict:
    b = Bench(s)
    wstock = whale_stock(s)
    b.set_threshold_rule(big)
    det_names = list(b.dets) + ["threshold"]
    weeks = s.ev(s.w_cal).index
    starts = weeks[: max(len(weeks) - horizon, 1): stride]
    out = {"horizon": horizon, "starts": int(len(starts)), "threshold_btc": b.thr,
           "intensities": INTENSITIES, "mu_keys_per_week": mu, "laws": {}}
    zero_w = lambda win: np.zeros(len(win))
    base_rate, base_hits = {}, {}
    for d in det_names:
        hits = []
        for t0 in starts:
            win = weeks[weeks >= t0][:horizon]
            hits.append(b.alarms(d, win, zero_w(win), zero_w(win), zero_w(win)).any())
        base_rate[d] = float(np.mean(hits))
        base_hits[d] = np.asarray(hits, float)


    block = max(1, horizon // stride)
    idx_boot = block_indices(len(starts), block, n_boot=2000, rng=np.random.default_rng(20261017))
    out["natural_window_alarm_rate"] = base_rate
    for law in LAWS:
        uv_all = unit_value(s, s.w_cal, law)
        res = {d: {"detection_rate": [], "median_stolen_btc": [], "median_delay": []} for d in det_names}
        capped_by_m, taken_by_m = [], []
        for m in INTENSITIES:
            per = {d: ([], [], []) for d in det_names}
            capped, taken = [], []
            for t0 in starts:
                win = weeks[weeks >= t0][:horizon]
                plan = (m * b.sigma.loc[win]).values
                uv = uv_all.loc[win].values
                stock0 = float(wstock.loc[t0] if law == "whale" else s.w_cal.loc[t0, "stock_v"])
                add_v, add_n = feasible_theft(plan, uv, mu, stock0)
                capped.append(float(1.0 - add_v.sum() / max(plan.sum(), 1e-12)))
                taken.append(float(add_v.sum()))
                for d in det_names:
                    a = b.alarms(d, win, add_v, add_n, uv, law=law)
                    if a.any():
                        k = int(np.argmax(a))
                        per[d][0].append(1)
                        per[d][1].append(float(add_v[: k + 1].sum()))
                        per[d][2].append(k + 1)
                    else:
                        per[d][0].append(0)
                        per[d][1].append(float(add_v.sum()))
            capped_by_m.append(float(np.mean(capped)))
            taken_by_m.append(float(np.median(taken)))
            for d in det_names:
                det, stolen, delay = per[d]
                det_a = np.asarray(det, float)


                quiet = base_hits[d] == 0
                res[d].setdefault("median_stolen_quiet_btc", []).append(
                    float(np.median(np.asarray(stolen)[quiet])) if quiet.any() else None)
                res[d]["detection_rate"].append(float(np.mean(det)))
                res[d].setdefault("excess_detection", []).append(float(np.mean(det)) - base_rate[d])
                ex_boot = det_a[idx_boot].mean(axis=1) - base_hits[d][idx_boot].mean(axis=1)
                res[d].setdefault("excess_ci95", []).append([float(np.quantile(ex_boot, 0.025)),
                                                           float(np.quantile(ex_boot, 0.975))])
                res[d]["median_stolen_btc"].append(float(np.median(stolen)))
                res[d].setdefault("p90_stolen_btc", []).append(float(np.quantile(stolen, 0.9)))
                res[d]["median_delay"].append(float(np.median(delay)) if delay else None)
                res[d].setdefault("p90_delay", []).append(float(np.quantile(delay, 0.9)) if delay else None)
        out["laws"][law] = res
        out.setdefault("capped_share", {})[law] = capped_by_m
        out.setdefault("median_taken_btc", {})[law] = taken_by_m

    E = s.phi * s.ev(s.w_cal)["energy"]
    out["theory_covert_btc_eps0.1"] = float(2 * rq.EPS * np.sqrt(E.rolling(horizon).sum().dropna().median()))
    out["sigma_week_median_btc"] = float(b.sigma.loc[weeks].median())

    zero = np.zeros(len(weeks))
    out["natural_alarms"] = {d: int(b.alarms(d, weeks, zero, zero, zero).sum()) for d in det_names}
    return out


def rq6_response(s: "rq.Setup", rates=(10, 50, 100, 500, 1000, 5000, 20000),
                 latencies=(1, 2, 4), mu=(10, 100, 1000)) -> dict:

    w = s.ev(s.w_cal)
    natural = w["obs_v"]
    vw = whale_value(s)
    rows = []
    for rho in rates:
        throttled = float((natural > rho).mean())
        delayed = float((natural - rho).clip(lower=0).sum() / max(natural.sum(), 1e-9))
        for R in latencies:
            for m in mu:
                rows.append({"rho_btc_week": rho, "R_weeks": R, "mu_keys_week": m,
                             "post_alarm_loss_btc": min(R * rho, m * R * vw),
                             "unlimited_loss_btc": m * R * vw,
                             "friction_weeks_throttled": throttled,
                             "friction_value_delayed": delayed})
    return {"rows": rows, "whale_value_btc": vw,
            "natural_outflow_median_btc": float(natural.median()),
            "natural_outflow_p95_btc": float(natural.quantile(0.95))}
