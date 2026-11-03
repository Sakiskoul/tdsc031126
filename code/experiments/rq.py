from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm

import common as C

EPS = 0.1
HORIZONS = [1, 13, 52, 104]


@dataclass
class Setup:


    panel: pd.DataFrame
    train: tuple[str, str]
    evalw: tuple[str, str]
    target_arl: float = 104.0
    seed: int = 20261016
    refit: int = 13
    lookback: int = 104
    gap: int = 13

    def __post_init__(self):
        p = self.panel
        tr = p[(p["wk"] >= self.train[0]) & (p["wk"] < self.train[1])]
        self.model = C.NullModel().fit(tr)
        self.w_cal = self._stats(calibrated=True)
        self.w_unc = self._stats(calibrated=False)
        self.phi = C.estimate_phi(self.w_cal.loc[self.train[0]:self.train[1]])
        self.phi_unc = C.estimate_phi(self.w_unc.loc[self.train[0]:self.train[1]])
        self.w_band = self._band_stats()
        self.rng = np.random.default_rng(self.seed)

    def _band_stats(self) -> pd.DataFrame:

        p = self.panel
        static = C.weekly_band_stats(p, self.model)
        if not self.refit:
            return static
        ev_start = pd.Timestamp(self.evalw[0])
        weeks = static.index.unique()
        weeks = weeks[weeks >= ev_start]
        parts = [static[static.index < ev_start]]
        for b in range(0, len(weeks), self.refit):
            block = weeks[b:b + self.refit]
            hi = block[0] - pd.Timedelta(weeks=self.gap)
            lo = hi - pd.Timedelta(weeks=self.lookback)
            fit = p[(p["wk"] >= lo) & (p["wk"] < hi)]
            m = C.NullModel().fit(fit) if len(fit) else self.model
            wb = C.weekly_band_stats(p, m)
            parts.append(wb[wb.index.isin(block)])
        return pd.concat(parts).sort_index()

    def _stats(self, calibrated: bool) -> pd.DataFrame:
        p = self.panel
        static = C.weekly_stats(p, self.model, calibrated=calibrated)
        if not self.refit:
            return static
        ev_start = pd.Timestamp(self.evalw[0])
        weeks = static.index[static.index >= ev_start]
        parts = [static[static.index < ev_start]]
        for b in range(0, len(weeks), self.refit):
            block = weeks[b:b + self.refit]
            hi = block[0] - pd.Timedelta(weeks=self.gap)
            lo = hi - pd.Timedelta(weeks=self.lookback)
            fit = p[(p["wk"] >= lo) & (p["wk"] < hi)]
            m = C.NullModel().fit(fit) if len(fit) else self.model
            parts.append(C.weekly_stats(p, m, calibrated=calibrated).loc[block])
        return pd.concat(parts).sort_index()

    def tr(self, w):
        return w.loc[self.train[0]:self.train[1]]

    def ev(self, w):
        return w.loc[self.evalw[0]:self.evalw[1]]


def rq1_budget(s: Setup) -> dict:
    w = s.ev(s.w_cal).copy()
    w["E_eff"] = s.phi * w["energy"]
    out = {"phi": s.phi, "weeks": int(len(w)), "eps": EPS,
           "stock_btc_median": float(w["stock_v"].median()),
           "flow_btc_per_week_median": float(w["exp_v"].median()),
           "energy_sqrt_btc_median": float(np.sqrt(w["E_eff"]).median())}
    for H in HORIZONS:
        roll = w["E_eff"].rolling(H).sum().dropna()
        bud = 2 * EPS * np.sqrt(roll)
        out[f"covert_btc_H{H}_median"] = float(bud.median())
        out[f"covert_frac_stock_H{H}_median"] = float((bud / w["stock_v"].loc[bud.index]).median())
        out[f"flow_btc_H{H}_median"] = float(w["exp_v"].rolling(H).sum().dropna().median())
    th = C.control_theta(s.panel, s.model)["theta"].loc[s.evalw[0]:s.evalw[1]]
    eta = float(np.quantile(np.abs(np.log(th.clip(1e-3))), 0.5))
    out["regime_eta_median_abs_log"] = eta
    out["linear_loophole_btc_per_week_median"] = float((eta * w["exp_v"]).median())
    out["diffuse"] = diffuse_budget(s)
    out["band_noise"] = band_noise(s)
    return out


BAND_CAP = {"v0_001": 0.01, "v0_01": 0.1, "v0_1": 1.0, "v1": 10.0, "v10": 100.0}


def band_noise(s: Setup, mu: float = 1000.0) -> dict:


    wb = s.w_band
    ev = s.w_cal.loc[s.evalw[0]:s.evalw[1]]
    uv = float((ev["exp_v"] / ev["exp_n"].clip(lower=1e-12)).median())
    weekly = mu * uv
    share = C.band_shares(wb, "mimic").loc[s.evalw[0]:s.evalw[1]].median()
    out = {"mimic_weekly_btc": weekly, "bands": {}}
    for b in C.BANDS:
        x = wb[wb["vband"] == b]
        row = {}
        for name, (lo, hi) in {"train": s.train, "eval": s.evalw}.items():
            S = x["S"].loc[lo:hi]
            if len(S) < 10:
                continue
            robust = float(1.4826 * (S - S.median()).abs().median())
            binom = float(np.sqrt(x["var"].loc[lo:hi]).median())
            row[name] = {"binomial_sd_btc": binom, "robust_sd_btc": robust,
                         "ratio": robust / binom if binom > 0 else None}
        if "eval" in row:
            m = weekly * float(share.get(b, 0.0))
            row["mimic_btc_week"] = m
            row["mimic_over_robust_sd"] = m / row["eval"]["robust_sd_btc"] if row["eval"]["robust_sd_btc"] > 0 else None
        out["bands"][b] = row
    return out


def diffuse_budget(s: Setup, eta: float = 0.5, beta: float = 0.1) -> dict:


    e = s.panel[(s.panel["pop"] == "exposed") & (s.panel["n"] > 0)].copy()
    e["h"] = s.model.hazard(e)
    th = C.control_theta(s.panel, s.model)
    e = e.join(th[["theta"]], on="wk")
    e["p"] = (e["theta"].fillna(1.0) * e["h"]).clip(upper=0.25)
    e["energy"] = e["p"] * e["sv2"]
    eb = e.groupby(["wk", "vband"])["energy"].sum().unstack(fill_value=0.0)
    eb = eb.loc[s.evalw[0]:s.evalw[1]]


    noise = band_noise(s)["bands"]
    phib = {b: max(1.0, (noise.get(b, {}).get("train", {}).get("ratio") or 1.0) ** 2) for b in eb.columns}
    kappa_max = eta ** 2 * EPS / np.log(1 / beta)
    out = {"eta": eta, "beta": beta, "kappa_threshold": float(kappa_max), "caps": {}}
    for cap_name, cap in BAND_CAP.items():
        cols = [b for b, c in BAND_CAP.items() if c <= cap and b in eb.columns]
        ec = eb[cols].sum(axis=1)
        eb_batch = sum(eb[c] * phib[c] for c in cols) if cols else ec
        row = {}
        for H in HORIZONS:
            roll = ec.rolling(H).sum().dropna()
            if roll.empty:
                continue
            kappa = cap / np.sqrt(roll.clip(lower=1e-12))
            rollb = eb_batch.rolling(H).sum().dropna()
            row[f"H{H}"] = {"loss_btc_median": float((1 - eta) * 2 * EPS * np.sqrt((s.phi * roll).median())),
                            "loss_batch_btc_median": float((1 - eta) * 2 * EPS * np.sqrt(rollb.median())),
                            "kappa_bound_median": float(kappa.median()),
                            "diffuse_share_of_weeks": float((kappa <= kappa_max).mean())}
        out["caps"][f"<= {cap} BTC"] = row
    return out


def rq2_control(s: Setup) -> dict:
    th = C.control_theta(s.panel, s.model)
    te = C.exposed_theta(s.panel, s.model).rename("theta_exp")
    j = th.join(te, how="inner").dropna()
    j = j[(j["theta_raw"] > 0) & (j["theta_exp"] > 0)]
    lc, le = np.log(j["theta_raw"]), np.log(j["theta_exp"])
    corr = float(np.corrcoef(lc, le)[0, 1])

    e = s.panel[(s.panel["pop"] == "exposed") & (s.panel["n"] > 0)].copy()
    e["mu0"] = s.model.hazard(e) * e["n"]
    ew = e.groupby("wk")[["e_n", "mu0"]].sum().join(np.log(j["theta"]).rename("lt"), how="inner")
    X = sm.add_constant(ew["lt"].values)
    glm = sm.GLM(ew["e_n"].values, X, family=sm.families.Poisson(),
                 offset=np.log(ew["mu0"].values)).fit(scale="X2")
    beta, se = float(glm.params[1]), float(glm.bse[1])
    r2 = corr ** 2


    ex = s.panel[(s.panel["pop"] == "exposed") & (s.panel["n"] > 0)].copy()
    ex["mu0"] = s.model.hazard(ex) * ex["n"]
    ew13 = ex.groupby("wk")[["e_n", "mu0"]].sum().sort_index().rolling(13).sum()
    cc = s.panel[(s.panel["pop"] == "control") & (s.panel["n"] > 0)].copy()
    cc["mu0"] = s.model.hazard(cc) * cc["n"]
    cw13 = cc.groupby("wk")[["e_n", "mu0"]].sum().sort_index().rolling(13).sum()
    r = ((ew13["e_n"] / ew13["mu0"]) / (cw13["e_n"] / cw13["mu0"])).replace([np.inf, -np.inf], np.nan)
    d = (r - 1.0).dropna()
    drift = {}
    for name, (a, b) in {"train": s.train, "eval": s.evalw}.items():
        dd = d.loc[a:b].abs()
        if len(dd):
            drift[name] = {"q50": float(dd.quantile(0.5)), "q90": float(dd.quantile(0.9)),
                           "q99": float(dd.quantile(0.99))}
    w = s.ev(s.w_cal)
    if "train" in drift:
        drift["floor_btc_per_week_median_at_train_q90"] = float(
            (2 * drift["train"]["q90"] * w["exp_v"]).median())
    return {"weeks": int(len(j)), "corr_log_theta": corr, "r2": r2,
            "glm_beta": beta, "glm_beta_ci95": [beta - 1.96 * se, beta + 1.96 * se],
            "glm_dispersion": float(glm.scale), "differential_drift": drift}


def _det_value(s: Setup, w_all: pd.DataFrame, phi: float) -> C.RobustDetector:
    d = C.RobustDetector(phi=phi)
    z_tr = d.z(s.tr(w_all))
    return C.calibrate(d, z_tr, s.target_arl, s.rng)


def _count_stats(w: pd.DataFrame) -> pd.DataFrame:
    w = w.copy()
    w["var_n"] = w["exp_n"].clip(lower=1e-9)
    return w


def detectors(s: Setup) -> dict:


    if getattr(s, "_dets", None) is not None:
        return s._dets
    dets = {}
    dets["argus"] = ("S", "var", True, _det_value(s, s.w_cal, s.phi))
    dets["uncalibrated"] = ("S", "var", False, _det_value(s, s.w_unc, s.phi_unc))
    wc = _count_stats(s.w_cal)


    phi_n = C.estimate_phi(s.tr(wc)[["Sn", "var_n"]].rename(columns={"Sn": "S", "var_n": "var"}))
    dn = C.RobustDetector(phi=phi_n)
    dn = C.calibrate(dn, dn.z(s.tr(wc), "Sn", "var_n"), s.target_arl, s.rng)
    dets["count"] = ("Sn", "var_n", True, dn)


    margin = C.drift_margin(s.panel, s.model, s.train)
    dets["eprocess"] = ("E", None, True, C.EDetector(c=s.target_arl, drift=margin))


    z_tr = C.RobustDetector(phi=s.phi).z(s.tr(s.w_cal))
    for k, (name, det) in enumerate((("focus", C.FOCuSDetector(phi=s.phi)),
                                     ("newma", C.NEWMADetector(phi=s.phi, ref=z_tr)))):
        rng = np.random.default_rng(s.seed + 201 + k)
        dets[name] = ("S", "var", True, C.calibrate(det, z_tr, s.target_arl, rng))
    dets["ctm"] = ("CTM", None, True, C.ConformalDetector(c=s.target_arl))
    x_tr = C.ecod_features(s.tr(s.w_cal), s.w_band, s.phi)
    dets["ecod"] = ("ECOD", None, True, C.ECODDetector().fit(x_tr, s.target_arl))
    s._dets = dets


    return dets


def run_detector(s: Setup, name: str, spec, injected: pd.DataFrame | None = None) -> pd.Series:

    col, var_col, calibrated, det = spec
    w = (s.w_cal if calibrated else s.w_unc).loc[s.train[0]:].copy()
    if injected is not None:
        inj = injected.reindex(w.index).fillna(0.0)
        w["obs_v"] += inj["add_v"]
        w["obs_n"] += inj["add_n"]
        w["S"] = w["obs_v"] - w["exp_v"]
        w["Sn"] = w["obs_n"] - w["exp_n"]
    if col == "E":
        return pd.Series(det.run(det.y(w)), index=w.index)
    if col == "CTM":
        z = w["S"].values / np.sqrt(s.phi * w["var"].clip(lower=1e-12).values)
        return pd.Series(det.run(det.pvalues(np.array([]), z)), index=w.index)
    if col == "ECOD":
        return pd.Series(det.run(C.ecod_features(w, s.w_band, s.phi)), index=w.index)
    if col == "B":
        z = det.z_frame(s.w_band).reindex(w.index).fillna(0.0)
        return pd.Series(det.run_z(z), index=w.index)
    if col == "Sn":
        w = _count_stats(w)
    z = pd.Series(det.z(w, col, var_col), index=w.index)
    return pd.Series(det.run(z.values), index=w.index)


def rq3_false_alarms(s: Setup, big: pd.DataFrame | None = None) -> dict:
    dets = detectors(s)
    out = {"target_arl_weeks": s.target_arl, "eval_weeks": int(len(s.ev(s.w_cal)))}
    for name, spec in dets.items():
        a = s.ev(run_detector(s, name, spec))
        weeks = [d.date().isoformat() for d in a.index[a.values]]
        out[name] = {"alarms": int(a.sum()), "alarm_weeks": weeks,
                     "observed_arl": float(len(a) / max(a.sum(), 1)),
                     **{k: getattr(spec[3], k) for k in ("h", "z_max", "phi", "c", "drift", "window", "lam")
                        if hasattr(spec[3], k)}}
    if big is not None:
        thr = threshold_rule_level(s, big)
        b = big[(big["pop"] == "exposed")].copy()
        b["wk"] = pd.to_datetime(b["wk"])
        ev = b[(b["wk"] >= s.evalw[0]) & (b["wk"] < s.evalw[1]) & (b["sat"] / 1e8 >= thr)]
        out["threshold_rule"] = {"threshold_btc": thr, "alarms": int(ev["wk"].nunique()),
                                 "observed_arl": float(len(s.ev(s.w_cal)) / max(ev["wk"].nunique(), 1))}
    return out


def threshold_rule_level(s: Setup, big: pd.DataFrame) -> float:

    b = big[big["pop"] == "exposed"].copy()
    b["wk"] = pd.to_datetime(b["wk"])
    tr = b[(b["wk"] >= s.train[0]) & (b["wk"] < s.train[1])]
    weekly_max = tr.groupby("wk")["sat"].max() / 1e8
    n_weeks = len(s.tr(s.w_cal))
    k = max(int(n_weeks / s.target_arl), 1)
    return float(np.sort(weekly_max.values)[::-1][min(k, len(weekly_max) - 1)])


