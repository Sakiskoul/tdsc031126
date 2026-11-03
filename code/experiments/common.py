from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import pandas as pd
import data_io as IO

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
STRATUM = ["fam", "era", "vband", "band"]
KEY = ["pop"] + STRATUM


def load_panel(results_dir: Path, t_end: str | None = None) -> pd.DataFrame:

    risk = IO.read_frame(results_dir / "risk_weekly.parquet")
    ev = IO.read_frame(results_dir / "events_weekly.parquet")
    risk["wk"] = pd.to_datetime(risk["wk"])
    ev["wk"] = pd.to_datetime(ev["wk"])
    last = pd.Timestamp(t_end) if t_end else risk["wk"].max()
    first = risk["wk"].min()
    weeks = pd.date_range(first, last, freq="W-MON")
    parts = []
    for key, g in risk.groupby(KEY, sort=False):
        g = g.set_index("wk")[["n", "sv", "sv2"]].sort_index()
        g = g[~g.index.duplicated(keep="last")].reindex(weeks).ffill().fillna(0.0)
        g = g.reset_index().rename(columns={"index": "wk"})
        for k, v in zip(KEY, key):
            g[k] = v
        parts.append(g)
    p = pd.concat(parts, ignore_index=True)
    p["sv"] = p["sv"] / 1e8
    p.loc[p["n"] <= 0, ["n", "sv", "sv2"]] = 0.0
    p["sv2"] = p["sv2"].clip(lower=0.0)
    ev = ev.rename(columns={"n": "e_n", "sv": "e_sv", "sv2": "e_sv2"})
    ev["e_sv"] = ev["e_sv"] / 1e8
    p = p.merge(ev[KEY + ["wk", "e_n", "e_sv", "e_sv2"]], on=KEY + ["wk"], how="left")
    p[["e_n", "e_sv", "e_sv2"]] = p[["e_n", "e_sv", "e_sv2"]].fillna(0.0)
    return p[p["wk"] < last].reset_index(drop=True)


@dataclass
class NullModel:


    shrink: float = 2000.0
    iters: int = 50
    lam: pd.Series = field(default=None)
    theta_train: pd.Series = field(default=None)

    def fit(self, panel: pd.DataFrame) -> "NullModel":
        d = panel[panel["n"] > 0]
        theta = pd.Series(1.0, index=sorted(d["wk"].unique()))
        parent = ["pop", "fam", "band"]
        for _ in range(self.iters):
            th = d["wk"].map(theta)
            exp_w = d["n"] * th
            g = d.assign(w=exp_w).groupby(KEY)[["e_n", "w"]].sum()
            gp = d.assign(w=exp_w).groupby(parent)[["e_n", "w"]].sum()
            lam_parent = (gp["e_n"] / gp["w"]).rename("lp")
            g = g.join(lam_parent, on=parent)
            lam = (g["e_n"] + self.shrink * g["lp"]) / (g["w"] + self.shrink)
            mu = d[KEY].merge(lam.rename("lam").reset_index(), on=KEY, how="left")["lam"].values * d["n"].values
            num = d.assign(mu=mu).groupby("wk")[["e_n", "mu"]].sum()
            theta = (num["e_n"] / num["mu"]).fillna(1.0)
            theta = theta / theta.mean()
        self.lam = lam
        self.theta_train = theta
        return self

    def hazard(self, panel: pd.DataFrame) -> np.ndarray:
        lam = panel[KEY].merge(self.lam.rename("lam").reset_index(), on=KEY, how="left")["lam"]
        return lam.fillna(0.0).values


def control_theta(panel: pd.DataFrame, model: NullModel, prior_events: float = 20.0,
                  span: int = 8) -> pd.DataFrame:


    c = panel[(panel["pop"] == "control") & (panel["n"] > 0)].copy()
    c["mu0"] = model.hazard(c) * c["n"]
    w = c.groupby("wk")[["e_n", "mu0"]].sum().sort_index()
    raw = (w["e_n"] / w["mu0"]).replace([np.inf, -np.inf], np.nan)
    prior = raw.ewm(span=span, min_periods=1).mean().shift(1).fillna(1.0)
    w["theta"] = (w["e_n"] + prior_events * prior) / (w["mu0"] + prior_events)
    w["var_theta"] = w["theta"].clip(lower=1e-9) / (w["mu0"] + prior_events)
    w["theta_raw"] = raw
    return w[["theta", "var_theta", "mu0", "theta_raw"]]


def exposed_theta(panel: pd.DataFrame, model: NullModel) -> pd.Series:

    e = panel[(panel["pop"] == "exposed") & (panel["n"] > 0)].copy()
    e["mu0"] = model.hazard(e) * e["n"]
    w = e.groupby("wk")[["e_n", "mu0"]].sum()
    return w["e_n"] / w["mu0"]


def weekly_stats(panel: pd.DataFrame, model: NullModel, calibrated: bool = True,
                 injected: pd.DataFrame | None = None) -> pd.DataFrame:


    e = panel[(panel["pop"] == "exposed") & (panel["n"] > 0)].copy()
    e["h"] = model.hazard(e)
    if calibrated:
        th = control_theta(panel, model)
        e = e.join(th, on="wk")
        e["theta"] = e["theta"].fillna(1.0)
        e["var_theta"] = e["var_theta"].fillna(0.0)
    else:
        e["theta"] = 1.0
        e["var_theta"] = 0.0
    e["p"] = (e["theta"] * e["h"]).clip(upper=0.25)
    e["exp_v"] = e["p"] * e["sv"]
    e["bin_var"] = e["p"] * (1 - e["p"]) * e["sv2"]
    e["energy"] = e["p"] * e["sv2"]
    e["hF"] = e["h"] * e["sv"]
    e["lam0"] = e["h"] * e["n"]
    e["exp_n"] = e["p"] * e["n"]
    w = e.groupby("wk").agg(obs_v=("e_sv", "sum"), obs_n=("e_n", "sum"), exp_v=("exp_v", "sum"),
                            exp_n=("exp_n", "sum"),
                            bin_var=("bin_var", "sum"), energy=("energy", "sum"),
                            hF=("hF", "sum"), var_theta=("var_theta", "first"),
                            stock_v=("sv", "sum"), stock_n=("n", "sum"), lam0_A=("lam0", "sum"))

    c = panel[(panel["pop"] == "control") & (panel["n"] > 0)].copy()
    c["lam0"] = model.hazard(c) * c["n"]
    cw = c.groupby("wk").agg(ctl_n=("e_n", "sum"), lam0_C=("lam0", "sum"))
    w = w.join(cw, how="left").fillna({"ctl_n": 0.0, "lam0_C": 0.0})
    if injected is not None:
        w = w.join(injected, how="left").fillna({"add_v": 0.0, "add_n": 0.0})
        w["obs_v"] += w["add_v"]
        w["obs_n"] += w["add_n"]
    w["S"] = w["obs_v"] - w["exp_v"]
    w["var"] = w["bin_var"] + w["var_theta"] * w["hF"] ** 2
    w["Sn"] = w["obs_n"] - w["exp_n"]
    return w


BANDS = ["v0_001", "v0_01", "v0_1", "v1", "v10", "v100"]


def weekly_band_stats(panel: pd.DataFrame, model: NullModel, calibrated: bool = True) -> pd.DataFrame:


    e = panel[(panel["pop"] == "exposed") & (panel["n"] > 0)].copy()
    e["h"] = model.hazard(e)
    if calibrated:
        th = control_theta(panel, model)
        e = e.join(th, on="wk")
        e["theta"] = e["theta"].fillna(1.0)
        e["var_theta"] = e["var_theta"].fillna(0.0)
    else:
        e["theta"] = 1.0
        e["var_theta"] = 0.0
    e["p"] = (e["theta"] * e["h"]).clip(upper=0.25)
    e["exp_v"] = e["p"] * e["sv"]
    e["bin_var"] = e["p"] * (1 - e["p"]) * e["sv2"]
    e["energy"] = e["p"] * e["sv2"]
    e["hF"] = e["h"] * e["sv"]
    e["exp_n"] = e["p"] * e["n"]
    w = e.groupby(["wk", "vband"]).agg(obs_v=("e_sv", "sum"), obs_n=("e_n", "sum"),
                                       exp_v=("exp_v", "sum"), exp_n=("exp_n", "sum"),
                                       bin_var=("bin_var", "sum"), energy=("energy", "sum"),
                                       hF=("hF", "sum"), var_theta=("var_theta", "first"),
                                       stock_v=("sv", "sum"), stock_n=("n", "sum"))
    w["S"] = w["obs_v"] - w["exp_v"]
    w["var"] = w["bin_var"] + w["var_theta"] * w["hF"] ** 2
    return w.reset_index(level="vband")


def band_shares(wb: pd.DataFrame, law: str) -> pd.DataFrame:


    col = {"tilted": "energy", "mimic": "exp_v", "uniform": "stock_v"}.get(law)
    if col is None:
        sh = pd.DataFrame(0.0, index=wb.index.unique(), columns=BANDS)
        sh["v100"] = 1.0
        return sh
    m = wb.pivot_table(index=wb.index, columns="vband", values=col, aggfunc="sum").reindex(columns=BANDS).fillna(0.0)
    return m.div(m.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)


def band_unit_value(wb: pd.DataFrame, law: str) -> pd.DataFrame:

    num, den = {"tilted": ("energy", "exp_v"), "mimic": ("exp_v", "exp_n")}.get(law, ("stock_v", "stock_n"))
    a = wb.pivot_table(index=wb.index, columns="vband", values=num, aggfunc="sum").reindex(columns=BANDS)
    b = wb.pivot_table(index=wb.index, columns="vband", values=den, aggfunc="sum").reindex(columns=BANDS)
    return (a / b.replace(0, np.nan)).fillna(0.0)


@dataclass
class RobustDetector:


    phi: float = 1.0
    k: float = 0.5
    cap: float = 4.0
    h: float = 8.0
    z_max: float = 12.0

    def z(self, w: pd.DataFrame, col: str = "S", var_col: str = "var") -> np.ndarray:
        return (w[col] / np.sqrt(self.phi * w[var_col].clip(lower=1e-12))).values

    def winsor(self, z):


        return np.minimum(np.asarray(z, float), self.cap)

    def run(self, z: np.ndarray) -> np.ndarray:

        g, out = 0.0, np.zeros(len(z), dtype=bool)
        for t, zt in enumerate(z):
            g = max(0.0, g + min(zt, self.cap) - self.k)
            if g > self.h or zt > self.z_max:
                out[t] = True
                g = 0.0
        return out

    def scores(self, z: np.ndarray) -> np.ndarray:


        z = np.asarray(z, float)
        g, acc = 0.0, np.empty(len(z))
        for t, zt in enumerate(z):
            g = max(0.0, g + min(zt, self.cap) - self.k)
            acc[t] = g
        return np.maximum(acc / self.h, z / self.z_max)


@dataclass
class GLRDetector(RobustDetector):


    window: int = 52
    h_max: float = 1000.0

    def run(self, z: np.ndarray) -> np.ndarray:
        return self.run_many(np.asarray(z, float)[None, :])[0]

    def run_many(self, Z: np.ndarray) -> np.ndarray:

        n, T = Z.shape
        W = T if self.window is None else self.window
        zc = self.winsor(Z)
        P = np.zeros((n, W))
        valid = np.zeros(n, dtype=int)
        lens = np.arange(1, W + 1, dtype=float)
        out = np.zeros((n, T), dtype=bool)
        for t in range(T):
            P[:, 1:] = P[:, :-1] + zc[:, t:t + 1]
            P[:, 0] = zc[:, t]
            valid = np.minimum(valid + 1, W)
            g = np.where(lens[None, :] <= valid[:, None], np.maximum(P, 0.0) ** 2 / (2 * lens), 0.0)
            a = (g.max(axis=1) > self.h) | (Z[:, t] > self.z_max)
            out[:, t] = a
            valid[a] = 0
        return out

    def scores(self, z: np.ndarray) -> np.ndarray:

        z = np.asarray(z, float)
        T = len(z)
        W = T if self.window is None else self.window
        zc = self.winsor(z)
        P, acc = np.zeros(W), np.empty(T)
        lens = np.arange(1, W + 1, dtype=float)
        for t in range(T):
            P[1:] = P[:-1] + zc[t]
            P[0] = zc[t]
            n = min(t + 1, W)
            acc[t] = float((np.maximum(P[:n], 0.0) ** 2 / (2 * lens[:n])).max())
        return np.maximum(acc / self.h, z / self.z_max)


@dataclass
class EWMADetector(RobustDetector):


    lam: float = 0.1

    def run(self, z: np.ndarray) -> np.ndarray:
        return self.run_many(np.asarray(z, float)[None, :])[0]

    def run_many(self, Z: np.ndarray) -> np.ndarray:
        n, T = Z.shape
        zc = self.winsor(Z)
        lim = self.h * np.sqrt(self.lam / (2.0 - self.lam))
        e = np.zeros(n)
        out = np.zeros((n, T), dtype=bool)
        for t in range(T):
            e = np.maximum(0.0, (1.0 - self.lam) * e + self.lam * zc[:, t])
            a = (e > lim) | (Z[:, t] > self.z_max)
            out[:, t] = a
            e[a] = 0.0
        return out


@dataclass
class FOCuSDetector(GLRDetector):


    window: int | None = None


@dataclass
class NEWMADetector(RobustDetector):


    big_lam: float = 0.1341462260694219
    lam: float = 0.03109641791071921
    m: int = 50
    seed: int = 20200427
    ref: np.ndarray | None = None
    h_min: float = 0.0
    h_max: float = 4.0

    def frequencies(self) -> np.ndarray:
        r = self.winsor(self.ref if self.ref is not None else [0.0, 1.0])
        sigmasq = float(np.median(((r[:, None] - r[None, :]) ** 2).ravel()))
        return np.random.default_rng(self.seed).standard_normal(self.m) / np.sqrt(sigmasq)

    def features(self, x: np.ndarray, W: np.ndarray) -> np.ndarray:
        t = np.asarray(x, float)[..., None] * W
        return np.concatenate([np.cos(t), np.sin(t)], axis=-1) / np.sqrt(self.m)

    def start(self, W: np.ndarray) -> np.ndarray:

        r = self.winsor(self.ref if self.ref is not None else [0.0])
        return self.features(r, W).mean(axis=0)

    def statistics(self, z: np.ndarray, init: float | None = None) -> np.ndarray:


        W = self.frequencies()
        F = self.features(self.winsor(z), W)
        s0 = self.start(W) if init is None else self.features(self.winsor(init), W)
        fast, slow = s0.copy(), s0.copy()
        out = np.empty(len(F))
        for t in range(len(F)):
            fast = (1.0 - self.big_lam) * fast + self.big_lam * F[t]
            slow = (1.0 - self.lam) * slow + self.lam * F[t]
            out[t] = np.linalg.norm(fast - slow)
        return out

    def run(self, z: np.ndarray) -> np.ndarray:
        return self.run_many(np.asarray(z, float)[None, :])[0]

    def scores(self, z: np.ndarray) -> np.ndarray:

        z = np.asarray(z, float)
        return np.maximum(self.statistics(z) / self.h, z / self.z_max)

    def run_many(self, Z: np.ndarray) -> np.ndarray:
        n, T = Z.shape
        W = self.frequencies()
        F = self.features(self.winsor(Z), W)
        s0 = self.start(W)
        fast = np.tile(s0, (n, 1))
        slow = np.tile(s0, (n, 1))
        restart = np.zeros(n, dtype=bool)
        out = np.zeros((n, T), dtype=bool)
        for t in range(T):
            fast[restart] = s0
            slow[restart] = s0
            fast = (1.0 - self.big_lam) * fast + self.big_lam * F[:, t]
            slow = (1.0 - self.lam) * slow + self.lam * F[:, t]
            a = (np.linalg.norm(fast - slow, axis=1) > self.h) | (Z[:, t] > self.z_max)
            out[:, t] = a
            restart = a
        return out


@dataclass
class ConformalDetector:


    c: float = 104.0
    eps: tuple = (0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9)
    seed: int = 20210220

    def pvalues(self, hist: np.ndarray, z: np.ndarray) -> np.ndarray:
        rng = np.random.default_rng(self.seed)
        seq = np.concatenate([np.asarray(hist, float), np.asarray(z, float)])
        k = len(hist)
        out = np.empty(len(z))
        for t in range(len(z)):
            past = seq[: k + t + 1]
            a = seq[k + t]
            out[t] = (np.sum(past > a) + rng.uniform() * np.sum(past == a)) / len(past)
        return np.clip(out, 1e-12, 1.0)

    def run(self, p: np.ndarray) -> np.ndarray:
        e = np.asarray(self.eps, float)
        m = np.zeros(len(e))
        out = np.zeros(len(p), dtype=bool)
        for t, pt in enumerate(p):
            m = e * pt ** (e - 1.0) * (m + 1.0)
            if m.mean() >= self.c:
                out[t] = True
                m[:] = 0.0
        return out

    def scores(self, p: np.ndarray) -> np.ndarray:


        e = np.asarray(self.eps, float)
        m, out = np.zeros(len(e)), np.empty(len(p))
        for t, pt in enumerate(p):
            m = e * pt ** (e - 1.0) * (m + 1.0)
            out[t] = m.mean() / self.c
        return out


@dataclass
class ECODDetector:


    model: object = None

    def fit(self, train: np.ndarray, target_arl: float) -> "ECODDetector":
        from pyod.models.ecod import ECOD
        self.model = ECOD(contamination=1.0 / target_arl).fit(np.asarray(train, float))
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(np.asarray(X, float))
        return np.array([self.model.decision_function(x[None, :])[0] for x in X])

    def run(self, X: np.ndarray) -> np.ndarray:
        return self.score(X) > self.model.threshold_

    def scores(self, X: np.ndarray) -> np.ndarray:

        return self.score(X) / self.model.threshold_


def ecod_features(w: pd.DataFrame, wb: pd.DataFrame, phi: float, add_v=None, add_n=None,
                  shares: pd.DataFrame | None = None) -> np.ndarray:


    S = wb.pivot_table(index=wb.index, columns="vband", values="S", aggfunc="sum").reindex(index=w.index, columns=BANDS).fillna(0.0)
    V = wb.pivot_table(index=wb.index, columns="vband", values="var", aggfunc="sum").reindex(index=w.index, columns=BANDS).fillna(0.0)
    av = np.zeros(len(w)) if add_v is None else np.asarray(add_v, float)
    an = np.zeros(len(w)) if add_n is None else np.asarray(add_n, float)
    if shares is not None:
        S = S + shares.reindex(index=w.index, columns=BANDS).fillna(0.0).values * av[:, None]
    zb = (S / np.sqrt(V.clip(lower=1e-12))).values
    zn = (w["Sn"].values + an) / np.sqrt(w["exp_n"].clip(lower=1e-9).values)
    z = (w["S"].values + av) / np.sqrt(phi * w["var"].clip(lower=1e-12).values)
    return np.column_stack([zb, zn, z])


@dataclass
class EDetector:


    c: float = 104.0
    drift: float = 0.0
    lambdas: tuple = (0.01, 0.02, 0.05, 0.1, 0.2, 0.4, 0.8)

    def y(self, w: pd.DataFrame, add_v=None, add_n=None) -> np.ndarray:
        r = 1.0 + self.drift
        obs_v = w["obs_v"].values + (0.0 if add_v is None else add_v)
        n = w["obs_n"].values + w["ctl_n"].values + (0.0 if add_n is None else add_n)
        m = r * w["hF"].values / np.clip(r * w["lam0_A"].values + w["lam0_C"].values, 1e-12, None)
        with np.errstate(divide="ignore", invalid="ignore"):
            y = np.where((n > 0) & (m > 0), obs_v / (n * m) - 1.0, 0.0)
        return np.maximum(y, -1.0)

    def run(self, y: np.ndarray) -> np.ndarray:

        lam = np.asarray(self.lambdas, float)
        w = np.full(len(lam), 1.0 / len(lam))
        m = np.zeros(len(lam))
        out = np.zeros(len(y), dtype=bool)
        for t, yt in enumerate(y):
            m = (1.0 + lam * yt) * (m + 1.0)
            if float(w @ m) >= self.c:
                out[t] = True
                m[:] = 0.0
        return out

    def scores(self, y: np.ndarray) -> np.ndarray:

        lam = np.asarray(self.lambdas, float)
        w = np.full(len(lam), 1.0 / len(lam))
        m, out = np.zeros(len(lam)), np.empty(len(y))
        for t, yt in enumerate(y):
            m = (1.0 + lam * yt) * (m + 1.0)
            out[t] = float(w @ m) / self.c
        return out


@dataclass
class BandDetector:


    dets: dict = field(default_factory=dict)

    def z_frame(self, wb: pd.DataFrame, add_v: pd.DataFrame | None = None) -> pd.DataFrame:
        S = wb.pivot_table(index=wb.index, columns="vband", values="S", aggfunc="sum").reindex(columns=BANDS)
        V = wb.pivot_table(index=wb.index, columns="vband", values="var", aggfunc="sum").reindex(columns=BANDS)
        if add_v is not None:
            S = S.add(add_v.reindex(index=S.index, columns=BANDS).fillna(0.0), fill_value=0.0)
        z = pd.DataFrame(index=S.index)
        for b in BANDS:
            d = self.dets.get(b)
            if d is None:
                continue
            z[b] = (S[b] / np.sqrt(d.phi * V[b].clip(lower=1e-12))).fillna(0.0)
        return z

    def run_z(self, z: pd.DataFrame) -> np.ndarray:
        out = np.zeros(len(z), dtype=bool)
        for b, d in self.dets.items():
            if b in z:
                out |= d.run(z[b].values)
        return out


def calibrate_bands(wb_train: pd.DataFrame, target_arl: float, rng) -> BandDetector:
    bd = BandDetector()
    present = [b for b in BANDS if (wb_train["vband"] == b).any()]
    for b in present:
        tb = wb_train[wb_train["vband"] == b]
        if tb["var"].sum() <= 0:
            continue
        d = RobustDetector(phi=estimate_phi(tb[["S", "var"]]))
        z = (tb["S"] / np.sqrt(d.phi * tb["var"].clip(lower=1e-12))).values
        bd.dets[b] = calibrate(d, z, target_arl * len(present), rng)
    return bd


def drift_series(panel: pd.DataFrame, model: "NullModel", window: int = 13) -> pd.Series:

    def rate(pop):
        x = panel[(panel["pop"] == pop) & (panel["n"] > 0)].copy()
        x["mu0"] = model.hazard(x) * x["n"]
        return x.groupby("wk")[["e_n", "mu0"]].sum().sort_index().rolling(window).sum()
    ex, cc = rate("exposed"), rate("control")
    r = ((ex["e_n"] / ex["mu0"]) / (cc["e_n"] / cc["mu0"])).replace([np.inf, -np.inf], np.nan)
    return (r - 1.0).dropna().rename("drift13")


def drift_margin(panel: pd.DataFrame, model: "NullModel", train: tuple[str, str], q: float = 0.95,
                 window: int = 13) -> float:


    d = drift_series(panel, model, window).loc[train[0]:train[1]]
    return float(max(d.quantile(q), 0.0)) if len(d) else 0.0


def estimate_phi(w_train: pd.DataFrame) -> float:


    if w_train.columns.duplicated().any():
        raise ValueError(f"duplicate columns {list(w_train.columns)}")
    r = w_train["S"] ** 2 / w_train["var"].clip(lower=1e-12)
    med = float(np.median(r))
    if not np.isfinite(med):
        raise ValueError("non-finite dispersion estimate")

    return float(max(1.0, med / 0.4549))


def block_bootstrap(z: np.ndarray, n_weeks: int, block: int, rng) -> np.ndarray:
    out = []
    while len(out) < n_weeks:
        s = rng.integers(0, len(z) - block)
        out.extend(z[s:s + block])
    return np.array(out[:n_weeks])


def calibrate(det: RobustDetector, z_train: np.ndarray, target_arl: float, rng,
              n_rep: int = 400, block: int = 8) -> RobustDetector:

    sims = [block_bootstrap(z_train, int(4 * target_arl), block, rng) for _ in range(n_rep)]
    allz = np.concatenate(sims)
    det.z_max = float(np.quantile(allz, 1 - 0.5 / target_arl))
    lo, hi = getattr(det, "h_min", 0.5), getattr(det, "h_max", 200.0)
    batch = np.stack(sims) if hasattr(det, "run_many") else None
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        d = replace(det, h=mid)
        if batch is not None:
            rate = float(d.run_many(batch).mean())
        else:
            rate = np.mean([d.run(s).sum() / len(s) for s in sims])
        if rate > 1.0 / target_arl:
            lo = mid
        else:
            hi = mid
    det.h = hi
    return det


def save_json(name: str, obj) -> Path:
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / f"{name}.json"
    path.write_text(json.dumps(obj, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    return path
