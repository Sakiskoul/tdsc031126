from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import data_io as IO

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C
import rq
import rq56

TRAIN = ("2014-01-06", "2019-12-30")
EVAL = ("2020-01-06", "2026-09-28")
T_END = "2026-10-01"
HORIZON, BLOCK = 52, 13
START_STRIDE = 4
INTENSITIES = [0.25, 0.5, 1.0, 2.0, 4.0]
MU = 1000
BAND_EDGES = np.array([0.01, 0.1, 1.0, 10.0, 100.0])
KAPPAS = np.geomspace(0.25, 8.0, 41)
FA_TARGETS = (0.05, 0.10, 0.20)


def cached_panel(results: Path) -> pd.DataFrame:
    cache = results / "panel_cache.parquet"
    if cache.exists():
        return IO.read_frame(cache)
    p = C.load_panel(results, T_END)
    p.to_parquet(cache)
    return p


def starts(panel: pd.DataFrame) -> list[pd.Timestamp]:
    weeks = pd.DatetimeIndex(sorted(panel["wk"].unique()))
    ev = weeks[(weeks >= EVAL[0]) & (weeks <= EVAL[1])]
    return list(ev[: len(ev) - HORIZON + 1: START_STRIDE])


def fit_window(panel: pd.DataFrame, block0: pd.Timestamp, lookback=104, gap=13) -> pd.DataFrame:

    hi = block0 - pd.Timedelta(weeks=gap)
    lo = hi - pd.Timedelta(weeks=lookback)
    return panel[(panel["wk"] >= lo) & (panel["wk"] < hi)]


def refit_anchor(week: pd.Timestamp) -> pd.Timestamp:

    origin = pd.Timestamp(EVAL[0])
    offset = int((pd.Timestamp(week) - origin) // pd.Timedelta(weeks=1))
    return origin + pd.Timedelta(weeks=(offset // BLOCK) * BLOCK)


def prep(args) -> int:
    if args.cache:
        light, *_ = pd.read_pickle(args.cache)
        panel = light.panel
    else:
        panel = cached_panel(Path(args.results))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    st = starts(panel)
    pd.DataFrame({"sid": range(len(st)), "t0": [t.date().isoformat() for t in st]}).to_csv(out / "starts.csv", index=False)
    rows = []
    models = {}
    for sid, t0 in enumerate(st):
        anchor = refit_anchor(t0)
        if anchor not in models:
            models[anchor] = C.NullModel().fit(fit_window(panel, anchor))
        lam = models[anchor].lam.rename("lam").reset_index()
        lam = lam[lam["pop"] == "exposed"]
        rows.append(lam.assign(sid=sid)[["sid", "fam", "era", "vband", "lam"]])
    pd.concat(rows).to_csv(out / "lam.csv", index=False)
    print(f"{len(st)} starts, {sum(len(r) for r in rows)} hazard rows -> {out}")
    return 0


def load_pool(pool_dir: Path, k: int, sid: int):


    import pyarrow.dataset as pds
    flt = pds.field("sid") == sid
    dataset = pds.dataset(sorted(pool_dir.glob("pool_*.parquet")), format="parquet")
    columns = dataset.schema.names + ["__filename"]
    pool = dataset.to_table(filter=flt, columns=columns).to_pandas(categories=["__filename"])
    if "draw" not in pool:
        pool["draw"] = 0
    kb = int((pool_dir / "meta.txt").read_text().split("kb=")[1].split()[0])
    check_pool_quota(pool, k, kb)
    pool = pool.sort_values(["law", "draw", "key"])
    pool["rank"] = pool.groupby(["law", "draw"]).cumcount()
    cut = pool[pool["rank"] == k - 1].set_index(["law", "draw"])["key"]
    pool = pool[pool["rank"] < k].drop(columns=["rank"])
    ids = pool["id"].unique()
    chg = pds.dataset(sorted(pool_dir.glob("chg_*.parquet")), format="parquet").to_table(filter=flt).to_pandas()
    ev = pds.dataset(sorted(pool_dir.glob("ev_*.parquet")), format="parquet").to_table(filter=flt).to_pandas()
    meta = {"kb": kb, "draws": int(pool["draw"].nunique()), "pool_rows": int(len(pool)),
            "bucket_quota_verified": True,
            "cut_max": float(cut.max()) if len(cut) else None}
    return pool.drop(columns=["__filename"]), chg[chg["id"].isin(ids)], ev[ev["id"].isin(ids)], meta


def check_pool_quota(pool: pd.DataFrame, k: int, kb: int) -> None:


    cut = pool.groupby(["law", "draw"], observed=True)["key"].apply(
        lambda x: x.nsmallest(k).max() if len(x) >= k else np.nan)
    tails = pool.groupby(["__filename", "law", "draw"], observed=True)["key"].agg(["size", "max"])
    for (_, law, draw), tail in tails[tails["size"] >= kb].iterrows():
        threshold = cut.loc[(law, draw)]
        if not np.isfinite(threshold) or tail["max"] <= threshold:
            raise ValueError(f"insufficient per-bucket replay quota for {law}, draw {draw}; re-export with larger kb")


def trajectories(pool_s: pd.DataFrame, chg_s: pd.DataFrame, ev_s: pd.DataFrame, t0: pd.Timestamp):

    ids = pool_s.drop_duplicates("id")[["id", "v0", "fam", "era"]].reset_index(drop=True)
    row = pd.Series(np.arange(len(ids)), index=ids["id"].values)
    V = np.full((len(ids), HORIZON), np.nan)
    V[:, 0] = ids["v0"].values / 1e8
    if len(chg_s):
        c = chg_s.sort_values("ts").copy()


        c["j"] = ((pd.to_datetime(c["ts"]) - t0) // pd.Timedelta(weeks=1)).values + 1
        c = c[(c["j"] >= 1) & (c["j"] < HORIZON)].drop_duplicates(["id", "j"], keep="last")
        V[row.loc[c["id"].values].values, c["j"].values] = c["v_new"].values / 1e8
    V = pd.DataFrame(V).ffill(axis=1).values
    V[V < 0] = 0.0
    E = np.zeros_like(V, dtype=bool)
    if len(ev_s):
        j = ((pd.to_datetime(ev_s["wk"]) - t0) // pd.Timedelta(weeks=1)).values
        ok = (j >= 0) & (j < HORIZON)
        E[row.loc[ev_s["id"].values[ok]].values, j[ok]] = True
    E &= V > 0
    return ids, row, V, E


def thief(order: np.ndarray, V: np.ndarray, plan: np.ndarray, mu: int = MU) -> np.ndarray:

    week = np.full(len(order), -1)
    Vo = V[order]
    budget = np.cumsum(plan)
    taken = 0.0
    for j in range(V.shape[1]):
        rem = budget[j] - taken
        if rem <= 0:
            continue
        v = Vo[:, j]
        avail = (week < 0) & (v > 0) & (v <= rem)
        cs = np.cumsum(np.where(avail, v, 0.0))
        idx = np.flatnonzero(avail & (cs <= rem))[:mu]
        week[idx] = j
        taken += float(v[idx].sum())
    return week


def vband(v: np.ndarray) -> np.ndarray:
    return np.asarray(C.BANDS, dtype=object)[np.searchsorted(BAND_EDGES, v, side="right")]


def deltas(ids: pd.DataFrame, rows: np.ndarray, weeks_taken: np.ndarray, V: np.ndarray, E: np.ndarray,
           wk: pd.DatetimeIndex) -> pd.DataFrame:

    cols = C.KEY + ["wk", "dn", "dsv", "dsv2", "de_n", "de_sv", "de_sv2"]
    if not len(rows):
        return pd.DataFrame(columns=cols)
    j = np.arange(HORIZON)[None, :]
    js = weeks_taken[:, None]
    Vr, Er = V[rows], E[rows]


    t_r = np.arange(len(rows))
    c_r, c_j = np.nonzero(Er & (j >= js))
    k_r, k_j = np.nonzero((Vr > 0) & (j > js))
    r = np.concatenate([t_r, c_r, k_r])
    jj = np.concatenate([weeks_taken, c_j, k_j])
    e = np.concatenate([np.ones(len(t_r)), -np.ones(len(c_r)), np.zeros(len(k_r))])
    n = np.concatenate([np.zeros(len(t_r) + len(c_r)), -np.ones(len(k_r))])
    val = Vr[r, jj]
    src = rows[r]
    d = pd.DataFrame({"pop": "exposed", "fam": ids["fam"].values[src], "era": ids["era"].values[src],
                      "vband": vband(val), "band": "all", "wk": wk[jj],
                      "dn": n, "dsv": n * val, "dsv2": n * val ** 2,
                      "de_n": e, "de_sv": e * val, "de_sv2": e * val ** 2})
    return d.groupby(C.KEY + ["wk"], as_index=False)[cols[6:]].sum()


def contaminate(panel: pd.DataFrame, d: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    if not len(d):
        return panel, 0.0
    p = panel.merge(d, on=C.KEY + ["wk"], how="left")
    miss = len(d) - int(p["dn"].notna().sum())
    for col, dc in (("n", "dn"), ("sv", "dsv"), ("sv2", "dsv2"), ("e_n", "de_n"), ("e_sv", "de_sv"), ("e_sv2", "de_sv2")):
        p[col] = (p[col] + p[dc].fillna(0.0)).clip(lower=0.0)
    return p[panel.columns], miss / max(len(d), 1)


class Replayer:
    def __init__(self, s: "rq.Setup", t0: pd.Timestamp):
        self.s, self.t0 = s, t0
        weeks = pd.DatetimeIndex(sorted(s.panel["wk"].unique()))
        self.win = weeks[weeks >= t0][:HORIZON]
        labels = pd.DatetimeIndex([refit_anchor(w) for w in self.win])
        self.anchors = list(labels.unique())
        self.blocks = [self.win[labels == anchor] for anchor in self.anchors]

        self.slice = s.panel["wk"] <= self.win[-1]
        cache = getattr(s, "_replay_null_models", {})
        for anchor in self.anchors:
            if anchor not in cache:
                cache[anchor] = C.NullModel().fit(fit_window(s.panel, anchor))
        s._replay_null_models = cache
        self.models = [cache[anchor] for anchor in self.anchors]

    def stats(self, panel: pd.DataFrame, frozen: bool = False, band: bool = True):


        p = panel[self.slice.values] if len(panel) == len(self.slice) else panel
        cal, unc, bands = [], [], []
        for anchor, b, m0 in zip(self.anchors, self.blocks, self.models):
            fit = fit_window(panel, anchor)
            m = m0 if frozen or fit["wk"].max() < self.t0 else C.NullModel().fit(fit)
            cal.append(C.weekly_stats(p, m, calibrated=True).reindex(b))
            unc.append(C.weekly_stats(p, m, calibrated=False).reindex(b))
            if band:
                wb = C.weekly_band_stats(p, m)
                bands.append(wb[wb.index.isin(b)])
        return pd.concat(cal), pd.concat(unc), (pd.concat(bands) if band else None)


def _inputs(col, det, w, wb, hist, phi):

    if col == "E":
        return det.y(w)
    if col == "CTM":
        z = w["S"].values / np.sqrt(phi * w["var"].clip(lower=1e-12).values)
        return det.pvalues(hist, z)
    if col == "ECOD":
        return None if wb is None else C.ecod_features(w, wb, phi)
    if col == "Sn":
        return w["Sn"].values / np.sqrt(det.phi * w["exp_n"].clip(lower=1e-9).values)
    return w["S"].values / np.sqrt(det.phi * w["var"].clip(lower=1e-12).values)


def alarms(dets: dict, wc: pd.DataFrame, wu: pd.DataFrame, wb: pd.DataFrame, thr: float,
           nat_max: np.ndarray, theft_max: np.ndarray, hist: np.ndarray, phi: float) -> dict:
    out = {}
    for name, (col, var_col, calibrated, det) in dets.items():
        x = _inputs(col, det, wc if calibrated else wu, wb, hist, phi)
        if x is not None:
            out[name] = det.run(x)
    out["threshold"] = np.maximum(nat_max, theft_max) >= thr
    return out


def det_scores(dets: dict, wc: pd.DataFrame, wu: pd.DataFrame, wb: pd.DataFrame, thr: float,
               nat_max: np.ndarray, theft_max: np.ndarray, hist: np.ndarray, phi: float) -> dict:


    out = {}
    for name, (col, var_col, calibrated, det) in dets.items():
        x = _inputs(col, det, wc if calibrated else wu, wb, hist, phi)
        out[name] = det.scores(x) if x is not None else np.full(HORIZON, np.nan)
    out["threshold"] = np.maximum(nat_max, theft_max) / thr
    return out


def natural_max_after_theft(events: pd.DataFrame, row: pd.Series, taken_rows: np.ndarray,
                            weeks_taken: np.ndarray, t0: pd.Timestamp) -> np.ndarray:


    j = ((pd.to_datetime(events["wk"]) - t0) // pd.Timedelta(weeks=1)).to_numpy()
    valid = (j >= 0) & (j < HORIZON)
    event = events.loc[valid]
    j = j[valid]
    position = row.reindex(event["id"].to_numpy()).fillna(-1).to_numpy(dtype=np.int64)
    removed_at = np.full(len(row), HORIZON, dtype=np.int16)
    removed_at[taken_rows] = weeks_taken
    known = position >= 0
    keep = np.ones(len(event), dtype=bool)
    keep[known] = j[known] < removed_at[position[known]]
    out = np.zeros(HORIZON)
    np.maximum.at(out, j[keep], event["value"].to_numpy()[keep])
    return out


def natural_movements(results: Path) -> pd.DataFrame:
    import duckdb

    big = IO.read_frame(results / "big_events.parquet")
    big = big[big["pop"] == "exposed"][["addr", "wk", "sat"]]
    con = duckdb.connect()
    con.register("events", big)
    events = con.execute("SELECT CAST(hash(addr) >> 1 AS BIGINT) AS id, wk, sat / 1e8 AS value FROM events").fetchdf()
    con.close()
    events["wk"] = pd.to_datetime(events["wk"])
    return events


def one_start(job) -> dict:
    sid, t0, s, dets, thr, nat_max, sigma, pool_dir, k, out_dir = job
    npz = Path(out_dir) / f"start_{sid:03d}.npz"
    js_path = Path(out_dir) / f"start_{sid:03d}.json"
    if npz.exists() and js_path.exists():
        return json.loads(js_path.read_text())
    t = time.time()
    tm = {"load": 0.0, "thief": 0.0, "deltas": 0.0, "stats": 0.0, "alarms": 0.0}
    pool_s, chg_s, ev_s, meta = load_pool(Path(pool_dir), k, sid)
    tm["load"] = time.time() - t
    draws = sorted(pool_s["draw"].unique())
    R = Replayer(s, t0)
    ids, row, V, E = trajectories(pool_s, chg_s, ev_s, t0)
    wc0, wu0, wb0 = R.stats(s.panel)
    zh = s.w_cal["S"] / np.sqrt(s.phi * s.w_cal["var"].clip(lower=1e-12))
    hist = zh[(zh.index >= s.train0) & (zh.index < t0)].values
    check = float((wc0["S"] - s.w_cal["S"].reindex(R.win)).abs().max() / s.w_cal["S"].abs().median())
    if not np.isfinite(check) or check > 1e-7:
        raise ValueError(f"null replay does not reproduce the reference: start {sid}, relative error {check:.3g}")
    nm = nat_max.reindex(R.win).fillna(0.0).values
    natural = s.natural_big[(s.natural_big["wk"] >= t0) & (s.natural_big["wk"] <= R.win[-1])]
    reproduced = natural_max_after_theft(natural, row, np.array([], dtype=int), np.array([], dtype=int), t0)
    if not np.allclose(reproduced, nm, rtol=1e-12, atol=1e-9):
        raise ValueError("natural large-event table does not reproduce the threshold-rule reference")
    zero = np.zeros(HORIZON)
    base = alarms(dets, wc0, wu0, wb0, thr, nm, zero, hist, s.phi)
    names = list(base)
    base_sc = det_scores(dets, wc0, wu0, wb0, thr, nm, zero, hist, s.phi)
    D, L, M, ND = len(draws), len(rq56.LAWS), len(INTENSITIES), len(names)
    SC = np.full((D, L, M, ND, HORIZON), np.nan, dtype=np.float32)
    STOLEN = np.zeros((D, L, M, HORIZON), dtype=np.float32)
    KEYS = np.zeros((D, L, M), dtype=np.int32)
    res = {"sid": sid, "t0": t0.date().isoformat(), "check_rel": check, "pool": meta,
           "natural": {d: int(a.any()) for d, a in base.items()}, "runs": []}
    plan_unit = sigma.reindex(R.win).values
    for di, draw in enumerate(draws):
        for li, law in enumerate(rq56.LAWS):
            pl = pool_s[(pool_s["law"] == law) & (pool_s["draw"] == draw)].sort_values("key")
            order = row.loc[pl["id"].values].values
            for mi, m in enumerate(INTENSITIES):
                plan = m * plan_unit
                t1 = time.time()
                wk_taken = thief(order, V, plan)
                tm["thief"] += time.time() - t1
                took = wk_taken >= 0
                rows_t, jt = order[took], wk_taken[took]
                stolen_w = np.bincount(jt, weights=V[rows_t, jt], minlength=HORIZON)
                keys_w = np.bincount(jt, minlength=HORIZON)
                tmax = np.zeros(HORIZON)
                np.maximum.at(tmax, jt, V[rows_t, jt])
                surviving_max = natural_max_after_theft(natural, row, rows_t, jt, t0)
                t1 = time.time()
                d = deltas(ids, rows_t, jt, V, E, R.win)
                pc, miss = contaminate(s.panel, d)
                t2 = time.time()
                wc, wu, wb = R.stats(pc)
                t3 = time.time()
                al = alarms(dets, wc, wu, wb, thr, surviving_max, tmax, hist, s.phi)
                sc = det_scores(dets, wc, wu, wb, thr, surviving_max, tmax, hist, s.phi)
                for ni, dn in enumerate(names):
                    SC[di, li, mi, ni] = sc[dn]
                STOLEN[di, li, mi] = stolen_w
                KEYS[di, li, mi] = int(took.sum())
                run = {"draw": int(draw), "law": law, "m": m, "planned_btc": float(plan.sum()),
                       "stolen_btc": float(stolen_w.sum()), "keys": int(took.sum()),
                       "keys_week_max": int(keys_w.max()), "pool_size": int(len(order)),
                       "unmatched_delta_share": miss, "det": {}}
                if di == 0:
                    wcf, wuf, wbf = R.stats(pc, frozen=True, band=False)
                    alf = alarms(dets, wcf, wuf, wbf, thr, surviving_max, tmax, hist, s.phi)
                    run["frozen_alarm"] = {dn: int(a.any()) for dn, a in alf.items()}
                    run["frozen_delay"] = {dn: (int(np.argmax(a)) + 1 if a.any() else None) for dn, a in alf.items()}
                tm["deltas"] += t2 - t1
                tm["stats"] += t3 - t2
                tm["alarms"] += time.time() - t3
                for dn, a in al.items():
                    if a.any():
                        kk = int(np.argmax(a))
                        run["det"][dn] = {"alarm": 1, "delay": kk + 1, "loss_btc": float(stolen_w[: kk + 1].sum())}
                    else:
                        run["det"][dn] = {"alarm": 0, "delay": None, "loss_btc": float(stolen_w.sum())}
                res["runs"].append(run)
    res["seconds"] = time.time() - t
    res["timing"] = tm
    np.savez_compressed(npz.with_suffix(".tmp.npz"), names=np.array(names), draws=np.array(draws),
                        laws=np.array(rq56.LAWS), intensities=np.array(INTENSITIES),
                        base=np.array([base_sc[n] for n in names], dtype=np.float32),
                        scores=SC, stolen=STOLEN, keys=KEYS)
    npz.with_suffix(".tmp.npz").rename(npz)
    js_path.write_text(json.dumps(res))
    print(f"start {sid} {t0.date()} done in {res['seconds']:.0f}s, draws {D}, check {check:.2e}, "
          + ", ".join(f"{k} {v:.0f}s" for k, v in tm.items()), flush=True)
    return res


def summarise(per_start: list[dict], det_names: list[str]) -> dict:


    n = len(per_start)
    block = max(1, HORIZON // START_STRIDE)
    idx = rq56.block_indices(n, block, n_boot=2000, rng=np.random.default_rng(20261018))
    nat = {d: np.array([r["natural"][d] for r in per_start], float) for d in det_names}
    out = {"starts": n, "natural_window_alarm_rate": {d: float(v.mean()) for d, v in nat.items()}, "laws": {}}
    for law in rq56.LAWS:
        L = {}
        for m in INTENSITIES:
            runs = [[x for x in r["runs"] if x["law"] == law and x["m"] == m] for r in per_start]
            flat = [x for rr in runs for x in rr]
            row = {"realised_share": float(np.median([x["stolen_btc"] / max(x["planned_btc"], 1e-9) for x in flat])),
                   "stolen_btc_median": float(np.median([x["stolen_btc"] for x in flat])),
                   "keys_median": float(np.median([x["keys"] for x in flat])),
                   "draws": int(np.median([len(rr) for rr in runs])), "det": {}}
            for d in det_names:
                a = np.array([np.mean([x["det"][d]["alarm"] for x in rr]) for rr in runs], float)
                per_draw = np.array([[x["det"][d]["alarm"] for x in rr] for rr in runs if len(rr) == row["draws"]], float)
                ex = a[idx].mean(axis=1) - nat[d][idx].mean(axis=1)
                delays = [x["det"][d]["delay"] for x in flat if x["det"][d]["alarm"]]
                loss = [x["det"][d]["loss_btc"] for x in flat]
                first = [rr[0] for rr in runs]
                af = np.array([x.get("frozen_alarm", {}).get(d, np.nan) for x in first], float)
                dfz = [x["frozen_delay"][d] for x in first if x.get("frozen_alarm", {}).get(d)]
                row["det"][d] = {"detection_rate": float(a.mean()), "excess": float(a.mean() - nat[d].mean()),
                                 "excess_draw_sd": float(per_draw.mean(axis=0).std()) if per_draw.size else None,
                                 "excess_frozen": (None if np.isnan(af).any() else float(af.mean() - nat[d].mean())),
                                 "excess_first_draw": float(np.mean([x["det"][d]["alarm"] for x in first]) - nat[d].mean()),
                                 "median_delay_frozen": float(np.median(dfz)) if dfz else None,
                                 "excess_ci95": [float(np.quantile(ex, 0.025)), float(np.quantile(ex, 0.975))],
                                 "median_delay": float(np.median(delays)) if delays else None,
                                 "p90_delay": float(np.quantile(delays, 0.9)) if delays else None,
                                 "median_loss_btc": float(np.median(loss)),
                                 "p90_loss_btc": float(np.quantile(loss, 0.9))}
            L[str(m)] = row
        out["laws"][law] = L
    return out


def aggregate_counterpart(s: "rq.Setup", big: pd.DataFrame, st: list[pd.Timestamp]) -> dict:

    b = rq56.Bench(s)
    b.set_threshold_rule(big)
    names = list(b.dets) + ["threshold"]
    weeks = s.ev(s.w_cal).index
    wst = rq56.whale_stock(s)
    out = {}
    for law in rq56.LAWS:
        uv_all = rq56.unit_value(s, s.w_cal, law)
        out[law] = {}
        for m in INTENSITIES:
            hits = {d: [] for d in names}
            for t0 in st:
                win = weeks[weeks >= t0][:HORIZON]
                plan = (m * b.sigma.loc[win]).values
                uv = uv_all.loc[win].values
                stock0 = float(wst.loc[t0] if law == "whale" else s.w_cal.loc[t0, "stock_v"])
                add_v, add_n = rq56.feasible_theft(plan, uv, MU, stock0)
                for d in names:
                    hits[d].append(float(b.alarms(d, win, add_v, add_n, uv, law=law).any()))
            out[law][str(m)] = {d: float(np.mean(h)) for d, h in hits.items()}
    return out


class Light:


    def __init__(self, s: "rq.Setup"):
        self.panel = s.panel
        self.w_cal = s.w_cal[["S", "var"]].copy()
        self.phi = s.phi
        self.train0 = pd.Timestamp(s.train[0])


def run(args) -> int:
    t = time.time()
    res_dir, pool_dir = Path(args.results), Path(args.pool)
    out_dir = pool_dir / f"runs_{args.tag}"
    out_dir.mkdir(exist_ok=True)
    cache = pool_dir / "setup_cache.pkl"
    if cache.exists():
        light, dets, thr, nat_max, sigma, agg = pd.read_pickle(cache)
    else:
        panel = cached_panel(res_dir)
        big = IO.read_frame(res_dir / "big_events.parquet")
        s = rq.Setup(panel, TRAIN, EVAL, target_arl=104.0, refit=13)
        dets = rq.detectors(s)
        b = rq56.Bench(s)
        b.set_threshold_rule(big)
        light, thr, nat_max, sigma = Light(s), b.thr, b.natural_max, b.sigma
        agg = aggregate_counterpart(s, big, starts(panel))
        pd.to_pickle((light, dets, thr, nat_max, sigma, agg), cache)
    st = starts(light.panel)
    light.natural_big = natural_movements(res_dir)
    if thr * float(KAPPAS.min()) < 10.0:
        raise ValueError("the threshold grid reaches below the large-event extraction floor")
    print(f"setup {time.time() - t:.0f}s, {len(st)} starts", flush=True)
    if args.setup_only:
        return 0
    jobs = [(sid, t0, light, dets, thr, nat_max, sigma, str(pool_dir), args.k, str(out_dir))
            for sid, t0 in enumerate(st) if args.only is None or sid in args.only]
    if args.workers <= 1:
        per = [one_start(j) for j in jobs]
    else:

        extra = {"max_tasks_per_child": 1} if sys.version_info >= (3, 11) else {}
        with ProcessPoolExecutor(max_workers=args.workers, **extra) as ex:
            per = list(ex.map(one_start, jobs))
    names = list(dets) + ["threshold"]
    out = {"horizon": HORIZON, "start_stride": START_STRIDE, "mu": MU, "intensities": INTENSITIES,
           "k": args.k, "threshold_btc": thr, "summary": summarise(per, names),
           "aggregate_same_starts": agg if args.only is None else None,
           "per_start": per, "seconds": time.time() - t}
    path = C.save_json(f"replay_{args.tag}", out)
    print("saved", path, f"in {time.time() - t:.0f}s")
    return 0


def first_passage(S: np.ndarray, kappa: float) -> np.ndarray:

    hit = S >= kappa
    return np.where(hit.any(axis=-1), hit.argmax(axis=-1), HORIZON)


def curves(args) -> int:


    run_dir = Path(args.pool) / f"runs_{args.tag}"
    files = sorted(run_dir.glob("start_*.npz"))
    z = [np.load(f, allow_pickle=False) for f in files]
    names = [str(x) for x in z[0]["names"]]
    base = np.stack([x["base"] for x in z])
    SC = np.stack([x["scores"] for x in z])
    ST = np.stack([x["stolen"] for x in z])
    cum = np.concatenate([np.zeros(ST.shape[:-1] + (1,), np.float32), np.cumsum(ST, axis=-1)], axis=-1)
    laws = [str(x) for x in z[0]["laws"]]
    out = {"starts": len(files), "draws": int(SC.shape[1]), "kappas": KAPPAS.tolist(), "detectors": {}}
    for ni, dn in enumerate(names):
        nb = base[:, ni]
        if np.isnan(nb).all():
            continue
        fa = [float((first_passage(nb, k) < HORIZON).mean()) for k in KAPPAS]
        D = {"natural_rate": fa, "laws": {}}
        for li, law in enumerate(laws):
            D["laws"][law] = {}
            for mi, m in enumerate(INTENSITIES):
                S = SC[:, :, li, mi, ni]
                det, loss = [], []
                for k in KAPPAS:
                    fp = first_passage(S, k)
                    det.append(float((fp < HORIZON).mean()))

                    lv = np.take_along_axis(cum[:, :, li, mi], np.minimum(fp + 1, HORIZON)[..., None], -1)[..., 0]
                    loss.append(float(np.median(lv)))
                D["laws"][law][str(m)] = {"detection": det, "median_loss_btc": loss}
        out["detectors"][dn] = D

    matched = {}
    for target in FA_TARGETS:
        matched[str(target)] = {}
        for dn, D in out["detectors"].items():
            ok = [i for i, f in enumerate(D["natural_rate"]) if f <= target]
            if not ok:
                continue
            i = ok[0]
            matched[str(target)][dn] = {
                "kappa": float(KAPPAS[i]), "natural_rate": D["natural_rate"][i],
                "laws": {law: {m: {"detection": v["detection"][i], "excess": v["detection"][i] - D["natural_rate"][i],
                                   "median_loss_btc": v["median_loss_btc"][i]}
                               for m, v in L.items()} for law, L in D["laws"].items()}}
    out["matched"] = matched
    path = C.save_json(f"replay_curves_{args.tag}", out)
    print("saved", path)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("prep")
    a.add_argument("--results", required=True)
    a.add_argument("--out", required=True)
    a.add_argument("--cache", default=None, help="validated setup cache, to use the exact calibration panel")
    b = sub.add_parser("run")
    b.add_argument("--results", required=True)
    b.add_argument("--pool", required=True)
    b.add_argument("--tag", default="addr2")
    b.add_argument("--k", type=int, default=64000)
    b.add_argument("--workers", type=int, default=1)
    b.add_argument("--only", type=int, nargs="*", default=None)
    b.add_argument("--setup-only", action="store_true")
    c = sub.add_parser("curves")
    c.add_argument("--pool", required=True)
    c.add_argument("--tag", default="addr2")
    args = ap.parse_args()
    return {"prep": prep, "run": run, "curves": curves}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
