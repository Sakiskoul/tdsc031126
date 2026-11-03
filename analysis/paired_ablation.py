from __future__ import annotations
import os
for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'
import argparse
import json
import sys
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from types import SimpleNamespace
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / 'results/generated/paired_mechanisms'
HERE.mkdir(parents=True, exist_ok=True)
INPUT = ROOT / 'data/paired_mechanisms'
sys.path.insert(0, str(ROOT / 'code/experiments'))
import common as C
import replay as R
import data_io as IO
STATE = {}
COLS = ['n', 'sv', 'sv2', 'e_n', 'e_sv', 'e_sv2']
DCOLS = ['dn', 'dsv', 'dsv2', 'de_n', 'de_sv', 'de_sv2']

def components_to_delta(frame, case, mechanism):
    f = frame.copy()
    f = f[(f.component == 'add') | (f.component == 'cancel') & (mechanism != 'add_only') | (f.component == 'risk') & (mechanism in ('risk', 'refit'))]
    f['pop'] = 'exposed'
    f['wk'] = pd.Timestamp(case['t0']) + pd.to_timedelta(7 * f.week, unit='D')
    for col in DCOLS:
        f[col] = 0.0
    for (tag, sign) in [('add', 1.0), ('cancel', -1.0)]:
        sel = f.component == tag
        for (src, dst) in [('n', 'de_n'), ('sv', 'de_sv'), ('sv2', 'de_sv2')]:
            f.loc[sel, dst] = sign * f.loc[sel, src]
    sel = f.component == 'risk'
    for (src, dst) in [('n', 'dn'), ('sv', 'dsv'), ('sv2', 'dsv2')]:
        f.loc[sel, dst] = -f.loc[sel, src]
    return f.groupby(C.KEY + ['wk'], as_index=False)[DCOLS].sum()

def prepare():
    protocol = json.loads((ROOT / 'config/paired_mechanisms.json').read_text())
    addr = pd.concat([pd.read_csv(p, float_precision='round_trip') for p in sorted((INPUT / 'address_components').glob('part_*.csv'))], ignore_index=True)
    out = pd.concat([pd.read_csv(p, float_precision='round_trip') for p in sorted((INPUT / 'output_components').glob('part_*.csv'))], ignore_index=True)
    av = addr.groupby(['sid', 'component', 'week']).sv.sum()
    ov = out.groupby(['sid', 'component', 'week']).sv.sum()
    for component in ['add', 'risk']:
        x = av.xs(component, level='component').sort_index()
        y = ov.xs(component, level='component').reindex(x.index)
        assert np.allclose(x, y, rtol=1e-11, atol=1e-06), (component, float((x - y).abs().max()))
    stolen = addr[addr.component == 'add'].groupby(['sid', 'week']).sv.sum()
    thresholds = json.loads((ROOT / 'config/fitted_thresholds.json').read_text())
    result = {'protocol': protocol, 'components': {'output': out, 'address': addr}, 'stolen': stolen, 'thresholds': thresholds, 'representations': {}}
    for (representation, tag) in [('output', 'full'), ('address', 'addr')]:
        t = time.time()
        panel = C.load_panel(ROOT / 'data' / ('results_' + tag), R.T_END)
        weekly = pd.read_csv(ROOT / 'results' / ('weekly_' + tag + '.csv'), parse_dates=['wk'], index_col='wk', float_precision='round_trip')
        anchors = sorted(set((R.refit_anchor(pd.Timestamp(c['t0']) + pd.Timedelta(weeks=w)) for c in protocol['case_parameters'] for w in range(52))))
        models = {}
        baseline = {}
        details = {}
        for anchor in anchors:
            m = C.NullModel().fit(R.fit_window(panel, anchor))
            models[anchor] = m
            weeks = pd.date_range(anchor, periods=13, freq='W-MON').intersection(weekly.index)
            cw = C.weekly_stats(panel, m, True).reindex(weeks)
            uw = C.weekly_stats(panel, m, False).reindex(weeks)
            th = C.control_theta(panel, m).reindex(weeks)
            baseline[anchor] = (cw, uw)
            details[anchor] = (th, m.lam)
            for w in weeks:
                assert np.isclose(cw.loc[w, 'S'], weekly.loc[w, 'S'], rtol=1e-09, atol=0.0001)
                assert np.isclose(cw.loc[w, 'var'], weekly.loc[w, 'var'], rtol=1e-09, atol=0.001)
        result['representations'][representation] = {'panel': panel, 'weekly': weekly, 'models': models, 'baseline': baseline, 'details': details, 'tag': tag}
        print('prepared', representation, len(anchors), 'anchors', round(time.time() - t, 1), 'seconds', flush=True)
    pd.to_pickle(result, HERE / 'paired_cache.pkl')
    return result

def init_worker(cache):
    global STATE
    STATE = pd.read_pickle(cache)

def frozen_stats(rep, delta, win, calibrated):
    d = STATE['representations'][rep]
    pieces = []
    for anchor in sorted(set((R.refit_anchor(w) for w in win))):
        b = win[[R.refit_anchor(w) == anchor for w in win]]
        w = d['baseline'][anchor][0 if calibrated else 1].reindex(b).copy()
        df = delta[delta.wk.isin(b)].copy()
        (th, lam) = d['details'][anchor]
        df['h'] = df[C.KEY].merge(lam.rename('h').reset_index(), on=C.KEY, how='left').h.fillna(0.0).to_numpy()
        theta = df.wk.map(th.theta).fillna(1.0) if calibrated else 1.0
        df['p'] = np.minimum(theta * df.h, 0.25)
        df['expected'] = df.p * df.dsv
        df['bin'] = df.p * (1 - df.p) * df.dsv2
        df['hf'] = df.h * df.dsv
        a = df.groupby('wk')[['de_sv', 'expected', 'bin', 'hf']].sum().reindex(b).fillna(0.0)
        w['S'] += a.de_sv - a.expected
        w['var'] += a.bin + w.var_theta * ((w.hF + a.hf) ** 2 - w.hF ** 2)
        pieces.append(w)
    return pd.concat(pieces).reindex(win)

def strict_contaminate(panel, delta):
    p = panel.merge(delta, on=C.KEY + ['wk'], how='left', validate='one_to_one')
    assert int(p.dn.notna().sum()) == len(delta), 'Unmatched delta rows'
    for (c, dc) in zip(COLS, DCOLS):
        values = p[c] + p[dc].fillna(0.0)
        bad = values < -np.maximum(1e-05, 1e-10 * p[c].abs())
        assert not bad.any(), (c, int(bad.sum()), float(values.min()))
        p[c] = values.clip(lower=0.0)
    return p[panel.columns]

def scores(z, hist, params, accumulator):
    if accumulator == 'ARGUS':
        return C.RobustDetector(**params).scores(z)
    c = C.ConformalDetector()
    return c.scores(c.pvalues(hist, z))

def one_case(case):
    sid = case['sid']
    win = pd.date_range(case['t0'], periods=52, freq='W-MON')
    stolen = STATE['stolen'].loc[sid].reindex(range(52), fill_value=0.0).to_numpy()
    cum = np.cumsum(stolen)
    rows = []
    series = []
    checks = []
    for rep in ['output', 'address']:
        data = STATE['representations'][rep]
        frame = STATE['components'][rep]
        frame = frame[frame.sid == sid]
        parameters = STATE['thresholds'][data['tag']]
        replayer = None
        for mechanism in ['add_only', 'cancel', 'risk', 'refit']:
            delta = components_to_delta(frame, case, mechanism)
            if mechanism == 'refit' or sid == 0:
                contaminated = strict_contaminate(data['panel'], delta)
            if mechanism == 'refit':
                light = SimpleNamespace(panel=data['panel'], _replay_null_models=data['models'])
                replayer = R.Replayer(light, win[0])
                (wc, wu, _) = replayer.stats(contaminated, band=False)
            else:
                wc = frozen_stats(rep, delta, win, True)
                wu = frozen_stats(rep, delta, win, False)
                if sid == 0:
                    light = SimpleNamespace(panel=data['panel'], _replay_null_models=data['models'])
                    rr = R.Replayer(light, win[0])
                    (cw, uw, _) = rr.stats(contaminated, frozen=True, band=False)
                    for (x, y) in [(wc, cw), (wu, uw)]:
                        assert np.allclose(x.S, y.S, rtol=1e-09, atol=0.0001)
                        assert np.allclose(x['var'], y['var'], rtol=1e-09, atol=0.001)
            for (control, w) in [('control', wc), ('no_control', wu)]:
                params = parameters['argus' if control == 'control' else 'uncalibrated']
                z = w.S.to_numpy() / np.sqrt(params['phi'] * w['var'].clip(lower=1e-12).to_numpy())
                hz = data['weekly']['z_argus' if control == 'control' else 'z_uncalibrated']
                hist = hz[(hz.index >= R.TRAIN[0]) & (hz.index < win[0])].to_numpy()
                for acc in ['ARGUS', 'CTM']:
                    score = scores(z, hist, params, acc)
                    hit = score > 1 if acc == 'ARGUS' else score >= 1
                    k = int(np.argmax(hit)) if hit.any() else 51
                    baseline = data['weekly'].loc[win]
                    z0 = baseline['z_argus' if control == 'control' else 'z_uncalibrated'].to_numpy()
                    sc0 = scores(z0, hist, params, acc)
                    natural = int((sc0 > 1).any() if acc == 'ARGUS' else (sc0 >= 1).any())
                    rows.append({'sid': sid, 'representation': rep, 'control': control, 'accumulator': acc, 'mechanism': mechanism, 'natural': natural, 'alarm': int(hit.any()), 'delay': k + 1 if hit.any() else None, 'loss_btc': float(cum[k]), 'total_btc': float(cum[-1])})
                    series.extend(({'sid': sid, 'representation': rep, 'control': control, 'accumulator': acc, 'mechanism': mechanism, 'week': j, 'score': float(score[j]), 'natural_score': float(sc0[j]), 'stolen_btc': float(stolen[j])} for j in range(52)))
    return (rows, series)

def summarise():
    cases = sorted((HERE / 'paired_runs').glob('start_*.csv'))
    assert len(cases) == 76
    outcomes = pd.concat([pd.read_csv(p, float_precision='round_trip') for p in cases], ignore_index=True)
    assert len(outcomes) == 2432
    outcomes.to_csv(HERE / 'paired_outcomes.csv', index=False, float_format='%.17g')
    original = IO.replay_report()
    gate = []
    for c in original['per_start']:
        r = next((r for r in c['runs'] if r['draw'] == 0 and r['law'] == 'tilted' and (r['m'] == 1.0)))
        for (control, acc, name) in [('control', 'ARGUS', 'argus'), ('no_control', 'ARGUS', 'uncalibrated'), ('control', 'CTM', 'ctm')]:
            row = outcomes[(outcomes.sid == c['sid']) & (outcomes.representation == 'address') & (outcomes.control == control) & (outcomes.accumulator == acc) & (outcomes.mechanism == 'refit')].iloc[0]
            e = r['det'][name]
            assert row.alarm == e['alarm']
            assert pd.isna(row.delay) and e['delay'] is None or row.delay == e['delay']
            assert np.isclose(row.loss_btc, e['loss_btc'], rtol=1e-11, atol=1e-06)
            gate.append({'sid': c['sid'], 'detector': name, 'alarm': int(row.alarm), 'delay': row.delay, 'loss_error_btc': float(row.loss_btc - e['loss_btc'])})
    pd.DataFrame(gate).to_csv(HERE / 'paired_original_check.csv', index=False)
    keys = ['representation', 'control', 'accumulator', 'mechanism']
    summary = outcomes.groupby(keys).agg(n=('sid', 'size'), natural=('natural', 'mean'), detection=('alarm', 'mean'), mean_loss_btc=('loss_btc', 'mean'), median_loss_btc=('loss_btc', 'median')).reset_index()
    summary.to_csv(HERE / 'paired_summary.csv', index=False, float_format='%.17g')
    rng = np.random.default_rng(2026110309)
    ix = np.concatenate([(rng.integers(0, 76, (3000, 1)) + np.arange(13)) % 76 for _ in range(6)], axis=1)[:, :76]
    piv = outcomes.pivot(index='sid', columns=keys, values='loss_btc').sort_index()
    pairs = []
    for rep in ['output', 'address']:
        for ctl in ['control', 'no_control']:
            for acc in ['ARGUS', 'CTM']:
                for (before, after) in [('add_only', 'cancel'), ('cancel', 'risk'), ('risk', 'refit')]:
                    diff = (piv[rep, ctl, acc, after] - piv[rep, ctl, acc, before]).to_numpy()
                    boot = diff[ix].mean(axis=1)
                    pairs.append({'representation': rep, 'control': ctl, 'accumulator': acc, 'comparison': after + ' minus ' + before, 'mean_difference_btc': float(diff.mean()), 'ci_low_btc': float(np.quantile(boot, 0.025)), 'ci_high_btc': float(np.quantile(boot, 0.975))})
    for ctl in ['control', 'no_control']:
        for acc in ['ARGUS', 'CTM']:
            for mech in ['add_only', 'cancel', 'risk', 'refit']:
                diff = (piv['address', ctl, acc, mech] - piv['output', ctl, acc, mech]).to_numpy()
                boot = diff[ix].mean(axis=1)
                pairs.append({'representation': 'address minus output', 'control': ctl, 'accumulator': acc, 'comparison': mech, 'mean_difference_btc': float(diff.mean()), 'ci_low_btc': float(np.quantile(boot, 0.025)), 'ci_high_btc': float(np.quantile(boot, 0.975))})
    for rep in ['output', 'address']:
        for acc in ['ARGUS', 'CTM']:
            for mech in ['add_only', 'cancel', 'risk', 'refit']:
                diff = (piv[rep, 'control', acc, mech] - piv[rep, 'no_control', acc, mech]).to_numpy()
                boot = diff[ix].mean(axis=1)
                pairs.append({'representation': rep, 'control': 'control minus no_control', 'accumulator': acc, 'comparison': mech, 'mean_difference_btc': float(diff.mean()), 'ci_low_btc': float(np.quantile(boot, 0.025)), 'ci_high_btc': float(np.quantile(boot, 0.975))})
    pd.DataFrame(pairs).to_csv(HERE / 'paired_differences.csv', index=False, float_format='%.17g')
    print(summary.to_string(index=False))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=3)
    ap.add_argument('--limit', type=int, default=76)
    ap.add_argument('--summarise-only', action='store_true')
    args = ap.parse_args()
    if args.summarise_only:
        summarise()
        return
    cache = HERE / 'paired_cache.pkl'
    if not cache.exists():
        prepare()
    protocol = json.loads((ROOT / 'config/paired_mechanisms.json').read_text())
    out = HERE / 'paired_runs'
    out.mkdir(exist_ok=True)
    pending = [c for c in protocol['case_parameters'][:args.limit] if not (out / ('start_%03d.csv' % c['sid'])).exists()]
    with ProcessPoolExecutor(max_workers=args.workers, initializer=init_worker, initargs=(cache,)) as pool:
        futures = {pool.submit(one_case, c): c['sid'] for c in pending}
        for future in as_completed(futures):
            sid = futures[future]
            (rows, series) = future.result()
            pd.DataFrame(rows).to_csv(out / ('start_%03d.csv' % sid), index=False, float_format='%.17g')
            pd.DataFrame(series).to_csv(out / ('scores_%03d.csv' % sid), index=False, float_format='%.17g')
            print('completed', sid, flush=True)
    if args.limit == 76:
        summarise()
if __name__ == '__main__':
    main()
