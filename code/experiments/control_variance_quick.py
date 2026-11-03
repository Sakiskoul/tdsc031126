from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import group_regime as G
import common as C
import rq56
import numpy as np
import pandas as pd


def control_innovation(panel, model):
    c = panel[(panel['pop'] == 'control') & (panel.n > 0)].copy()
    h = model.hazard(c)
    c['L'] = h * c.n
    c['F'] = h * c.sv
    c['Q'] = h * c.sv2
    w = c.groupby('wk').agg(L=('L', 'sum'), F=('F', 'sum'), Q=('Q', 'sum'),
                            W=('e_sv', 'sum'))
    th = C.control_theta(panel, model).reindex(w.index)
    a = w.F / (w.L + 20.)
    w['U'] = w.W - th.theta * w.F

    w['V'] = (th.theta * (w.Q - 2*a*w.F + a*a*w.L)).clip(lower=1e-12)
    w['ratio'] = w.U.pow(2).rolling(13, min_periods=13).sum() / w.V.rolling(13, min_periods=13).sum()
    return w


def feature_frame(panel, out, name):
    path = out / f'control_innovation_{name}.csv'
    if path.exists():
        return pd.read_csv(path, parse_dates=['wk'], index_col='wk')
    m = C.NullModel().fit(G.subset(panel, G.TRAIN))
    static = control_innovation(panel, m)
    parts = [static[static.index < G.EVAL[0]]]
    weeks = static.index[static.index >= G.EVAL[0]]
    for i in range(0, len(weeks), 13):
        block = weeks[i:i+13]
        hi = block[0] - pd.Timedelta(weeks=13)
        fit = (hi-pd.Timedelta(weeks=104), hi)
        mm = C.NullModel().fit(G.subset(panel, fit))
        parts.append(control_innovation(panel, mm).loc[block])
        print(name, 'control variance block', str(block[0].date()), flush=True)
    result = pd.concat(parts).sort_index()
    result.to_csv(path)
    return result


def fit_multiplier(controls):
    normalizer = float(controls.loc[G.TRAIN[0]:G.TRAIN[1], 'ratio'].median())
    if not np.isfinite(normalizer) or normalizer <= 0:
        raise ValueError('invalid training normalization')
    multiplier = (controls.ratio / normalizer).fillna(1.).clip(lower=1., upper=4.)
    return multiplier, normalizer


def evaluate(name, out):
    root = G.ROOT
    panel = G.load_numeric(name, out.parent / 'group_regime')
    ctl = feature_frame(panel, out, name)
    multiplier, normalizer = fit_multiplier(ctl)
    w = pd.read_csv(root/'results'/f'weekly_{name}.csv', parse_dates=['wk'], index_col='wk')
    phi = C.estimate_phi(w.loc[G.TRAIN[0]:G.TRAIN[1]])
    sigma = np.sqrt(phi*w['var'].clip(lower=1e-12))
    weeks = w.loc[G.EVAL[0]:G.EVAL[1]].index
    starts = weeks[:len(weeks)-52:4]
    models = {}
    for variant in ('scalar', 'dynamic'):
        scale = sigma if variant == 'scalar' else sigma * np.sqrt(multiplier.reindex(w.index))
        z = w.S / scale
        arg = C.calibrate(C.RobustDetector(phi=phi), z.loc[G.TRAIN[0]:G.TRAIN[1]].values,
                          104., np.random.default_rng(20261016))
        for kind, det in [('argus', arg), ('ctm', C.ConformalDetector(c=104.))]:
            zz = z.loc[G.TRAIN[0]:]
            alarms = det.run(det.pvalues(np.array([]), zz.values)) if kind == 'ctm' else det.run(zz.values)
            series = pd.Series(alarms, index=zz.index).loc[G.EVAL[0]:G.EVAL[1]]
            key = variant + '_' + kind
            models[key] = {'det': det, 'kind': kind, 'z': z, 'scale': scale,
                           'historical': {'count': int(series.sum()),
                                          'weeks': [str(x.date()) for x in series.index[series]],
                                          'h': getattr(det, 'h', None),
                                          'z_max': getattr(det, 'z_max', None)}}
    reference_weekly=pd.read_csv(root/'results'/f'weekly_{name}.csv')
    reference={'rq3':{kind:{'alarm_weeks':reference_weekly.loc[(reference_weekly.wk>=G.EVAL[0]) & (reference_weekly.wk<=G.EVAL[1]) & reference_weekly['alarm_'+kind],'wk'].tolist()} for kind in ('argus','ctm')}}
    for kind in ('argus', 'ctm'):
        if models['scalar_'+kind]['historical']['weeks'] != reference['rq3'][kind]['alarm_weeks']:
            raise ValueError('reference historical alarms do not reproduce')
    raw = {key: {'base': [], 'attacks': {str(m): [] for m in (.25, 1.)}} for key in models}
    uv = w.energy / w.exp_v.clip(lower=1e-12)
    for t0 in starts:
        win = weeks[weeks >= t0][:52]
        plans = {str(m): rq56.feasible_theft((m*sigma.loc[win]).values, uv.loc[win].values,
                                           1000., float(w.loc[t0, 'stock_v']))[0] for m in (.25, 1.)}
        for key, model in models.items():
            det, z, scale = model['det'], model['z'], model['scale']
            hist = z[(z.index >= G.TRAIN[0]) & (z.index < t0)].values
            def alarm(add):
                zz = (w.loc[win, 'S'].values + add) / scale.loc[win].values
                return det.run(det.pvalues(hist, zz)) if model['kind'] == 'ctm' else det.run(zz)
            raw[key]['base'].append(int(alarm(np.zeros(52)).any()))
            for intensity, plan in plans.items():
                hit = alarm(plan)
                delay = int(hit.argmax())+1 if hit.any() else None
                raw[key]['attacks'][intensity].append({'loss': float(plan[:delay].sum()) if delay else float(plan.sum()),
                                                      'alarm': int(hit.any()), 'delay': delay})
    boot = rq56.block_indices(len(starts), 13, 2000, np.random.default_rng(20261102))
    summary = {}
    for key, model in models.items():
        base = float(np.mean(raw[key]['base']))
        summary[key] = {'historical': model['historical'], 'natural_window_rate': base, 'attacks': {}}
        for intensity, rows in raw[key]['attacks'].items():
            delays = [r['delay'] for r in rows if r['delay'] is not None]
            summary[key]['attacks'][intensity] = {
                'median_loss_btc': float(np.median([r['loss'] for r in rows])),
                'detection': float(np.mean([r['alarm'] for r in rows])),
                'excess': float(np.mean([r['alarm'] for r in rows]))-base,
                'median_delay': float(np.median(delays)) if delays else None}
    paired = {}
    for ctm in ('scalar_ctm', 'dynamic_ctm'):
        paired[ctm] = {}
        for intensity in ('0.25', '1.0'):
            a = np.array([r['loss'] for r in raw['dynamic_argus']['attacks'][intensity]])
            b = np.array([r['loss'] for r in raw[ctm]['attacks'][intensity]])
            ci = np.quantile(np.median(a[boot], axis=1)-np.median(b[boot], axis=1), [.025, .975])
            paired[ctm][intensity] = {
                'difference_of_medians_btc': float(np.median(a)-np.median(b)), 'paired_ci95': ci.tolist(),
                'passes_criterion': bool(summary['dynamic_argus']['historical']['count'] <= summary[ctm]['historical']['count']
                and summary['dynamic_argus']['natural_window_rate'] <= summary[ctm]['natural_window_rate'] and ci[1] < 0)}
    result = {'panel': name, 'reference_historical_validation': True, 'starts': len(starts),
              'training_control_normalizer': normalizer,
              'multiplier_eval': multiplier.loc[G.EVAL[0]:G.EVAL[1]].describe().to_dict(),
              'summary': summary, 'paired_loss': paired,
              'scope': 'exploratory aggregate injections, fixed BTC attacks; no event-level refits of this variant'}
    (out/f'evaluation_{name}.json').write_text(json.dumps(result, indent=2))
    (out/f'windows_{name}.json').write_text(json.dumps({'starts': [str(x.date()) for x in starts], 'raw': raw}, indent=2))
    multiplier.rename('variance_multiplier').to_csv(out/f'multiplier_{name}.csv')
    print(json.dumps(result, indent=2), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--panel', choices=('full', 'addr'), required=True)
    args = ap.parse_args()
    out = Path(args.out)
    protocol = json.loads((out/'protocol.json').read_text())
    script_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if protocol['published_script_sha256'] != script_hash:
        raise ValueError('script changed after protocol freeze')
    evaluate(args.panel, out)


if __name__ == '__main__':
    main()
