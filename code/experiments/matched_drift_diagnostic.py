from __future__ import annotations

import os
for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[name] = '1'
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import data_io as IO
import common as C

ROOT = Path(__file__).resolve().parents[2]
TRAIN = ('2014-01-06', '2019-12-30')
EVAL = ('2020-01-06', '2026-09-28')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    panel_path = ROOT/'data/exploratory/grouped/panel_full.parquet'
    panel = IO.read_frame(panel_path)
    fit = panel[(panel.wk >= TRAIN[0]) & (panel.wk < TRAIN[1])]
    m = C.NullModel().fit(fit)
    def totals(pop):
        x = panel[(panel['pop'] == pop) & (panel.n > 0)].copy()
        x['baseline_n'] = m.hazard(x)*x.n
        return x.groupby('wk')[['e_n', 'baseline_n']].sum().sort_index().rolling(13).sum()
    ex, ctl = totals('exposed'), totals('control')
    a, c = ex.e_n/ex.baseline_n, ctl.e_n/ctl.baseline_n
    diag = pd.DataFrame({'theta_exposed13': a, 'theta_control13': c})
    diag['control_relative'] = a/c-1.
    diag['no_control_relative'] = a-1.
    diag['control_absolute'] = a-c
    diag = diag.replace([np.inf, -np.inf], np.nan)
    q = {key: float(diag.loc[TRAIN[0]:TRAIN[1], key].abs().quantile(.9))
         for key in ('control_relative', 'no_control_relative', 'control_absolute')}
    reference = pd.read_csv(ROOT/'results/weekly_full.csv', parse_dates=['wk'], index_col='wk')
    expected = float(IO.read_csv(ROOT/'config/reference_checks.csv').set_index('quantity').loc['control_2B_btc_week','value'])
    got = float((2*q['control_relative']*reference.loc[EVAL[0]:EVAL[1], 'exp_v']).median())
    if not np.isclose(got, expected, rtol=1e-12):
        raise ValueError('published 2B diagnostic does not reproduce')


    weeks = reference.loc[EVAL[0]:EVAL[1]].index
    rows = []
    for i in range(0, len(weeks), 13):
        block = weeks[i:i+13]
        hi = block[0]-pd.Timedelta(weeks=13)
        lo = hi-pd.Timedelta(weeks=104)
        f = panel[(panel.wk >= lo) & (panel.wk < hi)]
        model = C.NullModel().fit(f)
        base = C.weekly_stats(panel, model, calibrated=False).loc[block]
        check = C.weekly_stats(panel, model, calibrated=True).loc[block]
        if not np.allclose(check.exp_v, reference.loc[block, 'exp_v'], rtol=1e-10, atol=1e-6):
            raise ValueError('calendar control-flow reconstruction differs')
        rows.append(base[['exp_v', 'hF']])
        print('matched flow block', str(block[0].date()), flush=True)
    unc = pd.concat(rows).rename(columns={'exp_v': 'flow_no_control', 'hF': 'flow_base_uncapped'})
    flow = reference.loc[weeks, ['exp_v']].rename(columns={'exp_v': 'flow_control'}).join(unc)
    for key in ('control_relative', 'no_control_relative', 'control_absolute'):
        flow['twice_'+key+'_common_flow'] = 2*q[key]*flow.flow_control
    flow['B_control'] = q['control_relative']*flow.flow_control
    flow['B_no_control'] = q['no_control_relative']*flow.flow_no_control
    result = {
        'panel': 'output', 'training': TRAIN, 'evaluation': EVAL,
        'diagnostic': 'absolute 13-week count-rate residual; training q90, then median evaluation BTC conversion',
        'model': 'both 13-week regime estimates use same initial 2014-2019 fitted hazards; evaluation flows use original rolling fits',
        'q90_training': q,
        'method_specific_reference_flow': {
            'control_B_btc_week': float(flow.B_control.median()),
            'control_2B_btc_week': got,
            'no_control_B_btc_week': float(flow.B_no_control.median()),
            'no_control_2B_btc_week': float((2*flow.B_no_control).median()),
            'control_flow_btc_week': float(flow.flow_control.median()),
            'no_control_flow_btc_week': float(flow.flow_no_control.median())},
        'common_control_flow_conversion': {
            'control_relative_2B_btc_week': got,
            'no_control_relative_2B_btc_week': float(flow.twice_no_control_relative_common_flow.median()),
            'note': 'common BTC normalization only; not the no-control theorem floor because theta=1 predicts a different flow'},
        'common_base_absolute_residual_conversion': {
            'control_2B_btc_week': float((2*q['control_absolute']*flow.flow_no_control).median()),
            'no_control_2B_btc_week': float((2*q['no_control_relative']*flow.flow_no_control).median()),
            'note': 'same absolute regime-error diagnostic and same baseline flow; distinct from the original relative-drift floor'},
        'reference_2B_reproduced': True,
        'probability_cap_flow_max_abs_difference_btc': float((flow.flow_no_control-flow.flow_base_uncapped).abs().max()),
        'limitations': 'empirical quantiles are not latent uniform drift bounds; method-specific flows differ; do not infer a theorem-level ranking from these diagnostics',
    }
    diag.to_csv(out/'matched_drift_weekly.csv', index_label='wk')
    flow.to_csv(out/'matched_flow_weekly.csv', index_label='wk')
    (out/'matched_drift.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
