import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'code/experiments'))

import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import data_io as IO
import audit_replay_outputs as A

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'data/event_replay'
LAWS = ['tilted', 'mimic', 'uniform', 'whale']
INTENSITIES = [.25, .5, 1., 2., 4.]


def main():
    full = IO.replay_report()
    curves = IO.replay_curves()
    audit = A.audit(OUT/'runs',76)
    assert audit['passed'] and audit['starts'] == 76 and audit['attack_cases'] == 12160
    starts = full['per_start']
    assert len(starts) == 76 and {s['sid'] for s in starts} == set(range(76))
    assert full['summary']['starts'] == curves['starts'] == 76 and curves['draws'] == 8
    arrays = []
    for sid in range(76):
        with IO.replay_arrays(OUT/'runs'/f'start_{sid:03d}.npz') as d:
            a = {key: d[key] for key in d.files}
        assert list(a['draws']) == list(range(8))
        assert list(a['laws']) == LAWS and list(a['intensities']) == INTENSITIES
        assert not np.isnan(a['scores']).any() and not np.isnan(a['base']).any()
        assert not np.isneginf(a['scores']).any() and not np.isneginf(a['base']).any()
        assert np.isfinite(a['stolen']).all() and (a['stolen'] >= 0).all()
        arrays.append(a)
    names = list(map(str, arrays[0]['names']))
    assert all(list(map(str, a['names'])) == names for a in arrays)
    scores = np.stack([a['scores'] for a in arrays])
    base = np.stack([a['base'] for a in arrays])
    stolen = np.stack([a['stolen'] for a in arrays]).astype(np.float64)
    cum = np.concatenate([np.zeros(stolen.shape[:-1]+(1,)), stolen.cumsum(axis=-1)], axis=-1)
    max_loss_rounding = 0.
    rows = []
    for ni, name in enumerate(names):
        compare = np.greater_equal if name in ('eprocess', 'ctm', 'threshold') else np.greater
        untouched = np.array([s['natural'][name] for s in starts])
        assert np.isclose(untouched.mean(), full['summary']['natural_window_alarm_rate'][name])
        fa = []
        for k in curves['kappas']:
            fa.append(compare(base[:, ni], k).any(axis=-1).mean())
        assert np.allclose(fa, curves['detectors'][name]['natural_rate'], rtol=0, atol=1e-12)
        assert (np.diff(fa) <= 1e-12).all()
        for li, law in enumerate(LAWS):
            for mi, m in enumerate(INTENSITIES):
                flat = [run for s in starts for run in s['runs'] if run['law'] == law and run['m'] == m]
                assert len(flat) == 608
                reported = full['summary']['laws'][law][str(m)]['det'][name]
                detection = np.mean([r['det'][name]['alarm'] for r in flat])
                loss = np.median([r['det'][name]['loss_btc'] for r in flat])
                assert np.isclose(detection, reported['detection_rate'], atol=1e-12)
                assert np.isclose(loss, reported['median_loss_btc'], rtol=1e-12)
                assert np.isclose(detection-untouched.mean(), reported['excess'], atol=1e-12)
                cv = curves['detectors'][name]['laws'][law][str(m)]
                rec_det, rec_loss = [], []
                s = scores[:, :, li, mi, ni]
                for k in curves['kappas']:
                    hits = compare(s, k)
                    first = np.where(hits.any(axis=-1), hits.argmax(axis=-1)+1, 52)
                    values = np.take_along_axis(cum[:, :, li, mi], first[..., None], axis=-1)[..., 0]
                    rec_det.append(float(hits.any(axis=-1).mean()))
                    rec_loss.append(float(np.median(values)))
                assert np.allclose(rec_det, cv['detection'], rtol=0, atol=1e-12)
                delta = float(np.max(np.abs(np.array(rec_loss)-cv['median_loss_btc'])))
                max_loss_rounding = max(max_loss_rounding, delta)
                assert np.allclose(rec_loss, cv['median_loss_btc'], rtol=5e-7, atol=.5)
                assert (np.diff(rec_det) <= 1e-12).all()
                assert (np.diff(rec_loss) >= -1e-6).all()
                rows.append({'law': law, 'm': m, 'detector': name,
                    'natural_rate': float(untouched.mean()), 'detection': detection,
                    'excess': reported['excess'], 'excess_low': reported['excess_ci95'][0],
                    'excess_high': reported['excess_ci95'][1], 'draw_sd': reported['excess_draw_sd'],
                    'median_delay': reported['median_delay'], 'median_loss_btc': loss,
                    'p90_loss_btc': reported['p90_loss_btc'],
                    'excess_first_draw': reported['excess_first_draw'],
                    'excess_frozen': reported['excess_frozen']})
    generated = ROOT/'results/generated'
    generated.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_csv(generated/'nominal_results.csv', index=False)
    result = {'passed': True, 'starts': 76, 'attack_cases': 12160, 'draws': 8,
              'all_summary_rates_and_median_losses_reproduced': True,
              'all_41_threshold_curves_reproduced_with_detector_specific_inequalities': True,
              'max_curve_loss_float32_rounding_btc': max_loss_rounding,
              'positive_infinite_curve_entries': int(np.isposinf(scores).sum()+np.isposinf(base).sum()),
              'recorded_no_attack_reconstruction_residuals_checked': True,
              'ctm_minimum_natural_rate_in_grid': min(curves['detectors']['ctm']['natural_rate']),
              'eprocess_minimum_natural_rate_in_grid': min(curves['detectors']['eprocess']['natural_rate']),
              'matched_unavailable_note': 'CTM and ARGUS-e do not attain any 5/10/20% natural-window target in the prespecified kappa grid; do not extrapolate'}
    pd.DataFrame([{'check':k,'value':v} for k,v in result.items()]).to_csv(generated/'replay_checks.csv',index=False)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
