import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'code/experiments'))
from pathlib import Path
import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import sys
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
sys.path.insert(0,str(ROOT/'code/experiments'))
import numpy as np
import pandas as pd
import data_io as IO
import common as C
import rq
import rq56
import json

OUT=ROOT/'results/generated';OUT.mkdir(exist_ok=True,parents=True)
TRAIN=('2014-01-06','2019-12-30');EVAL=('2020-01-06','2026-09-28')


def statistics(panel):
    initial=C.NullModel().fit(panel[(panel.wk>=TRAIN[0])&(panel.wk<TRAIN[1])])
    raw=C.weekly_stats(panel,initial)
    weeks=raw.index[raw.index>=EVAL[0]];parts=[raw[raw.index<EVAL[0]]]
    for n in range(0,len(weeks),13):
        win=weeks[n:n+13];hi=win[0]-pd.Timedelta(weeks=13);lo=hi-pd.Timedelta(weeks=104)
        model=C.NullModel().fit(panel[(panel.wk>=lo)&(panel.wk<hi)])
        parts.append(C.weekly_stats(panel,model).reindex(win))
    w=pd.concat(parts).sort_index();tr=w.loc[TRAIN[0]:TRAIN[1]]
    phi=C.estimate_phi(tr);det=C.calibrate(C.RobustDetector(phi=phi),
        (tr.S/np.sqrt(phi*tr['var'].clip(lower=1e-12))).values,104.,np.random.default_rng(20261016))
    return w,phi,det


def main():
    ref=C.load_panel(ROOT/'data/results_full','2026-10-01')
    specs={'reference':ref,
           'recognised_removed':IO.read_frame(ROOT/'data/screened/recognised/screened_panel.parquet'),
           'simple_screened':IO.read_frame(ROOT/'data/screened/restricted/screened_panel.parquet')}
    stats={};stock=[]
    for name,panel in specs.items():
        pd.testing.assert_frame_equal(ref[ref['pop']=='exposed'].reset_index(drop=True),
                                      panel[panel['pop']=='exposed'].reset_index(drop=True))
        print('fitting',name,len(panel),flush=True)
        w,phi,det=statistics(panel);stats[name]=(w,phi,det)
        c=panel[panel['pop']=='control'].groupby('wk').agg(stock=('sv','sum'),units=('n','sum')).loc[EVAL[0]:EVAL[1]]
        stock.append({'cohort':name,'median_stock_btc':c.stock.median(),'last_stock_btc':c.stock.iloc[-1],
                      'median_units':c.units.median(),'last_units':c.units.iloc[-1],
                      'phi':phi,'h':det.h,'z_max':det.z_max})
        w.to_csv(OUT/('screened_weekly_'+name+'.csv'))
    w0,phi0,_=stats['reference'];weeks=w0.loc[EVAL[0]:EVAL[1]].index
    starts=weeks[:len(weeks)-52:4];rows=[]
    sigma=np.sqrt(phi0*w0['var']);uv=(w0.energy/w0.exp_v.clip(lower=1e-12))
    for onset in starts:
        win=weeks[weeks>=onset][:52]
        for m in [.25,1.]:
            add,n=rq56.feasible_theft((m*sigma.loc[win]).values,uv.loc[win].values,
                                      1000.,w0.loc[onset,'stock_v'])
            for name,(w,phi,det) in stats.items():
                z=(w.loc[win,'S'].values+add)/np.sqrt(phi*w.loc[win,'var'].clip(lower=1e-12).values)
                hit=det.run(z);natural=det.run(det.z(w.loc[win]));fp=int(np.argmax(hit)) if hit.any() else 51
                rows.append({'onset':str(onset.date()),'cohort':name,'intensity':m,
                             'attack_btc':float(add.sum()),'max_proxy_spends':float(n.max()),
                             'natural_window_alarm':int(natural.any()),'alarm':int(hit.any()),
                             'loss_btc':float(add[:fp+1].sum()),'delay':fp+1 if hit.any() else None})
    df=pd.DataFrame(rows);df.to_csv(OUT/'screened_paired_attacks.csv',index=False)
    summary=df.groupby(['cohort','intensity']).agg(starts=('alarm','size'),alarm=('alarm','mean'),
        natural=('natural_window_alarm','mean'),median_loss_btc=('loss_btc','median'),
        median_delay=('delay','median')).reset_index()
    summary['excess']=summary.alarm-summary.natural;summary.to_csv(OUT/'screened_power_summary.csv',index=False)
    pd.DataFrame(stock).to_csv(OUT/'screened_stock.csv',index=False)
    protocol={'scope':'Exploratory aggregate output sensitivity on already-inspected history, not event replay or epsilon-covertness certification',
              'seed':20261016,'horizon':52,'onset_stride':4,'onsets':len(starts),'targeting':'value-tilted',
              'intensities':[.25,1.],'attack_plan':'same absolute BTC/count plan from reference cohort in all three conditions',
              'fitting':'same 50-step hazard fit, 13-week refits, 104-week lookback, 13-week gap',
              'calibration':'training-only 400 eight-week bootstraps, recomputed per cohort, nominal 104-week target',
              'detector':'ARGUS robust CUSUM/Shewhart only','key_limit':'1000 output proxies per week',
              'unresolved':'Does not evaluate screened event-level target cancellation or arbitrary script exposure.'}
    (OUT/'screened_protocol.json').write_text(json.dumps(protocol,indent=2))
    print(summary.to_string(index=False));print(pd.DataFrame(stock).to_string(index=False))


if __name__=='__main__':main()
