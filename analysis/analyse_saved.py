import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'code/experiments'))
from pathlib import Path
import json
import math
import numpy as np
import pandas as pd
import data_io as IO

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SOURCE = ROOT/'data/event_replay'
OUT = ROOT/'results/generated'
OUT.mkdir(exist_ok=True,parents=True)
SEED = 2026110307


def main():
    full = IO.replay_report()
    starts = sorted(full['per_start'], key=lambda x:x['sid'])
    names = list(full['summary']['natural_window_alarm_rate'])
    n, draws = 76, 8
    loss = np.zeros((n,draws,len(names)))
    alarm = np.zeros_like(loss)
    natural = np.zeros((n,len(names)))
    frozen = []
    scores, bases, stolen = [], [], []
    for sid,s in enumerate(starts):
        natural[sid] = [s['natural'][k] for k in names]
        runs = sorted([r for r in s['runs'] if r['law']=='tilted' and r['m']==1.], key=lambda x:x['draw'])
        assert len(runs)==8
        for d,r in enumerate(runs):
            loss[sid,d] = [r['det'][k]['loss_btc'] for k in names]
            alarm[sid,d] = [r['det'][k]['alarm'] for k in names]
            if d==0:
                for k,v in r['frozen_alarm'].items():
                    frozen.append({'sid':sid,'detector':k,'frozen_alarm':v,'refitted_alarm':r['det'][k]['alarm']})
        with IO.replay_arrays(SOURCE/'runs'/f'start_{sid:03d}.npz',allow_pickle=False) as f:
            order=[list(map(str,f['names'])).index(k) for k in names]
            scores.append(f['scores'][:,0,2][:,order,:])
            bases.append(f['base'][order])
            stolen.append(f['stolen'][:,0,2])
    scores,bases,stolen=map(np.asarray,[scores,bases,stolen])
    assert scores.shape==(76,8,9,52)
    for j,k in enumerate(names):
        r=full['summary']['laws']['tilted']['1.0']['det'][k]
        assert np.isclose(np.median(loss[:,:,j]),r['median_loss_btc'])
        assert np.isclose(alarm[:,:,j].mean(),r['detection_rate'])
    np.savez_compressed(OUT/'compact_replay_m1.npz',names=np.array(names),scores=scores,base=bases,
                        stolen=stolen,loss_nominal=loss,alarm_nominal=alarm,natural=natural,
                        starts=np.array([s['t0'] for s in starts]))

    rng=np.random.default_rng(SEED)
    idx=(rng.integers(0,n,size=(3000,math.ceil(n/13),1))+np.arange(13)).reshape(3000,-1)[:,:n]%n
    A=names.index('argus')
    paired=[]
    for j,k in enumerate(names):
        if j==A: continue
        ex=(alarm[:,:,A]-alarm[:,:,j]).mean(axis=1)-(natural[:,A]-natural[:,j])
        boot_ex=ex[idx].mean(axis=1)
        la,lb=loss[:,:,A][idx].reshape(3000,-1),loss[:,:,j][idx].reshape(3000,-1)
        row={'comparison':'ARGUS minus '+k,'starts':n,'draws_per_start':8,
             'excess_difference_pp':100*ex.mean(),
             'excess_ci_low_pp':100*np.quantile(boot_ex,.025),'excess_ci_high_pp':100*np.quantile(boot_ex,.975)}
        for q,label in [(.5,'median'),(.9,'p90')]:
            diff=np.quantile(la,q,axis=1)-np.quantile(lb,q,axis=1)
            row[label+'_loss_difference_btc']=np.quantile(loss[:,:,A],q)-np.quantile(loss[:,:,j],q)
            row[label+'_ci_low_btc']=np.quantile(diff,.025)
            row[label+'_ci_high_btc']=np.quantile(diff,.975)
        paired.append(row)
    pd.DataFrame(paired).to_csv(OUT/'paired_uncertainty.csv',index=False)
    pd.DataFrame(frozen).to_csv(OUT/'paired_frozen_draw_zero.csv',index=False)

    f=pd.DataFrame(frozen)
    f.groupby('detector')[['frozen_alarm','refitted_alarm']].mean().to_csv(OUT/'refit_detection_draw_zero.csv')

    grid=np.array(IO.replay_curves()['kappas'])
    rows=[];case_rows=[]
    cumulative=np.concatenate([np.zeros((n,draws,1)),stolen.astype(float).cumsum(axis=-1)],axis=-1)
    for origin in [26,39,52,65]:
        train=np.arange(origin-13)
        test=np.arange(origin,min(origin+13,n))
        assert train.max()*4+52<origin*4
        for budget in [.05,.10,.20]:
            for j,k in enumerate(names):
                compare=np.greater_equal if k in ['eprocess','ctm','threshold'] else np.greater
                train_rate=np.array([compare(bases[train,j],t).any(axis=-1).mean() for t in grid])
                valid=np.flatnonzero(train_rate<=budget)
                row={'origin':origin,'calibration_starts':len(train),'evaluation_starts':len(test),
                     'calibration_last_window_end':str(pd.Timestamp(starts[train[-1]]['t0'])+pd.Timedelta(weeks=52)),
                     'evaluation_first_onset':starts[origin]['t0'],'budget':budget,'detector':k,'attained':bool(len(valid))}
                if len(valid):
                    g=valid[0];threshold=grid[g];hit=compare(scores[test,:,j],threshold)
                    fp=np.where(hit.any(axis=-1),hit.argmax(axis=-1)+1,52)
                    ll=np.take_along_axis(cumulative[test],fp[...,None],axis=-1)[...,0]
                    rate=compare(bases[test,j],threshold).any(axis=-1).mean()
                    row.update(threshold_multiplier=threshold,calibration_rate=train_rate[g],evaluation_rate=rate,
                               detection=hit.any(axis=-1).mean(),excess=hit.any(axis=-1).mean()-rate,
                               median_loss_btc=np.median(ll),p90_loss_btc=np.quantile(ll,.9))
                    for ti,sid in enumerate(test):
                        untouched=int(compare(bases[sid,j],threshold).any())
                        for d in range(draws):
                            case_rows.append({'origin':origin,'sid':int(sid),'draw':d,'budget':budget,
                                'detector':k,'threshold_multiplier':threshold,'natural_window_alarm':untouched,
                                'alarm':int(hit[ti,d].any()),'loss_btc':float(ll[ti,d])})
                rows.append(row)
    pd.DataFrame(rows).to_csv(OUT/'retrospective_rolling_origin.csv',index=False)
    cases=pd.DataFrame(case_rows)
    cases.to_csv(OUT/'retrospective_rolling_cases.csv',index=False)
    pooled=cases.groupby(['budget','detector']).agg(cases=('alarm','size'),
        natural_window_rate=('natural_window_alarm','mean'),detection=('alarm','mean'),
        median_loss_btc=('loss_btc','median'),p90_loss_btc=('loss_btc',lambda x:x.quantile(.9)))
    pooled.to_csv(OUT/'retrospective_rolling_pooled.csv')
    protocol={'status':'Retrospective exploratory reanalysis, not a fresh confirmatory holdout',
              'seed':SEED,'bootstrap_replicates':3000,'bootstrap_block_starts':13,
              'target':'value-tilted m=1, address replay, loss includes alarm week or full horizon',
              'temporal_units':76,'target_draws':8,'rolling_origins':[26,39,52,65],
              'purge':'all calibration windows end strictly before first evaluated onset',
              'thresholds':'original 41-point grid, smallest feasible multiplier, no extrapolation',
              'loss_precision':'nominal paired intervals use saved float64 outcomes, rolling analysis uses saved float32 weekly theft values',
              'factorial_ablation':'Frozen versus refitted draw-zero detection only. Representation/cancellation/risk factors are not isolated.'}
    (OUT/'analysis_protocol.json').write_text(json.dumps(protocol,indent=2))
    print(json.dumps({'paired_comparisons':len(paired),'rolling_operating_points':len(rows),
                      'CTM_comparison':next(r for r in paired if r['comparison'].endswith('ctm'))},indent=2))


if __name__=='__main__':main()
