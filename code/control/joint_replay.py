import os
for _k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_k] = '1'
from pathlib import Path
import sys
import json
import hashlib
import pickle
import time
import numpy as np
import pandas as pd
import data_io as IO

OUT = Path(__file__).resolve().parents[2]/'data/control_pilot'
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'code/experiments'))
sys.path.insert(0,str(ROOT/'code/control'))
import common as C
import replay as R
import control_common as D
from fast_stats import Numeric
from control_engine import ControlEngine,apply_delta,KEY,COLS

P = json.loads((ROOT/'config/control_pilot.json').read_text(encoding='utf-8'))
METHODS = {'count':None,'cap_0_1':.1}
MODES = ['frozen','refitted']
KAPPAS = np.unique(np.r_[np.geomspace(.25,128.,101),1.])
TRAIN0,TRAIN1 = '2014-01-06','2019-12-30'
D.PROTO['fit'][1] = TRAIN1
D.FIT_END = pd.Timestamp(TRAIN1)


def save(path,obj):
    path.write_text(json.dumps(obj,indent=2,default=lambda x:x.item() if hasattr(x,'item') else str(x)),encoding='utf-8')


def fingerprint():
    files = [Path(__file__),ROOT/'code/control/control_engine.py',ROOT/'code/control/fast_stats.py',ROOT/'config/control_pilot.json']
    return hashlib.sha256(b''.join(x.read_bytes() for x in files)).hexdigest()


def get_baseline(panel):
    cache = OUT/'baseline.pkl'
    if cache.exists():
        b = pickle.loads(cache.read_bytes())
        if b['fingerprint']==fingerprint():
            return b
    num = Numeric(panel)
    initial = D.fit(panel)
    cfgs = {n:D.config(panel,initial,cap) for n,cap in METHODS.items()}
    static = num.stats(initial,cfgs)

    parity = {}
    for n,cfg in cfgs.items():
        ref = C.weekly_stats(panel,initial) if n=='count' else D.stats(panel,initial,cfg)
        for col in ['S','exp_v','var']:
            a,b = static[n][col].align(ref[col],join='inner')
            if n!='count':


                a,b = a.loc[TRAIN0:],b.loc[TRAIN0:]
            err = float(np.max(abs(a-b)/np.maximum(1.,abs(b))))
            assert err<1e-9,(n,col,err)
            parity[n+'_'+col] = err
    pieces = {n:[w[w.index<pd.Timestamp(R.EVAL[0])]] for n,w in static.items()}
    models,allstats = {},{}
    for anchor,block in D.anchored_blocks(panel):
        model = D.fit(panel,anchor)
        models[anchor] = model
        stats = num.stats(model,cfgs)
        allstats[anchor] = stats
        for n in METHODS:
            pieces[n].append(stats[n].reindex(block))
        print('baseline refit',anchor.date(),flush=True)
    rolling = {n:pd.concat(v).sort_index() for n,v in pieces.items()}
    published = pd.read_csv(ROOT/'results/weekly_addr.csv',parse_dates=['wk']).set_index('wk')
    for col in ['S','exp_v','var']:
        a,b = rolling['count'][col].align(published[col],join='inner')
        err = float(np.max(abs(a-b)/np.maximum(1.,abs(b))))
        assert err<1e-8,(col,err)
        parity['published_'+col] = err
    dets,calibration,hist = {},{},{}
    for name in METHODS:
        w = rolling[name]
        tr = w.loc[TRAIN0:TRAIN1]
        phi = C.estimate_phi(tr)
        det = C.calibrate(C.RobustDetector(phi=phi),C.RobustDetector(phi=phi).z(tr),
                         104.,np.random.default_rng(20261016),n_rep=400)
        dets[name] = det
        hist[name] = pd.Series(det.z(w),index=w.index).loc[TRAIN0:]
        a = pd.Series(det.run(hist[name].values),index=hist[name].index).loc[R.EVAL[0]:R.EVAL[1]]
        ct = C.ConformalDetector(c=104.)
        ac = pd.Series(ct.run(ct.pvalues(np.array([]),hist[name].values)),index=hist[name].index).loc[R.EVAL[0]:R.EVAL[1]]
        calibration[name] = {'phi':phi,'h':det.h,'z_max':det.z_max,
                            'continuous_natural_argus':int(a.sum()),'continuous_natural_ctm':int(ac.sum())}
    official = json.loads((ROOT/'results/experiments_addr.json').read_text())['rq3']['argus']
    assert abs(calibration['count']['h']-official['h'])<1e-8
    starts = R.starts(panel)
    maxima,operating = {},{}
    for mode in MODES:
        for name in METHODS:
            scores = []
            for t0 in starts:
                weeks = pd.date_range(t0,periods=52,freq='W-MON')
                w = (allstats[R.refit_anchor(t0)][name] if mode=='frozen' else rolling[name]).reindex(weeks)
                z = dets[name].z(w)
                history = hist[name][hist[name].index<t0].values
                scores.append(score_paths(dets[name],history,z))
            for det in ['argus','ctm']:
                mx = np.array([s[det].max() for s in scores])
                key = mode+'_'+name+'_'+det
                maxima[key] = mx.tolist()
                rates = np.array([np.mean(mx>k) if det=='argus' else np.mean(mx>=k) for k in KAPPAS])
                points = {'calibrated':{'kappa':1.,'natural_window_far':float(np.mean(mx>1.) if det=='argus' else np.mean(mx>=1.))}}
                for target in [.25,.50]:
                    feasible = np.flatnonzero(rates<=target)
                    j = int(feasible[0]) if len(feasible) else None
                    points[str(target)] = {'target':target,'attainable':j is not None,
                        'kappa':float(KAPPAS[j]) if j is not None else None,
                        'natural_window_far':float(rates[j]) if j is not None else None}
                operating[key] = points
    registry = {'protocol_sha256':hashlib.sha256((ROOT/'config/control_pilot.json').read_bytes()).hexdigest(),
                 'locked_before_attack_outcomes':True,'starts':[str(t.date()) for t in starts],
                 'kappas':KAPPAS.tolist(),'natural_max_scores':maxima,'operating_points':operating,
                 'calibration':calibration,'parity':parity}
    save(OUT/'operating_points.json',registry)
    for n,w in rolling.items():
        w.to_csv(OUT/f'honest_{n}.csv')
    b = {'fingerprint':fingerprint(),'cfgs':cfgs,'models':models,'allstats':allstats,
         'rolling':rolling,'hist':hist,'dets':dets,'registry':registry}
    cache.write_bytes(pickle.dumps(b))
    print('OPERATING POINTS LOCKED',json.dumps(calibration),flush=True)
    return b


def score_paths(det,hist,z):
    ct = C.ConformalDetector(c=104.)
    return {'argus':det.scores(z),'ctm':ct.scores(ct.pvalues(hist,z))}


def statistics(panel,t0,b,mode):
    weeks = pd.date_range(t0,periods=52,freq='W-MON')
    num = Numeric(panel)
    if mode=='frozen':
        model = b['models'][R.refit_anchor(t0)]
        return {n:w.reindex(weeks) for n,w in num.stats(model,b['cfgs']).items()}
    parts = {n:[] for n in METHODS}
    for j in range(0,52,13):
        block = weeks[j:j+13]
        anchor = R.refit_anchor(block[0])
        hi = anchor-pd.Timedelta(weeks=13)
        model = b['models'][anchor] if hi<=t0 else D.fit(panel,anchor)
        sw = num.stats(model,b['cfgs'])
        for n in METHODS:
            parts[n].append(sw[n].reindex(block))
    return {n:pd.concat(v).sort_index() for n,v in parts.items()}


def evaluated(stats,t0,b,mode,stolen,condition,extra):
    rows = []
    for name,w in stats.items():
        z = b['dets'][name].z(w)
        hist = b['hist'][name][b['hist'][name].index<t0].values
        paths = score_paths(b['dets'][name],hist,z)
        for det,path in paths.items():
            key = mode+'_'+name+'_'+det
            for label,op in b['registry']['operating_points'][key].items():
                k = op.get('kappa')
                if k is None:
                    continue
                crossing = path>k if det=='argus' else path>=k
                first = int(np.argmax(crossing)) if crossing.any() else None
                rows.append(dict(extra,mode=mode,method=name,detector=det,
                    condition=condition,operating_point=label,kappa=k,
                    natural_window_far=op['natural_window_far'],
                    alarm=int(first is not None),delay=first+1 if first is not None else None,
                    loss_btc=float(stolen[:first+1].sum()) if first is not None else float(stolen.sum()),
                    total_stolen_btc=float(stolen.sum())))
    return rows


def masking_certificate(observed,baseline,model,cfg,t0,Cbudget):

    if Cbudget<=0:
        assert np.allclose(observed.S,baseline.S,atol=1e-7)
        return {'max_masking_btc':0.,'max_bound_btc':0.,'max_violation':0.}
    rho = 1000. if cfg['cap'] is None else 1.
    hc = model.lam[model.lam.index.get_level_values('pop')=='control']
    a = float(hc.max())*rho
    raw = (observed.Y/observed.D).replace([np.inf,-np.inf],np.nan)
    after = observed.index>=t0
    lower_raw = raw.copy()
    lower_raw.loc[after] = (observed.Y[after]-rho*Cbudget).clip(lower=0.)/(observed.D[after]+a*Cbudget)
    assert (observed.D[after]>0).all()
    ix = observed.has_control.values.astype(bool)
    lower_prior = pd.Series(lower_raw.values[ix]).ewm(span=8,min_periods=1).mean().shift(1).fillna(1.).values
    prior = pd.Series(1.,index=observed.index)
    prior.iloc[np.flatnonzero(ix)] = lower_prior
    denominator = observed.D+a*Cbudget*after.astype(float)+cfg['kappa']
    num = observed.Y.copy()
    num.loc[after] = (num.loc[after]-rho*Cbudget).clip(lower=0.)
    theta_lower = (num+cfg['kappa']*prior)/denominator
    assert ((theta_lower-baseline.theta)[after]<=1e-8).all()
    bound = 1.25*Cbudget+observed.F0*(observed.theta-theta_lower).clip(lower=0.)
    mask = baseline.S-observed.S
    violation = float((mask[after]-bound[after]).max())
    assert violation<1e-6,(violation,cfg)
    return {'max_masking_btc':float(mask[after].max()),'max_bound_btc':float(bound[after].max()),
            'max_violation':violation,'min_bound_theta':float(theta_lower[after].min())}


def main():
    started = time.monotonic()
    (OUT/'starts').mkdir(exist_ok=True)
    (OUT/'plans').mkdir(exist_ok=True)
    panel = IO.read_frame(ROOT/'data/exploratory/grouped/panel_addr.parquet')
    b = get_baseline(panel)
    engine = ControlEngine(OUT)
    summaries = []
    for sid,t0text in zip(P['start_ids'],P['starts']):
        checkpoint = OUT/f'starts/start_{sid:03d}.json'
        if checkpoint.exists():
            saved = json.loads(checkpoint.read_text())
            if saved['fingerprint']==fingerprint():
                summaries.append(saved)
                continue
        t0 = pd.Timestamp(t0text)
        weeks = pd.date_range(t0,periods=52,freq='W-MON')
        prefix = panel[panel.wk<=weeks[-1]].copy()
        info,poison,poison_stats,poison_rows = [],{}, {},[]
        for budget in P['control_budgets_btc']:
            for policy in (['none'] if budget==0 else ['frontload','uniform']):
                tag = f'{budget}_{policy}'
                if budget:
                    plan = engine.plan(sid,t0,budget,policy)
                    delta,details = engine.delta(plan,t0)
                    plan.to_parquet(OUT/f'plans/control_{sid:03d}_{tag}.parquet',index=False)
                    delta.to_parquet(OUT/f'plans/control_delta_{sid:03d}_{tag}.parquet',index=False)
                    details.update(budget_btc=budget,policy=policy,
                        inventory_exhausted=bool(len(plan)==50000),unused_budget_btc=budget-details['charged_btc'])
                else:
                    delta = pd.DataFrame(columns=KEY+['wk']+COLS)
                    details = dict(budget_btc=0,policy='none',charged_btc=0.,moved_btc=0.,addresses=0)
                info.append(details)
                poison[tag] = delta
                altered = apply_delta(prefix,delta)
                for mode in MODES:
                    sw = statistics(altered,t0,b,mode)
                    poison_stats[tag,mode] = sw
                    poison_rows.extend(evaluated(sw,t0,b,mode,np.zeros(52),'poisoning_only',
                                     dict(sid=sid,t0=t0text,budget_btc=budget,policy=policy)))
        pool,chg,ev,meta = R.load_pool(ROOT/'data/replay/pools',64000,sid)
        ids,row,V,E = R.trajectories(pool,chg,ev,t0)
        control_ids = engine.candidates[engine.candidates.sid==sid].id
        assert not np.intersect1d(ids.id.values,control_ids.values).size
        sigma = np.sqrt(b['dets']['count'].phi*b['rolling']['count']['var'])
        preceding = sigma[(sigma.index>=t0-pd.Timedelta(weeks=52))&(sigma.index<t0)]
        pace = float(preceding.median())
        assert len(preceding)==52 and pace>0
        runs,certs,curves = [],[],{}
        for law in P['theft_laws']:
            pl = pool[pool.law==law].sort_values('key')
            order = row.loc[pl.id.values].values
            for intensity in P['theft_intensities']:
                taken = R.thief(order,V,np.full(52,intensity*pace))
                ok = taken>=0
                rows,jt = order[ok],taken[ok]
                stolen = np.bincount(jt,weights=V[rows,jt],minlength=52)
                td = R.deltas(ids,rows,jt,V,E,weeks).rename(columns={
                    'dn':'n','dsv':'sv','dsv2':'sv2','de_n':'e_n','de_sv':'e_sv','de_sv2':'e_sv2'})
                shared = pd.DataFrame({'id':ids.id.values[rows],'week':jt,'btc':V[rows,jt]})
                shared.to_parquet(OUT/f'plans/theft_{sid:03d}_{law}_{intensity}.parquet',index=False)
                base_theft = {}
                for tag,delta in poison.items():
                    budget = int(tag.split('_')[0])
                    policy = tag.split('_')[1]
                    combined = pd.concat([td[KEY+['wk']+COLS],delta],ignore_index=True)
                    attacked = apply_delta(prefix,combined)
                    for mode in MODES:
                        sw = statistics(attacked,t0,b,mode)
                        if budget==0:
                            base_theft[mode] = sw
                        extra = dict(sid=sid,t0=t0text,law=law,intensity=intensity,budget_btc=budget,
                                     policy=policy,theft_pace_btc=intensity*pace,stolen_addresses=int(ok.sum()))
                        runs.extend(evaluated(sw,t0,b,mode,stolen,'joint',extra))
                        for name,w in sw.items():
                            curves[f'{law}_{intensity}_{tag}_{mode}_{name}_S'] = w.S.values
                            curves[f'{law}_{intensity}_{tag}_{mode}_{name}_z'] = b['dets'][name].z(w)
                        if mode=='frozen' and budget:

                            model = b['models'][R.refit_anchor(t0)]
                            obsfull = Numeric(attacked).stats(model,b['cfgs'])
                            pure = Numeric(apply_delta(prefix,td)).stats(model,b['cfgs'])
                            for name in METHODS:
                                certs.append(dict(extra,method=name,**masking_certificate(
                                    obsfull[name],pure[name],model,b['cfgs'][name],t0,budget)))
                print('trajectory',sid,law,intensity,'theft',round(stolen.sum()),flush=True)
        res = {'fingerprint':fingerprint(),'sid':sid,'t0':t0text,'sigma_at_onset':pace,
               'inventory':info,'pool':meta,'outcomes':runs,'poisoning_only':poison_rows,
               'certificates':certs}
        np.savez_compressed(OUT/f'starts/curves_{sid:03d}.npz',**curves)
        save(checkpoint,res)
        summaries.append(res)
        print('START COMPLETE',sid,'elapsed',round(time.monotonic()-started,1),flush=True)
    save(OUT/'joint_results.json',{'protocol':P,'fingerprint':fingerprint(),'starts':summaries,
        'operating_points_sha256':hashlib.sha256((OUT/'operating_points.json').read_bytes()).hexdigest(),
        'elapsed_seconds':time.monotonic()-started})
    print('JOINT PILOT COMPLETE',round(time.monotonic()-started,1),flush=True)


if __name__=='__main__':
    main()
