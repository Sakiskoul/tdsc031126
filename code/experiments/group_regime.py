from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

for _k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_k] = "1"
if os.name == "nt":
    import ctypes
    try:
        ctypes.windll.kernel32.GetCurrentProcess.restype = ctypes.c_void_p
        ctypes.windll.kernel32.SetPriorityClass.argtypes = [ctypes.c_void_p, ctypes.c_uint]
        ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x4000)
    except OSError:
        pass

import numpy as np
import pandas as pd
import data_io as IO
import common as C
import rq56
from control_sensitivity import numeric_panel

ROOT = Path(__file__).resolve().parents[2]
TRAIN = ("2014-01-06", "2019-12-30")
EVAL = ("2020-01-06", "2026-09-28")
FIT = ("2014-01-06", "2019-01-07")
SELECT = ("2019-01-07", "2019-12-30")


def subset(p, window):
    return p[(p.wk >= window[0]) & (p.wk < window[1])]


def group_key(p, group):
    key=p[group[0]].astype(str)
    for column in group[1:]:
        key=key.str.cat(p[column].astype(str),sep='|')
    return key


def group_weekly(panel, model, groups, prior):


    c = panel[(panel['pop'] == 'control') & (panel.n > 0)].copy()
    e = panel[(panel['pop'] == 'exposed') & (panel.n > 0)].copy()
    c['grp'], e['grp'] = group_key(c, groups), group_key(e, groups)
    c['lam0'] = model.hazard(c) * c.n
    e['h'] = model.hazard(e)
    cg = c.groupby(['wk', 'grp']).agg(Cg=('e_n','sum'), Lg=('lam0','sum'))
    weeks = pd.DatetimeIndex(sorted(e.wk.unique()))
    labels = sorted(set(c.grp) | set(e.grp))
    cg = cg.reindex(pd.MultiIndex.from_product([weeks, labels], names=['wk','grp']), fill_value=0.0)
    glob = C.control_theta(panel, model).reindex(weeks)
    glob['theta'] = glob.theta.fillna(1.0)
    glob['mu0'] = glob.mu0.fillna(0.0)
    cg['ng'] = cg.index.get_level_values('grp').map(prior).fillna(1.0).to_numpy()
    cg['D'] = cg.Lg + cg.ng
    cg['theta0'] = cg.index.get_level_values('wk').map(glob.theta).to_numpy()
    cg['theta_g'] = (cg.Cg + cg.ng*cg.theta0) / cg.D
    e = e.join(cg[['theta_g']], on=['wk','grp'])
    e['p'] = (e.theta_g * e.h).clip(upper=.25)
    e['exp_v'], e['exp_n'] = e.p*e.sv, e.p*e.n
    e['bin_var'] = e.p*(1-e.p)*e.sv2
    e['hF'] = e.h*e.sv
    fg = e.groupby(['wk','grp']).hF.sum().reindex(cg.index, fill_value=0)
    a = fg / cg.D
    A = (fg*cg.ng/cg.D).groupby(level='wk').sum()
    b = A/(glob.mu0+20.0)
    coeff = a + cg.index.get_level_values('wk').map(b).to_numpy()

    cv = (coeff**2 * cg.theta_g * cg.Lg).groupby(level='wk').sum()
    cv += b**2 * (20.0*glob.theta)
    w = e.groupby('wk').agg(obs_v=('e_sv','sum'),obs_n=('e_n','sum'),
                          exp_v=('exp_v','sum'),exp_n=('exp_n','sum'),
                          bin_var=('bin_var','sum'),stock_v=('sv','sum'))
    w['var'] = w.bin_var + cv
    w['S'] = w.obs_v - w.exp_v
    return w


def priors(panel, model, groups, rho, fit):
    c = subset(panel, fit)
    c = c[(c['pop']=='control') & (c.n>0)].copy()
    c['grp'] = group_key(c,groups)
    c['L'] = c.n * model.hazard(c)

    labels = sorted(c.grp.unique())
    weeks = pd.date_range(fit[0], pd.Timestamp(fit[1])-pd.Timedelta(days=1),freq='W-MON')
    v = c.groupby(['wk','grp']).L.sum().unstack('grp').reindex(index=weeks,columns=labels).fillna(0.)
    return (rho*v.median().clip(lower=1.0)).to_dict()


def diagnostic(w, window):
    z = w.loc[window[0]:window[1]].copy()
    out = {}
    for kind in ('v','n'):
        r = (z['obs_'+kind].rolling(13).sum()/z['exp_'+kind].rolling(13).sum().clip(lower=1e-12)-1).dropna()
        out[kind+'_q90_abs'] = float(r.abs().quantile(.9))
    return out


def load_numeric(panel_name, out):
    cache=out/f'panel_{panel_name}.parquet'
    if cache.exists(): return IO.read_frame(cache)
    p=numeric_panel(ROOT/'data'/f'results_{panel_name}',out/f'numeric_{panel_name}')
    p.to_parquet(cache,index=False)
    return p


def screen(panel, out, name):
    m=C.NullModel().fit(subset(panel,FIT))
    w=C.weekly_stats(panel,m)
    ref=diagnostic(w,SELECT)
    rows=[]
    for group in [('vband',),('era',),('vband','era')]:
        for rho in (.1,1.,10.):
            pr=priors(panel,m,group,rho,FIT)
            d=diagnostic(group_weekly(panel,m,group,pr),SELECT)
            row={'group':list(group),'rho':rho,'diagnostic':d,
                 'value_ratio':d['v_q90_abs']/ref['v_q90_abs'],
                 'count_ratio':d['n_q90_abs']/ref['n_q90_abs']}
            row['eligible']=row['value_ratio']<=.8 and row['count_ratio']<=1.1
            rows.append(row)
            print(name,'screen',row,flush=True)
    ok=[r for r in rows if r['eligible']]
    selected=min(ok,key=lambda r:r['diagnostic']['v_q90_abs']) if ok else None
    result={'reference':ref,'candidates':rows,'selected':selected,'selection_uses_evaluation':False}
    (out/f'screen_{name}.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    return result


def frames(panel, choice):
    group,rho=choice['group'],choice['rho']
    m=C.NullModel().fit(subset(panel,TRAIN))
    pr=priors(panel,m,group,rho,TRAIN)
    baseline=C.weekly_stats(panel,m)
    grouped=group_weekly(panel,m,group,pr)
    outputs={'scalar':[baseline[baseline.index<EVAL[0]]], 'grouped':[grouped[grouped.index<EVAL[0]]]}
    weeks=baseline.index[baseline.index>=EVAL[0]]
    for i in range(0,len(weeks),13):
        block=weeks[i:i+13];hi=block[0]-pd.Timedelta(weeks=13);lo=hi-pd.Timedelta(weeks=104)
        f=(lo,hi);mm=C.NullModel().fit(subset(panel,f))
        pp=priors(panel,mm,group,rho,f)
        outputs['scalar'].append(C.weekly_stats(panel,mm).loc[block])
        outputs['grouped'].append(group_weekly(panel,mm,group,pp).loc[block])
        print('rolling block',block[0].date(),flush=True)
    return {k:pd.concat(v).sort_index() for k,v in outputs.items()}


def evaluate(panel, selection, out, name):
    wf=frames(panel,selection)
    old=pd.read_csv(ROOT/'results'/f'weekly_{name}.csv',parse_dates=['wk'],index_col='wk')
    ix=wf['scalar'].index.intersection(old.index)
    validation={}
    for col in ['S','var']:
        a,b=wf['scalar'].loc[ix,col],old.loc[ix,col]
        validation[col+'_max_relative_error']=float((a-b).abs().max()/max(b.abs().median(),1e-12))
        if not np.allclose(a,b,rtol=1e-8,atol=1e-6):
            raise ValueError(f'scalar reference {col} differs from the published export')
    detectors={};natural={};base_hits={};losses={};raw={}
    weeks=wf['scalar'].loc[EVAL[0]:EVAL[1]].index
    starts=weeks[:len(weeks)-52:4]
    sig0=None
    for label,w in wf.items():
        tr=w.loc[TRAIN[0]:TRAIN[1]]
        phi=C.estimate_phi(tr)
        z=w.S/np.sqrt(phi*w['var'].clip(lower=1e-12))
        arg=C.calibrate(C.RobustDetector(phi=phi),z.loc[TRAIN[0]:TRAIN[1]].values,104.,np.random.default_rng(20261016))
        ctm=C.ConformalDetector(c=104.)
        if label=='scalar':sig0=np.sqrt(phi*w['var'].clip(lower=1e-12))
        for kind,d in [('argus',arg),('ctm',ctm)]:
            key=label+'_'+kind
            zz=z.loc[TRAIN[0]:]
            ah=d.run(d.pvalues(np.array([]),zz.values)) if kind=='ctm' else d.run(zz.values)
            ser=pd.Series(ah,index=zz.index).loc[EVAL[0]:EVAL[1]]
            natural[key]={'count':int(ser.sum()),'weeks':[x.date().isoformat() for x in ser.index[ser]],
                          'phi':phi,'covid_z':float(z.loc['2020-03-23'])}
            detectors[key]=(d,z,np.sqrt(phi*w['var'].clip(lower=1e-12)),kind)
    reference=wf['scalar']
    uv=reference['energy']/reference.exp_v.clip(lower=1e-12)
    for key,(det,z,sd,kind) in detectors.items():
        base_hits[key]=[];raw[key]={str(m):[] for m in (.25,1.)}
        for t0 in starts:
            win=weeks[weeks>=t0][:52];hist=z[(z.index>=TRAIN[0])&(z.index<t0)].values
            def alarm(add):
                zz=(wf[key.split('_')[0]].loc[win,'S'].values+add)/sd.loc[win].values
                return det.run(det.pvalues(hist,zz)) if kind=='ctm' else det.run(zz)
            base_hits[key].append(int(alarm(np.zeros(52)).any()))
            for intensity in (.25,1.):
                add,_=rq56.feasible_theft((intensity*sig0.loc[win]).values,uv.loc[win].values,1000.,float(reference.loc[t0,'stock_v']))
                a=alarm(add);delay=int(a.argmax())+1 if a.any() else None
                loss=float(add[:delay].sum()) if delay else float(add.sum())
                raw[key][str(intensity)].append({'loss':loss,'alarm':int(a.any()),'delay':delay})
    idx=rq56.block_indices(len(starts),13,2000,np.random.default_rng(20261102))
    paired={};summary={}
    for key in detectors:
        n=float(np.mean(base_hits[key]));summary[key]={'historical':natural[key],'natural_window_rate':n,'attacks':{}}
        for intensity,r in raw[key].items():
            loss=np.array([x['loss'] for x in r]);hit=np.array([x['alarm'] for x in r])
            delays=[x['delay'] for x in r if x['delay']]
            summary[key]['attacks'][intensity]={'median_loss_btc':float(np.median(loss)),
                'detection':float(hit.mean()),'excess':float(hit.mean()-n),'median_delay':float(np.median(delays)) if delays else None}
    for ctm in ['grouped_ctm','scalar_ctm']:
        paired[ctm]={}
        for intensity in ['0.25','1.0']:
            a=np.array([x['loss'] for x in raw['grouped_argus'][intensity]])
            b=np.array([x['loss'] for x in raw[ctm][intensity]])
            dif=np.median(a[idx],axis=1)-np.median(b[idx],axis=1)
            mn=(a-b)[idx].mean(axis=1)
            ci=np.quantile(dif,[.025,.975])
            paired[ctm][intensity]={'difference_of_medians_btc':float(np.median(a)-np.median(b)),
                'paired_ci95':ci.tolist(),'mean_paired_difference_btc':float((a-b).mean()),
                'mean_ci95':np.quantile(mn,[.025,.975]).tolist(),
                'passes_criterion':natural['grouped_argus']['count']<=natural[ctm]['count'] and
                np.mean(base_hits['grouped_argus'])<=np.mean(base_hits[ctm]) and ci[1]<0}
    result={'name':name,'selected':selection,'reference_validation':validation,'starts':len(starts),'reference_attack_units':'original scalar sigma in BTC',
            'summary':summary,'paired_loss':paired,'scope':'exploratory aggregate injection; no event-level refitting of grouped variants',
            'training_diagnostics':{k:diagnostic(v,TRAIN) for k,v in wf.items()}}
    for label,w in wf.items():w.to_csv(out/f'{name}_{label}_weekly.csv')
    (out/f'evaluation_{name}.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    (out/f'windows_{name}.json').write_text(json.dumps({'starts':[str(x.date()) for x in starts],'base':base_hits,'attacks':raw},indent=2),encoding='utf8')
    print(json.dumps(result,indent=2),flush=True)
    return result


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--panel',choices=['full','addr'],required=True)
    ap.add_argument('--out',required=True);ap.add_argument('--evaluate',action='store_true');args=ap.parse_args()
    out=Path(args.out);out.mkdir(exist_ok=True,parents=True)
    protocol=out/'protocol.json'
    if not protocol.exists():raise ValueError('a time-stamped protocol must exist before the experiment')
    p=load_numeric(args.panel,out)
    path=out/f'screen_{args.panel}.json'
    s=json.loads(path.read_text()) if path.exists() else screen(p,out,args.panel)
    print('selected',s['selected'],flush=True)
    if args.evaluate and s['selected']:
        (out/f'frozen_selection_{args.panel}.json').write_text(json.dumps({'selection':s['selected'],
            'screen_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'protocol_sha256':hashlib.sha256(protocol.read_bytes()).hexdigest()},indent=2),encoding='utf8')
        evaluate(p,s['selected'],out,args.panel)
    elif args.evaluate:
        print('NO GO: training selection gate failed; evaluation is not searched',flush=True)


if __name__=='__main__':main()
