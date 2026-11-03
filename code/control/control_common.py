import os
for _name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '1'
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
sys.path.insert(0, str(ROOT / 'code/experiments'))
import common as C

PROTO = json.loads((ROOT/'config/control_base.json').read_text(encoding='utf-8'))
START = pd.Timestamp(PROTO['attack_window'][0])
END = pd.Timestamp(PROTO['attack_window'][1])
FIT_END = pd.Timestamp(PROTO['fit'][1])
COLS = ['n','sv','sv2','e_n','e_sv','e_sv2']
METHODS = {'count':None, 'cap_0_1':.1}




def fit(p, anchor=None):
    if anchor is None:
        q = p[(p.wk>=PROTO['fit'][0]) & (p.wk<FIT_END)]
    else:
        hi = anchor-pd.Timedelta(weeks=13)
        q = p[(p.wk>=hi-pd.Timedelta(weeks=104)) & (p.wk<hi)]
    return C.NullModel().fit(q)


def weighted_parts(p, model, cap):
    c = p[(p['pop']=='control') & (p.n>0)].copy()
    h = model.hazard(c)
    if cap is None:
        R,Q,Y = c.n,c.n,c.e_n
    else:
        low = c.vband.map(dict(zip(C.BANDS,[.001,.01,.1,1.,10.,100.]))).values
        assert np.isfinite(low).all()
        above = low>=cap
        R = np.where(above,cap*c.n,c.sv)
        Q = np.where(above,cap*cap*c.n,c.sv2)
        Y = np.where(above,cap*c.e_n,c.e_sv)
    return pd.DataFrame({'wk':c.wk,'Y':Y,'D':h*R,'Q':h*Q}).groupby('wk').sum().sort_index()


def config(p, model, cap):
    c = weighted_parts(p,model,cap)
    count = weighted_parts(p,model,None)
    tr = (c.index>=PROTO['fit'][0]) & (c.index<FIT_END)
    return {'cap':cap, 'kappa':20.*float(c.D[tr].median()/count.D[tr].median()),
            'q0':float(c.Q[tr].median()/c.D[tr].median())}


def stats(p, model, cfg):
    ct = weighted_parts(p,model,cfg['cap'])
    raw = (ct.Y/ct.D).replace([np.inf,-np.inf],np.nan)
    ct['prior'] = raw.ewm(span=8,min_periods=1).mean().shift(1).fillna(1.)
    k = cfg['kappa']
    ct['theta'] = (ct.Y+k*ct.prior)/(ct.D+k)


    ct['var_theta'] = ct.theta.clip(lower=1e-9)*(ct.Q+k*cfg['q0'])/(ct.D+k)**2
    e = p[(p['pop']=='exposed') & (p.n>0)].copy()
    e['h'] = model.hazard(e)
    e = e.join(ct[['theta','var_theta']],on='wk')
    e['p'] = np.minimum(e.theta*e.h,.25)
    e['exp_v'] = e.p*e.sv
    e['bin_var'] = e.p*(1.-e.p)*e.sv2
    e['F0'] = e.h*e.sv
    w = e.groupby('wk').agg(obs_v=('e_sv','sum'),exp_v=('exp_v','sum'),
                            bin_var=('bin_var','sum'),F0=('F0','sum'),
                            var_theta=('var_theta','first'))
    w['S'] = w.obs_v-w.exp_v
    w['var'] = w.bin_var+w.var_theta*w.F0**2
    return w.join(ct[['Y','D','Q','theta','prior']])


def anchored_blocks(p, end=None):
    weeks = pd.DatetimeIndex(sorted(p.wk.unique()))
    weeks = weeks[weeks>=START]
    if end is not None:
        weeks = weeks[weeks<end]
    for j in range(0,len(weeks),13):
        yield weeks[j],weeks[j:j+13]














