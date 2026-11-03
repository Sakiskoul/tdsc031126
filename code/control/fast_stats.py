import numpy as np
import pandas as pd

KEY = ['pop','fam','era','vband','band']
EDGES = {'v0_001':.001,'v0_01':.01,'v0_1':.1,'v1':1.,'v10':10.,'v100':100.}


class Numeric:
    def __init__(self,panel):
        self.p = panel
        self.weeks = pd.DatetimeIndex(sorted(panel.wk.unique()),name='wk')
        self.t = self.weeks.get_indexer(panel.wk)
        self.N = len(self.weeks)
        self.ctl = ((panel['pop']=='control')&(panel.n>0)).values
        self.exp = ((panel['pop']=='exposed')&(panel.n>0)).values
        self.n = panel.n.to_numpy(float)
        self.v = panel.sv.to_numpy(float)
        self.v2 = panel.sv2.to_numpy(float)
        self.en = panel.e_n.to_numpy(float)
        self.ev = panel.e_sv.to_numpy(float)
        self.low = panel.vband.map(EDGES).to_numpy(float)
        self.hasctl = self.sum(self.ctl.astype(float))>0
        self.hasexp = self.sum(self.exp.astype(float))>0
        self.obs = self.sum(np.where(self.exp,self.ev,0.))
        self.weights = {}

    def sum(self,x):
        return np.bincount(self.t,weights=x,minlength=self.N)

    def stats(self,model,configs):
        h = model.hazard(self.p)
        F0 = self.sum(np.where(self.exp,h*self.v,0.))
        out = {}
        for name,cfg in configs.items():
            cap = cfg['cap']
            if cap not in self.weights:
                if cap is None:
                    W,Q,Y = self.n,self.n,self.en
                else:
                    above = self.low>=cap
                    W = np.where(above,cap*self.n,self.v)
                    Q = np.where(above,cap*cap*self.n,self.v2)
                    Y = np.where(above,cap*self.en,self.ev)
                self.weights[cap] = (np.where(self.ctl,W,0.),np.where(self.ctl,Q,0.),
                                     self.sum(np.where(self.ctl,Y,0.)))
            W,Q,Y = self.weights[cap]
            D,QQ = self.sum(h*W),self.sum(h*Q)
            raw = np.divide(Y,D,out=np.full(self.N,np.nan),where=D>0)
            ix = np.flatnonzero(self.hasctl)
            prior = np.ones(self.N)
            prior[ix] = pd.Series(raw[ix]).ewm(span=8,min_periods=1).mean().shift(1).fillna(1.).values
            k = cfg['kappa']
            theta = (Y+k*prior)/(D+k)
            vt = np.maximum(theta,1e-9)*(QQ+k*cfg['q0'])/(D+k)**2
            theta[~self.hasctl],vt[~self.hasctl] = 1.,0.
            p = np.minimum(theta[self.t]*h,.25)
            pred = self.sum(np.where(self.exp,p*self.v,0.))
            bv = self.sum(np.where(self.exp,p*(1-p)*self.v2,0.))
            out[name] = pd.DataFrame({'obs_v':self.obs,'exp_v':pred,'S':self.obs-pred,
               'var':bv+vt*F0**2,'F0':F0,'Y':Y,'D':D,'Q':QQ,'theta':theta,'prior':prior,
               'has_control':self.hasctl},index=self.weeks).loc[self.hasexp]
        return out
