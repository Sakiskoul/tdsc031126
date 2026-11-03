import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'code/experiments'))
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.stats import binom

HERE=Path(__file__).resolve().parent
OUT=ROOT/'results/generated';OUT.mkdir(exist_ok=True,parents=True)
SEED=2026110308


def main():
    rng=np.random.default_rng(SEED);reps=50000
    n,p,eps,mu=1000000,.001,.25,128
    energy=n*p;xi=np.exp(-mu/6);eps0=eps-xi;q=2*eps0*p/np.sqrt(energy)
    K=rng.binomial(n,q,size=reps);value=np.where(K<=mu,K,0)
    eta=.5;lower=(1-eta)*2*eps0*np.sqrt(energy)
    fail=np.mean(value<=lower);tail=np.exp(-eta*eta*eps0*np.sqrt(energy))+xi
    support=np.arange(int(binom.ppf(1-1e-14,n,p+q*(1-p)))+1)
    tv=.5*np.abs(binom.pmf(support,n,p+q*(1-p))-binom.pmf(support,n,p)).sum()
    budget={'n':n,'p':p,'epsilon':eps,'mu':mu,'epsilon_product':eps0,'expected_keys':n*q,
            'truncation_certificate':xi,'observed_budget_exceedances':int((K>mu).sum()),
            'loss_target':lower,'observed_lower_tail':fail,'proved_lower_tail_upper_bound':tail,
            'exact_untruncated_observation_TV':tv,'TV_support_omitted_mass_upper_bound':2e-14}


    sn,sp,sq,smu=8,.10,.08,2
    kk=np.arange(sn+1);pk=binom.pmf(kk,sn,sq)
    null=binom.pmf(kk,sn,sp);untruncated=np.zeros(sn+1);truncated=np.zeros(sn+1)
    for k in range(sn+1):
        law=np.zeros(sn+1);law[k:]=binom.pmf(np.arange(sn-k+1),sn-k,sp)
        untruncated+=pk[k]*law
        truncated+=pk[k]*(law if k<=smu else null)
    overflow=float(pk[smu+1:].sum())
    coupling_tv=float(.5*abs(truncated-untruncated).sum())
    assert np.isclose(truncated.sum(),1) and coupling_tv<=overflow+1e-14
    small={'n':sn,'p':sp,'q':sq,'mu':smu,'expected_keys':sn*sq,
        'overflow_probability':overflow,'chernoff_certificate':float(np.exp(-smu/6)),
        'exact_truncation_TV':coupling_tv,
        'untruncated_TV_to_null':float(.5*abs(untruncated-null).sum()),
        'truncated_TV_to_null':float(.5*abs(truncated-null).sum()),
        'untruncated_low_loss_probability':float(pk[:1].sum()),
        'truncated_low_loss_probability':float(pk[:1].sum()+overflow)}
    assert small['truncated_low_loss_probability']<=small['untruncated_low_loss_probability']+overflow+1e-14

    rows=[];v0=320;threshold=(1-p)*v0/2;bound=4*n*p*(1-p)/((1-p)**2*v0*v0)
    for name,batch,truep,estimate in [('oracle',1,p,p),('batch_4',4,p,p),('batch_16',16,p,p),
                                      ('hazard_error',1,p,.9*p),('drift_20pct',1,1.2*p,p)]:
        natural=batch*rng.binomial(n//batch,truep,size=reps)
        if batch==1:
            attacked=v0+rng.binomial(n-v0,truep,size=reps)
        else:

            attacked=v0+batch*rng.binomial((n-v0)//batch,truep,size=reps)
        rows.append({'setting':name,'replicates':reps,'theft_value':v0,'oracle_bound':bound,
                     'false_alarm':np.mean(natural-n*estimate>threshold),
                     'miss':np.mean(attacked-n*estimate<=threshold),
                     'actual_null_variance':batch*n*truep*(1-truep),
                     'declared_independent_variance':n*p*(1-p),'within_stated_model':name=='oracle'})
    pd.DataFrame(rows).to_csv(OUT/'controlled_bernoulli.csv',index=False)

    horizon,c=52,1000.;fractions=np.array([.01,.02,.05,.1,.2,.4,.8])
    erows=[]
    for name,batch_a,batch_c,ratio in [('poisson_oracle',1,1,1.),('control_batches_16',1,16,1.),
                                       ('control_batches_100',1,100,1.),('drift_50pct',1,1,1.5)]:
        M=np.zeros((reps,len(fractions)));alarm=np.zeros(reps,dtype=bool)
        ysum=0.
        for _ in range(horizon):
            A=batch_a*rng.poisson(100*ratio/batch_a,size=reps)
            C=batch_c*rng.poisson(100/batch_c,size=reps);total=A+C
            Y=np.divide(2*A,total,out=np.ones(reps),where=total>0)-1
            ysum+=Y.mean();M=(1+Y[:,None]*fractions)*(M+1)
            alarm|=M.mean(axis=1)>=c
        erows.append({'setting':name,'replicates':reps,'horizon':horizon,'threshold':c,
                      'run_length_bound_H_over_c':horizon/c,'observed_window_alarm':alarm.mean(),
                      'mean_Y':ysum/horizon,'within_stated_model':name=='poisson_oracle'})
    pd.DataFrame(erows).to_csv(OUT/'controlled_poisson.csv',index=False)
    result={'seed':SEED,'replicates':reps,'budget_check':budget,'exact_small_truncation':small,
            'scope':'Known-law Monte Carlo and exact homogeneous-binomial TV. Sensitivity cases deliberately violate the assumptions.',
            'bernoulli':rows,'poisson':erows}
    (OUT/'controlled_validation.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
