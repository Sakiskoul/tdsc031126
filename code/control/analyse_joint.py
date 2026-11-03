import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[k]='1'
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parents[2]/'data/control_pilot'


def main():
    result={'protocol':json.loads((Path(__file__).resolve().parents[2]/'config/control_pilot.json').read_text())}
    rows=pd.read_csv(OUT/'all_outcomes.csv',float_precision='round_trip')
    null=pd.read_csv(OUT/'all_poisoning_only.csv',float_precision='round_trip')
    inventory=pd.read_csv(OUT/'inventory_accounting.csv',float_precision='round_trip')
    certs=pd.read_csv(OUT/'all_masking_certificates.csv',float_precision='round_trip')
    rows=rows.drop(columns=['poison_only_alarm','excess_alarm'],errors='ignore')
    nullkeys = ['sid','mode','method','detector','operating_point','budget_btc','policy']
    rows = rows.merge(null[nullkeys+['alarm']].rename(columns={'alarm':'poison_only_alarm'}),on=nullkeys,validate='many_to_one')
    rows['excess_alarm'] = rows.alarm-rows.poison_only_alarm
    rows.to_csv(OUT/'all_outcomes.csv',index=False)
    null.to_csv(OUT/'all_poisoning_only.csv',index=False)
    inventory.to_csv(OUT/'inventory_accounting.csv',index=False)
    certs.to_csv(OUT/'all_masking_certificates.csv',index=False)
    gkeys = ['mode','operating_point','method','detector','budget_btc','policy']
    summary = rows.groupby(gkeys).agg(scenarios=('sid','size'),mean_loss_btc=('loss_btc','mean'),
        median_loss_btc=('loss_btc','median'),p90_loss_btc=('loss_btc',lambda x:x.quantile(.9)),
        alarm_rate=('alarm','mean'),poison_only_alarm_rate=('poison_only_alarm','mean'),
        excess_alarm_rate=('excess_alarm','mean'),natural_window_far=('natural_window_far','first')).reset_index()
    summary.to_csv(OUT/'summary.csv',index=False)
    rng = np.random.default_rng(20261103)
    starts = np.array(result['protocol']['start_ids'])
    blockstart = rng.integers(0,8,size=(10000,4))
    ix = np.stack([blockstart,(blockstart+1)%8],axis=-1).reshape(10000,8)

    comparisons = []
    def compare(a,b,keys,label):
        merged = a.merge(b,on=keys,suffixes=('_a','_b'),validate='one_to_one')
        merged['difference_btc'] = merged.loss_btc_a-merged.loss_btc_b
        merged.to_csv(OUT/f'paired_{label}_every_case.csv',index=False)
        groups = [x for x in keys if x not in ['sid','law','intensity']]
        for key,q in merged.groupby(groups):
            if not isinstance(key,tuple):
                key = (key,)
            perstart = q.groupby('sid').difference_btc.mean().reindex(starts)
            assert perstart.notna().all()
            z = perstart.to_numpy()
            boot = z[ix].mean(axis=1)
            lo,hi = np.quantile(boot,[.025,.975])
            comparisons.append(dict(zip(groups,key),comparison=label,mean_difference_btc=float(z.mean()),
                 ci95_lower_btc=float(lo),ci95_upper_btc=float(hi),starts=8,
                 scenario_pairs=len(q),start_differences=z.tolist(),
                 interval_scope='exploratory 8-start circular block bootstrap, length 2; not confirmatory'))
    keys = ['sid','law','intensity','mode','operating_point','detector','budget_btc','policy']
    compare(rows[rows.method=='cap_0_1'],rows[rows.method=='count'],keys,'cap_minus_count')
    keys = ['sid','law','intensity','mode','operating_point','method','budget_btc','policy']
    compare(rows[rows.detector=='argus'],rows[rows.detector=='ctm'],keys,'argus_minus_ctm')
    pairs = pd.DataFrame(comparisons)
    pairs.drop(columns='start_differences').to_csv(OUT/'paired_summary.csv',index=False)

    keys = ['sid','law','intensity','mode','operating_point','method','detector']
    base = rows[rows.budget_btc==0][keys+['loss_btc']].rename(columns={'loss_btc':'unpoisoned_loss_btc'})
    marginal = rows.merge(base,on=keys,validate='many_to_one')
    marginal['poisoning_loss_increase_btc'] = marginal.loss_btc-marginal.unpoisoned_loss_btc
    marginal.to_csv(OUT/'poisoning_marginal_effects.csv',index=False)
    harm = marginal.groupby(gkeys).poisoning_loss_increase_btc.agg(['mean','median','min','max']).reset_index()
    harm.to_csv(OUT/'poisoning_marginal_summary.csv',index=False)
    meta = {'starts':8,'theft_scenarios':32,'joint_attack_scenarios':224,
            'evaluated_outcomes':len(rows),'poisoning_only_outcomes':len(null),
            'fixed_hazard_certificate_cases':len(certs),
            'max_certificate_violation_btc':float(certs.max_violation.max()),
            'ci_scope':'Eight chronological starts, four fixed scenarios each, one target draw; intervals are exploratory and no multiple-comparison correction/confirmatory claim is made.',
            'operating_scope':'Retrospective natural curves over 76 overlapping windows. 25% and 50% are horizon window budgets, not per-week rates.',
            'paired_comparisons':comparisons}
    (OUT/'analysis.json').write_text(json.dumps(meta,indent=2,default=lambda x:x.item() if hasattr(x,'item') else str(x)),encoding='utf-8')
    fig,axes = plt.subplots(1,2,figsize=(11,4.5),sharey=True)
    wanted = [(10,'frontload'),(10,'uniform'),(100,'frontload'),(100,'uniform'),(1000,'frontload'),(1000,'uniform')]
    for ax,mode in zip(axes,['frozen','refitted']):
        q = pairs[(pairs.comparison=='cap_minus_count')&(pairs['mode']==mode)&
                  (pairs.operating_point=='0.25')&(pairs.detector=='argus')].set_index(['budget_btc','policy'])
        y = np.array([q.loc[k,'mean_difference_btc'] for k in wanted])/1000
        lo = np.array([q.loc[k,'ci95_lower_btc'] for k in wanted])/1000
        hi = np.array([q.loc[k,'ci95_upper_btc'] for k in wanted])/1000
        ax.errorbar(np.arange(6),y,yerr=np.vstack([y-lo,hi-y]),fmt='o',capsize=4,color='#215c85')
        ax.axhline(0,color='#555555',linewidth=.8)
        ax.set_xticks(np.arange(6),[f'{c}\n{p[:5]}' for c,p in wanted],fontsize=9)
        ax.set_title('Onset-frozen model' if mode=='frozen' else 'Contaminated rolling refits')
        ax.set_xlabel('Resource ceiling (BTC) and policy')
        ax.grid(axis='y',alpha=.2)
    axes[0].set_ylabel('Capped minus count: loss (thousand BTC)\nNegative favours capped control')
    fig.suptitle('Exploratory paired loss; ARGUS, natural-window budget 25%\n8 starts, one pool draw; 95% moving-block bootstrap intervals',fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT/'paired_loss.png',dpi=180)
    plt.close(fig)
    print(json.dumps({k:v for k,v in meta.items() if k!='paired_comparisons'},indent=2))


if __name__=='__main__':
    main()
