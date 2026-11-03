import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'code/experiments'))
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
sys.path.insert(0,str(ROOT/'code/control'))
import numpy as np
import pandas as pd
import data_io as IO
import json
import joint_replay as J

OUT=ROOT/'results/generated';OUT.mkdir(exist_ok=True,parents=True)
SRC=ROOT/'data/control_pilot'


def main():
    params=json.loads((ROOT/'config/control_score_parameters.json').read_text())
    baseline={'dets':{name:J.C.RobustDetector(**values) for name,values in params.items()},
              'hist':{name:pd.read_csv(SRC/f'history_{name}.csv',parse_dates=['wk'],index_col='wk')['z'] for name in params}}
    life=IO.read_frame(SRC/'life.parquet')
    rows=[];reconciled=0;maxerr=0.;theft_tables={}
    for sid,t0 in zip(J.P['start_ids'],J.P['starts']):
        checkpoint=IO.joint_checkpoint(sid)
        curves=IO.joint_curves(SRC/'starts'/f'curves_{sid:03d}.csv')
        score_cache={}
        for run in checkpoint['outcomes']:
            key=f"{run['law']}_{run['intensity']}_{run['budget_btc']}_{run['policy']}_{run['mode']}_{run['method']}_z"
            hist=baseline['hist'][run['method']][baseline['hist'][run['method']].index<pd.Timestamp(t0)].values
            if key not in score_cache:
                score_cache[key]=J.score_paths(baseline['dets'][run['method']],hist,curves[key])
            score=score_cache[key][run['detector']]
            hit=score>run['kappa'] if run['detector']=='argus' else score>=run['kappa']
            fp=int(np.argmax(hit)) if hit.any() else 51
            plan_key=(sid,run['law'],run['intensity'])
            if plan_key not in theft_tables:
                theft_tables[plan_key]=IO.read_frame(SRC/'plans'/f"theft_{sid:03d}_{run['law']}_{run['intensity']}.parquet")
            theft=theft_tables[plan_key]
            err=abs(theft.loc[theft.week<=fp,'btc'].sum()-run['loss_btc']);maxerr=max(maxerr,float(err))
            assert err<1e-7 and int(hit.any())==run['alarm'];reconciled+=1
        for budget in [10,100,1000]:
            for policy in ['frontload','uniform']:
                plan=IO.read_frame(SRC/'plans'/f'control_{sid:03d}_{budget}_{policy}.parquet')
                ts=pd.Timestamp(t0).value//10**9
                linked=life[life.id.isin(plan.id)].merge(plan[['id','activation']],on='id')
                spent=linked[(linked.created<ts)&(linked.created+365*86400<linked.activation)&
                             (linked.spent.isna()|(linked.spent>=linked.activation))]
                inputs=spent.groupby('id').sat.agg(['sum','size']).reindex(plan.id)
                assert np.array_equal(inputs['sum'].to_numpy(),plan.moved_sat.to_numpy())
                n_inputs=inputs['size'].to_numpy();vsize=10.5+68*n_inputs+31
                for fee_rate in [1,10,100]:
                    fees_sat=np.ceil(vsize*fee_rate)
                    rows.append({'sid':sid,'ceiling_btc':budget,'policy':policy,'addresses':len(plan),
                                 'charged_capital_btc':float(plan.initial_sat.sum()/1e8),
                                 'moved_btc':float(plan.moved_sat.sum()/1e8),'input_count':int(n_inputs.sum()),
                                 'assumed_sat_per_vbyte':fee_rate,'estimated_fees_btc':float(fees_sat.sum()/1e8),
                                 'fee_exceeds_input_addresses':int((fees_sat>=inputs['sum'].to_numpy()).sum()),
                                 'minimum_input_age_days':float(((spent.activation-spent.created)/86400).min())})
    pd.DataFrame(rows).to_csv(OUT/'joint_fee_scenarios.csv',index=False)
    df=pd.read_csv(SRC/'all_outcomes.csv')
    base=df[(df.budget_btc==0)&(df.policy=='none')]
    match=['sid','law','intensity','mode','method','detector','operating_point']
    paired=df.merge(base[match+['loss_btc','alarm']],on=match,suffixes=('','_no_poison'))
    paired['loss_change_btc']=paired.loss_btc-paired.loss_btc_no_poison
    paired['alarm_change']=paired.alarm-paired.alarm_no_poison
    paired.groupby(['mode','method','detector','operating_point','budget_btc','policy'])[
        ['loss_change_btc','alarm_change']].mean().to_csv(OUT/'joint_poisoning_effects.csv')
    result={'outcomes_reconciled':reconciled,'max_loss_difference_btc':maxerr,
            'fee_scope':'Illustrative one P2WPKH consolidation transaction per activated address, 68 vbytes/input, 31/output, 10.5 overhead, one change output. Script signing and actual fee markets are not measured.',
            'capital_scope':'Full initial linked inventory, fees paid out of moved value, not added again to committed capital. No re-entry within 364-day horizon.',
            'preparation_scope':'New deposits need 365 days. Existing mature holdings can be acquired later, so coin age is not a proved ownership/preparation duration.',
            'pilot_scope':'8 historical starts, one draw, bucket-0 candidates, exploratory. Broader inventory and fee-constrained replay remain untested.'}
    (OUT/'joint_checks.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))


if __name__=='__main__':main()
