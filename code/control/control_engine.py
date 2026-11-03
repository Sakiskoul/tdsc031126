from pathlib import Path
import numpy as np
import pandas as pd
import data_io as IO
import duckdb

KEY = ['pop','fam','era','vband','band']
COLS = ['n','sv','sv2','e_n','e_sv','e_sv2']
T_END = pd.Timestamp('2026-10-01')
DAY = 86400
VB = "CASE WHEN {v}<1000000 THEN 'v0_001' WHEN {v}<10000000 THEN 'v0_01' WHEN {v}<100000000 THEN 'v0_1' WHEN {v}<1000000000 THEN 'v1' WHEN {v}<10000000000 THEN 'v10' ELSE 'v100' END"


class ControlEngine:
    def __init__(self, folder):
        self.folder = Path(folder)
        self.candidates = IO.read_frame(self.folder/'candidates.parquet')
        self.life = IO.read_frame(self.folder/'life.parquet')
        self.segments = IO.read_frame(self.folder/'segments.parquet')
        self.con = duckdb.connect(':memory:')
        self.con.execute('SET threads=1')
        self.con.execute("SET memory_limit='2GB'")
        self.con.execute("SET max_temp_directory_size='0B'")
        self.con.execute('SET preserve_insertion_order=false')

    def plan(self, sid, t0, budget, policy):
        c = self.candidates[self.candidates.sid==sid].sort_values('rank').copy().reset_index(drop=True)
        assert not c.id.duplicated().any()
        ids = pd.Index(c.id)
        l = self.life[self.life.id.isin(ids)].copy()
        row = ids.get_indexer(l.id)
        created = l.created.to_numpy(dtype=np.int64)
        spent = l.spent.fillna(np.inf).to_numpy(dtype=float)
        revealed = l.revealed.fillna(np.inf).to_numpy(dtype=float)
        val = l.sat.to_numpy(dtype=np.int64)
        ts0 = int(pd.Timestamp(t0).value//10**9)
        initial = (created<ts0)&(spent>=ts0)
        stock = np.bincount(row[initial],weights=val[initial],minlength=len(c)).astype(np.int64)
        assert np.array_equal(stock,c.initial_sat.to_numpy(dtype=np.int64))
        assert (stock>=c.eligible_sat.to_numpy()).all()
        week = np.full(len(c),-1,dtype=int)
        moved = np.zeros(len(c),dtype=np.int64)
        charged = 0
        total = int(round(budget*1e8))
        for j in range(52):
            ts = ts0+j*7*DAY
            eligible = (created+365*DAY<ts)&(spent>=ts)&(revealed>=ts)
            v = np.bincount(row[eligible],weights=val[eligible],minlength=len(c)).astype(np.int64)

            lim = total if policy=='frontload' else total*(j+1)//52
            remaining = lim-charged
            ok = (week<0)&(v>0)&(stock<=remaining)
            cs = np.cumsum(np.where(ok,stock,0),dtype=np.int64)
            chosen = np.flatnonzero(ok&(cs<=remaining))
            week[chosen] = j
            moved[chosen] = v[chosen]
            charged += int(stock[chosen].sum())
        q = c[week>=0].copy()
        q['week'] = week[week>=0]
        q['activation'] = ts0+q.week*7*DAY
        q['moved_sat'] = moved[week>=0]
        assert q.initial_sat.sum()<=total
        assert (q.moved_sat<=q.initial_sat).all()
        q['policy'],q['budget_btc'] = policy,budget
        return q

    def new_segments(self, plan):
        ids = plan.id.unique()
        l = self.life[self.life.id.isin(ids)].copy()
        a = plan[['id','activation']]
        self.con.register('lif',l)
        self.con.register('act',a)


        sql = """
        WITH iv AS (
          SELECT lif.*,act.activation,
            make_timestamp(created*1000000) AS t_create,
            make_timestamp((created+365*86400)*1000000) AS t_dorm,
            CASE WHEN created+365*86400<activation AND (spent IS NULL OR spent>=activation)
                   THEN make_timestamp(activation*1000000)
                 WHEN spent<epoch(TIMESTAMP '2026-10-01') THEN make_timestamp(spent::BIGINT*1000000)
                 ELSE NULL END AS t_spend,
            make_timestamp(least(coalesce(revealed,activation),activation)::BIGINT*1000000) AS t_rev,
            CASE typ WHEN 1 THEN 'p2pk' WHEN 6 THEN 'p2tr' WHEN 2 THEN 'p2pkh'
                     WHEN 3 THEN 'p2pkh' ELSE 'p2sh' END AS fam
          FROM lif JOIN act USING(id)
        ), meta AS (
          SELECT *,CASE WHEN year(t_create)<=2010 THEN 'e2009_10'
                        WHEN year(t_create)<=2012 THEN 'e2011_12'
                        WHEN year(t_create)<=2016 THEN 'e2013_16'
                        WHEN year(t_create)<=2020 THEN 'e2017_20' ELSE 'e2021_' END AS era
          FROM iv
        ), intervals AS (
          SELECT 'control' AS pop,id,fam,era,sat,t_create,t_spend,t_dorm AS a,
                 least(coalesce(t_spend,TIMESTAMP '2026-10-01'),t_rev) AS b
          FROM meta WHERE fam IN ('p2pkh','p2sh')
          UNION ALL
          SELECT 'exposed',id,fam,era,sat,t_create,t_spend,
                 greatest(CASE WHEN typ IN (1,6) THEN t_create ELSE t_rev END,t_dorm),
                 coalesce(t_spend,TIMESTAMP '2026-10-01')
          FROM meta WHERE typ IN (1,6) OR t_rev IS NOT NULL
        )
        SELECT pop,id,fam,era,sat,t_spend AS spent,band,
               greatest(a,t_create+to_days(dlo)) AS a,
               least(b,t_create+to_days(dhi)) AS b
        FROM intervals CROSS JOIN (VALUES ('a1_2',365,730),('a2_5',730,1826),
                ('a5_10',1826,3652),('a10_',3652,40000)) AS bands(band,dlo,dhi)
        WHERE a<b AND greatest(a,t_create+to_days(dlo))<least(b,t_create+to_days(dhi))
        """
        return self.con.execute(sql).fetchdf()

    def contribution(self, seg, weeks):
        if not len(seg):
            return pd.DataFrame(columns=KEY+['wk']+COLS)
        s = seg.copy()
        for name in ['a','b','spent']:
            if pd.api.types.is_numeric_dtype(s[name]):


                s[name] = pd.to_datetime(s[name].astype('Int64'),unit='s')
        self.con.register('seg',s)
        self.con.register('calendar',pd.DataFrame({'wk':pd.DatetimeIndex(weeks)}))
        self.con.execute('''CREATE OR REPLACE TEMP TABLE meta AS
              SELECT pop,id,min(era) AS era,min(fam) AS fam FROM seg GROUP BY pop,id''')
        self.con.execute('''CREATE OR REPLACE TEMP TABLE x AS
          WITH d AS (SELECT pop,id,a AS ts,sat AS dv FROM seg UNION ALL
                     SELECT pop,id,b AS ts,-sat AS dv FROM seg),
            c AS (SELECT pop,id,ts,sum(dv)::BIGINT AS dv FROM d GROUP BY pop,id,ts)
          SELECT *,sum(dv) OVER (PARTITION BY pop,id ORDER BY ts
                    ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)::BIGINT AS v_new
          FROM c''')
        sql = f"""
          WITH y AS (
            SELECT x.*,x.v_new-x.dv AS v_old,m.fam,m.era,
                   date_trunc('week',x.ts)+INTERVAL 7 DAY AS wk
            FROM x JOIN meta m USING(pop,id) WHERE dv<>0
          ), rd AS (
            SELECT pop,fam,era,vband,wk,sum(dn) AS dn,sum(dv) AS dv,sum(dv2) AS dv2
            FROM (
              SELECT pop,fam,era,{VB.format(v='v_old')} AS vband,wk,
                     -1. AS dn,-v_old/1e8 AS dv,-(v_old/1e8)^2 AS dv2 FROM y WHERE v_old>0
              UNION ALL
              SELECT pop,fam,era,{VB.format(v='v_new')},wk,
                     1.,v_new/1e8,(v_new/1e8)^2 FROM y WHERE v_new>0
            ) GROUP BY ALL
          ) SELECT rd.pop,rd.fam,rd.era,rd.vband,'all' AS band,c.wk,
                   sum(dn)::DOUBLE AS n,sum(dv)::DOUBLE AS sv,sum(dv2)::DOUBLE AS sv2
            FROM rd JOIN calendar c ON rd.wk<=c.wk GROUP BY ALL
        """
        risk = self.con.execute(sql).fetchdf()
        events = self.con.execute(f"""
          WITH ev AS (
            SELECT DISTINCT pop,id,date_trunc('week',spent) AS wstart FROM seg
            WHERE spent IS NOT NULL AND spent>=a AND spent<=b
              AND spent<TIMESTAMP '2026-10-01'
              AND date_trunc('week',a)<date_trunc('week',spent)
              AND (pop='exposed' OR spent=b)
          ), j AS (
            SELECT ev.pop,ev.id,ev.wstart,x.v_new AS v FROM ev ASOF JOIN x
              ON ev.pop=x.pop AND ev.id=x.id AND ev.wstart>x.ts
          ) SELECT j.pop,m.fam,m.era,{VB.format(v='j.v')} AS vband,'all' AS band,
                   j.wstart AS wk,count(*)::DOUBLE AS e_n,
                   sum(j.v/1e8)::DOUBLE AS e_sv,sum((j.v/1e8)^2)::DOUBLE AS e_sv2
            FROM j JOIN meta m USING(pop,id) JOIN calendar c ON j.wstart=c.wk
            WHERE j.v>0 GROUP BY ALL
        """).fetchdf()
        out = risk.merge(events,on=KEY+['wk'],how='outer').fillna(0.)
        return out

    def delta(self, plan, t0):
        weeks = pd.date_range(t0,periods=52,freq='W-MON')
        if not len(plan):
            return pd.DataFrame(columns=KEY+['wk']+COLS),{}
        old = self.segments[self.segments.id.isin(plan.id)].copy()
        new = self.new_segments(plan)
        before = self.contribution(old,weeks).set_index(KEY+['wk'])[COLS]
        after = self.contribution(new,weeks).set_index(KEY+['wk'])[COLS]
        delta = after.subtract(before,fill_value=0.).reset_index()
        info = {'addresses':len(plan),'charged_btc':float(plan.initial_sat.sum()/1e8),
                'moved_btc':float(plan.moved_sat.sum()/1e8),
                'initial_immature_btc':float((plan.initial_sat-plan.eligible_sat).sum()/1e8),
                'original_control_events_btc':float(before.loc[before.index.get_level_values('pop')=='control','e_sv'].sum()),
                'original_exposed_events_btc':float(before.loc[before.index.get_level_values('pop')=='exposed','e_sv'].sum()),
                'new_exposed_events_btc':float(after.loc[after.index.get_level_values('pop')=='exposed','e_sv'].sum())}
        return delta,info


def apply_delta(panel, delta):
    if not len(delta):
        return panel.copy()
    base = panel.set_index(KEY+['wk'])[COLS]
    d = delta.groupby(KEY+['wk'])[COLS].sum()
    out = base.add(d,fill_value=0.)
    for col in COLS:

        bad = out[col]<-1e-5
        if bad.any():
            raise ValueError(f'Negative {col}: {out.loc[bad,col].min()}')
        out[col] = out[col].clip(lower=0.)
    return out.reset_index()
