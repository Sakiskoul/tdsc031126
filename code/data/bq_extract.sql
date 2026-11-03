WITH
outs AS (
  SELECT
    o.transaction_hash AS txh,
    o.index AS vout,
    o.block_timestamp AS t_create,
    o.type,
    o.addresses[SAFE_OFFSET(0)] AS addr,
    CAST(o.value AS INT64) AS sat
  FROM `bigquery-public-data.crypto_bitcoin.outputs` AS o
  WHERE o.block_timestamp < @t_end
    AND o.value >= @dust
    AND o.type IN ('pubkey', 'pubkeyhash', 'witness_v0_keyhash', 'scripthash',
                   'witness_v0_scripthash', 'witness_v1_taproot')
),
spends AS (
  SELECT i.spent_transaction_hash AS txh, i.spent_output_index AS vout, i.block_timestamp AS t_spend
  FROM `bigquery-public-data.crypto_bitcoin.inputs` AS i
  WHERE i.block_timestamp < @t_end
),
life AS (
  SELECT
    outs.*,
    spends.t_spend,
    CASE outs.type
      WHEN 'pubkey' THEN 'p2pk'
      WHEN 'witness_v1_taproot' THEN 'p2tr'
      WHEN 'pubkeyhash' THEN 'p2pkh'
      WHEN 'witness_v0_keyhash' THEN 'p2pkh'
      ELSE 'p2sh'
    END AS fam
  FROM outs LEFT JOIN spends USING (txh, vout)
),
reveal AS (
  SELECT addr,
         MIN(CASE WHEN type = 'pubkey' THEN t_create ELSE t_spend END) AS t_rev
  FROM life
  WHERE addr IS NOT NULL
  GROUP BY addr
),
iv AS (
  SELECT
    l.addr, l.fam, l.sat, l.t_create, l.t_spend,
    TIMESTAMP_ADD(l.t_create, INTERVAL @d_min_days DAY) AS t_dorm,
    CASE WHEN l.fam IN ('p2pk', 'p2tr') THEN l.t_create ELSE r.t_rev END AS t_exp,
    r.t_rev,
    CASE
      WHEN EXTRACT(YEAR FROM l.t_create) <= 2010 THEN 'e2009_10'
      WHEN EXTRACT(YEAR FROM l.t_create) <= 2012 THEN 'e2011_12'
      WHEN EXTRACT(YEAR FROM l.t_create) <= 2016 THEN 'e2013_16'
      WHEN EXTRACT(YEAR FROM l.t_create) <= 2020 THEN 'e2017_20'
      ELSE 'e2021_'
    END AS era,
    CASE
      WHEN l.sat < 1000000 THEN 'v0_001'
      WHEN l.sat < 10000000 THEN 'v0_01'
      WHEN l.sat < 100000000 THEN 'v0_1'
      WHEN l.sat < 1000000000 THEN 'v1'
      WHEN l.sat < 10000000000 THEN 'v10'
      ELSE 'v100'
    END AS vband
  FROM life AS l LEFT JOIN reveal AS r USING (addr)
),
ivals AS (
  SELECT 'exposed' AS pop, addr, fam, era, vband, sat, t_create, t_spend,
         GREATEST(t_exp, t_dorm) AS a, IFNULL(t_spend, @t_end) AS b
  FROM iv WHERE t_exp IS NOT NULL
  UNION ALL
  SELECT 'control', addr, fam, era, vband, sat, t_create, t_spend,
         t_dorm, LEAST(IFNULL(t_spend, @t_end), IFNULL(t_rev, @t_end))
  FROM iv WHERE fam IN ('p2pkh', 'p2sh')
),
seg AS (
  SELECT pop, addr, fam, era, vband, sat, t_spend, band,
         GREATEST(a, lo) AS a, LEAST(b, hi) AS b
  FROM ivals,
  UNNEST([
    STRUCT('a1_2' AS band, TIMESTAMP_ADD(t_create, INTERVAL 365 DAY) AS lo, TIMESTAMP_ADD(t_create, INTERVAL 730 DAY) AS hi),
    STRUCT('a2_5', TIMESTAMP_ADD(t_create, INTERVAL 730 DAY), TIMESTAMP_ADD(t_create, INTERVAL 1826 DAY)),
    STRUCT('a5_10', TIMESTAMP_ADD(t_create, INTERVAL 1826 DAY), TIMESTAMP_ADD(t_create, INTERVAL 3652 DAY)),
    STRUCT('a10_', TIMESTAMP_ADD(t_create, INTERVAL 3652 DAY), TIMESTAMP '2100-01-01')
  ])
  WHERE a < b AND GREATEST(a, lo) < LEAST(b, hi)
),
deltas AS (
  SELECT pop, fam, era, vband, band, DATE_TRUNC(DATE(a), WEEK(MONDAY)) + 7 AS wk,
         1 AS dn, sat AS dv, CAST(sat AS FLOAT64) * sat AS dv2
  FROM seg
  UNION ALL
  SELECT pop, fam, era, vband, band, DATE_TRUNC(DATE(b), WEEK(MONDAY)) + 7,
         -1, -sat, -CAST(sat AS FLOAT64) * sat
  FROM seg
),
risk AS (
  SELECT pop, fam, era, vband, band, wk,
         SUM(SUM(dn)) OVER w AS n, SUM(SUM(dv)) OVER w AS sv, SUM(SUM(dv2)) OVER w AS sv2
  FROM deltas
  GROUP BY pop, fam, era, vband, band, wk
  WINDOW w AS (PARTITION BY pop, fam, era, vband, band ORDER BY wk)
),
ev AS (
  SELECT pop, addr, fam, era, vband, band, sat,
         DATE_TRUNC(DATE(t_spend), WEEK(MONDAY)) AS wk
  FROM seg
  WHERE t_spend IS NOT NULL AND t_spend >= a AND t_spend <= b
    AND t_spend < @t_end
    AND DATE_TRUNC(DATE(a), WEEK(MONDAY)) < DATE_TRUNC(DATE(t_spend), WEEK(MONDAY))
    AND (pop = 'exposed' OR t_spend = b)
)
SELECT 'risk' AS kind, pop, fam, era, vband, band, wk,
       n AS cnt, sv AS sum_sat, sv2 AS sum_sat2, CAST(NULL AS INT64) AS n_addr,
       CAST(NULL AS STRING) AS addr
FROM risk WHERE n > 0
UNION ALL
SELECT 'event', pop, fam, era, vband, band, wk,
       COUNT(*), SUM(sat), SUM(CAST(sat AS FLOAT64) * sat), COUNT(DISTINCT addr), CAST(NULL AS STRING)
FROM ev GROUP BY pop, fam, era, vband, band, wk
UNION ALL
SELECT 'big', pop, ANY_VALUE(fam), MIN(era), MAX(vband), MAX(band), wk,
       COUNT(*), SUM(sat), CAST(NULL AS FLOAT64), 1, addr
FROM ev WHERE pop = 'exposed'
GROUP BY pop, addr, wk
HAVING SUM(sat) >= @big_btc * 100000000
