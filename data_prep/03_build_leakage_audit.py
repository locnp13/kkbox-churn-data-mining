"""Features for the leakage audit (mar cohort only): what the first version of the project used.

The first pipeline aggregated transactions_v2 and user_logs_v2 without any date filter. Both
files run to 2017-03-31, i.e. into the label month. This script rebuilds those aggregates so
mining/06_validation.py can measure, on the same users and split, how much each post-cutoff
source inflates model performance and rule confidence.

Output: data/leakage_mar.parquet with columns prefixed
  post_tx_*   transactions dated after the cutoff (renewals happening in the label month)
  post_log_*  listening in March 2017 (user_logs_v2)
  v1_*        the original v1 recipe (unfiltered transactions_v2 / user_logs_v2 aggregates)
"""
import sys
from pathlib import Path

import duckdb

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import COHORTS, DATA

T = int(COHORTS["mar"]["cutoff"].replace("-", ""))
LBL = f"read_csv_auto('{DATA}/{COHORTS['mar']['labels']}')"

duckdb.sql(f"""
COPY (
WITH u AS (SELECT msno FROM {LBL}),
tx2 AS (SELECT * FROM read_csv_auto('{DATA}/transactions_v2.csv') WHERE msno IN (SELECT msno FROM u)),
lg2 AS (SELECT * FROM read_csv_auto('{DATA}/user_logs_v2.csv') WHERE msno IN (SELECT msno FROM u)),
post_tx AS (
  SELECT msno, count(*) AS post_tx_n, max(is_cancel) AS post_tx_any_cancel,
         avg(is_auto_renew) AS post_tx_auto_renew
  FROM tx2 WHERE transaction_date > {T} GROUP BY msno),
v1_tx AS (
  SELECT msno, count(*) AS v1_n_tx, sum(is_cancel) AS v1_n_cancel,
         avg(is_auto_renew) AS v1_auto_renew_rate, avg(payment_plan_days) AS v1_avg_plan_days,
         avg(actual_amount_paid) AS v1_avg_paid
  FROM tx2 GROUP BY msno),
lg AS (
  SELECT msno, count(DISTINCT date) AS post_log_active_days,
         sum(least(greatest(total_secs, 0), 86400)) / 3600.0 AS post_log_hours,
         31 - max(date % 100) AS post_log_days_idle
  FROM lg2 GROUP BY msno)
SELECT u.msno,
       coalesce(p.post_tx_n, 0) AS post_tx_n, coalesce(p.post_tx_any_cancel, 0) AS post_tx_any_cancel,
       p.post_tx_auto_renew,
       coalesce(v.v1_n_tx, 0) AS v1_n_tx, coalesce(v.v1_n_cancel, 0) AS v1_n_cancel,
       v.v1_auto_renew_rate, v.v1_avg_plan_days, v.v1_avg_paid,
       coalesce(l.post_log_active_days, 0) AS post_log_active_days,
       coalesce(l.post_log_hours, 0) AS post_log_hours,
       coalesce(l.post_log_days_idle, 31) AS post_log_days_idle
FROM u LEFT JOIN post_tx p USING (msno) LEFT JOIN v1_tx v USING (msno) LEFT JOIN lg l USING (msno)
) TO '{DATA}/leakage_mar.parquet' (FORMAT parquet)""")
print(duckdb.sql(f"SELECT count(*) n, avg((post_tx_n > 0)::INT) share_with_post_tx FROM '{DATA}/leakage_mar.parquet'"))
