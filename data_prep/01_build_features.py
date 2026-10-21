"""Build the shared per-user feature table (data/features.parquet) with DuckDB.

Reads the raw CSVs lazily from DATA_DIR, aggregates per msno, cleans dirty
columns, and keeps a user sample stratified by churn. All mining notebooks read
the output of this script so numbers stay consistent across the team.
"""
import os
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("DATA_DIR", ROOT / "data"))
OUT = Path(os.environ.get("OUT_FILE", DATA / "features.parquet"))
N_PER_CLASS = int(os.environ.get("N_PER_CLASS", 100_000))  # users per churn class

SQL = f"""
COPY (
WITH sample AS (
    SELECT msno, is_churn FROM (
        SELECT *, row_number() OVER (PARTITION BY is_churn ORDER BY hash(msno)) AS rn
        FROM read_csv_auto('{DATA}/train_v2.csv')
    ) WHERE rn <= {N_PER_CLASS}
),
mem AS (
    SELECT msno, city, registered_via,
           CASE WHEN bd BETWEEN 10 AND 80 THEN bd END AS age,   -- bd has 0/negative/>100 junk
           coalesce(gender, 'unknown') AS gender,
           strptime(CAST(registration_init_time AS VARCHAR), '%Y%m%d') AS reg_date
    FROM read_csv_auto('{DATA}/members_v3.csv')
),
tx AS (
    SELECT msno,
           count(*) AS n_tx,
           sum(is_cancel) AS n_cancel,
           avg(is_auto_renew) AS auto_renew_rate,
           mode(payment_method_id) AS payment_method,
           avg(payment_plan_days) AS avg_plan_days,
           avg(actual_amount_paid) AS avg_paid,
           avg(plan_list_price - actual_amount_paid) AS avg_discount
    FROM read_csv_auto('{DATA}/transactions_v2.csv')
    WHERE msno IN (SELECT msno FROM sample)
    GROUP BY msno
),
lg AS (
    SELECT msno,
           count(DISTINCT date) AS active_days,
           sum(num_25 + num_50 + num_75 + num_985 + num_100) AS total_songs,
           sum(num_100) / nullif(sum(num_25 + num_50 + num_75 + num_985 + num_100), 0) AS complete_ratio,
           sum(num_25) / nullif(sum(num_25 + num_50 + num_75 + num_985 + num_100), 0) AS skip_ratio,
           avg(num_unq) AS avg_unique_songs,
           avg(least(total_secs, 86400)) AS avg_secs   -- cap impossible >24h/day values
    FROM read_csv_auto('{DATA}/user_logs_v2.csv')
    WHERE msno IN (SELECT msno FROM sample)
    GROUP BY msno
)
SELECT s.msno, s.is_churn, m.city, m.registered_via, m.age, m.gender, m.reg_date,
       t.* EXCLUDE (msno), l.* EXCLUDE (msno)
FROM sample s
LEFT JOIN mem m USING (msno)
LEFT JOIN tx t USING (msno)
LEFT JOIN lg l USING (msno)
) TO '{OUT}' (FORMAT parquet)
"""

if __name__ == "__main__":
    duckdb.sql(SQL)
    print(duckdb.sql(f"SELECT count(*) n, avg(is_churn) churn_rate FROM '{OUT}'"))
