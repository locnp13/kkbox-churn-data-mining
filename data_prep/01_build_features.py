"""Build leakage-safe per-user feature tables, one per cohort.

For every cohort (see config.COHORTS) every feature is computed from data dated on or before
the cohort's observation cutoff T. Nothing from the label month leaks in.

Outputs (in data/):
  tx_cohort.parquet        transactions of cohort users, both raw files merged and de-duplicated
  logs_window.parquet      daily listening logs restricted to both cohorts' windows
  features_<cohort>.parquet  one row per user: label + profile + subscription + listening
  transactions_<cohort>.parquet  per-user transaction history up to T (input to sequence mining)

All users of each cohort are kept (natural churn prior, ~9 %); mining scripts subsample.
"""
import json
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import duckdb

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import COHORTS, DATA, LOG_WINDOW_DAYS


def d(s):
    return date.fromisoformat(s)


def ymd(x: date) -> int:
    return int(x.strftime("%Y%m%d"))


con = duckdb.connect()
con.execute("SET preserve_insertion_order=false")
t0 = time.time()

# ---- cohort users -----------------------------------------------------------------------
con.execute("CREATE TABLE cohort AS " + " UNION ALL ".join(
    f"SELECT msno, is_churn, '{c}' AS cohort FROM read_csv_auto('{DATA}/{v['labels']}')"
    for c, v in COHORTS.items()))
con.execute("CREATE TABLE users AS SELECT DISTINCT msno FROM cohort")
print(con.sql("SELECT cohort, count(*) n, avg(is_churn) churn FROM cohort GROUP BY 1 ORDER BY 1"))

# ---- transactions: merge the two raw files, de-duplicate, keep cohort users ----------------
TX_COLS = ("msno, payment_method_id, payment_plan_days, plan_list_price, actual_amount_paid, "
           "is_auto_renew, transaction_date, membership_expire_date, is_cancel")
if not (DATA / "tx_cohort.parquet").exists():
    con.execute(f"""
    COPY (
      SELECT DISTINCT {TX_COLS} FROM (
        SELECT {TX_COLS} FROM read_csv_auto('{DATA}/transactions.csv')
        UNION ALL
        SELECT {TX_COLS} FROM read_csv_auto('{DATA}/transactions_v2.csv')
      ) WHERE msno IN (SELECT msno FROM users)
    ) TO '{DATA}/tx_cohort.parquet' (FORMAT parquet)""")
    print(f"tx_cohort done {time.time() - t0:.0f}s")

# ---- logs: one scan of the 28 GB file, keep only the union of both windows -----------------
starts = {c: d(v["cutoff"]) - timedelta(days=LOG_WINDOW_DAYS - 1) for c, v in COHORTS.items()}
lo, hi = ymd(min(starts.values())), max(ymd(d(v["cutoff"])) for v in COHORTS.values())
if not (DATA / "logs_window.parquet").exists() and (DATA / "user_logs.csv").exists():
    con.execute(f"""
    COPY (
      SELECT msno, date, num_25, num_50, num_75, num_985, num_100, num_unq,
             least(greatest(total_secs, 0), 86400) AS total_secs  -- impossible >24h/day values
      FROM read_csv_auto('{DATA}/user_logs.csv')
      WHERE date BETWEEN {lo} AND {hi} AND msno IN (SELECT msno FROM users)
    ) TO '{DATA}/logs_window.parquet' (FORMAT parquet)""")
    print(f"logs_window done {time.time() - t0:.0f}s")
HAS_LOGS = (DATA / "logs_window.parquet").exists()
if not HAS_LOGS:
    print("WARNING: user_logs.csv not available -> listening features will be empty")

con.execute(f"""
CREATE TABLE mem AS
SELECT msno, city, registered_via,
       CASE WHEN bd BETWEEN 10 AND 80 THEN bd END AS age,     -- bd has 0 / negative / >100 junk
       coalesce(gender, 'unknown') AS gender,
       strptime(CAST(registration_init_time AS VARCHAR), '%Y%m%d')::DATE AS reg_date
FROM read_csv_auto('{DATA}/members_v3.csv')
WHERE msno IN (SELECT msno FROM users)""")


ORDER_ASC = ("transaction_date, membership_expire_date, is_cancel, payment_method_id, payment_plan_days, "
             "plan_list_price, actual_amount_paid, is_auto_renew")
ORDER_DESC = ", ".join(c + " DESC" for c in ORDER_ASC.split(", "))
LOG_COLS = ["active_days", "active_days_last14", "total_songs", "complete_ratio", "skip_ratio",
            "avg_unique_songs", "total_hours", "mins_per_active_day", "listen_trend", "days_since_last_listen"]


def build(cohort: str, cutoff: date, start: date):
    T, Ti, Si = f"DATE '{cutoff}'", ymd(cutoff), ymd(start)
    half = ymd(start + timedelta(days=LOG_WINDOW_DAYS // 2))
    last14 = ymd(cutoff - timedelta(days=13))
    K = LOG_WINDOW_DAYS // 7
    con.execute(f"""
    CREATE OR REPLACE TEMP VIEW tx AS
    SELECT t.*,
           strptime(CAST(transaction_date AS VARCHAR), '%Y%m%d')::DATE AS tdate,
           strptime(CAST(membership_expire_date AS VARCHAR), '%Y%m%d')::DATE AS edate
    FROM '{DATA}/tx_cohort.parquet' t
    JOIN cohort c ON c.msno = t.msno AND c.cohort = '{cohort}'
    WHERE transaction_date <= {Ti}""")
    con.execute(f"""
    COPY (SELECT tx.*, c.is_churn FROM tx JOIN cohort c ON c.msno = tx.msno AND c.cohort = '{cohort}'
          ORDER BY tx.msno, transaction_date, membership_expire_date)
    TO '{DATA}/transactions_{cohort}.parquet' (FORMAT parquet)""")

    logs_sql = f"""
    lg AS (
      SELECT msno,
        count(DISTINCT date) AS active_days,
        count(DISTINCT CASE WHEN date >= {last14} THEN date END) AS active_days_last14,
        sum(num_25 + num_50 + num_75 + num_985 + num_100) AS total_songs,
        sum(num_100) / nullif(sum(num_25 + num_50 + num_75 + num_985 + num_100), 0) AS complete_ratio,
        sum(num_25) / nullif(sum(num_25 + num_50 + num_75 + num_985 + num_100), 0) AS skip_ratio,
        avg(num_unq) AS avg_unique_songs,
        sum(total_secs) / 3600.0 AS total_hours,
        sum(total_secs) / count(DISTINCT date) / 60.0 AS mins_per_active_day,
        ln(1 + sum(CASE WHEN date >= {half} THEN total_secs ELSE 0 END))
          - ln(1 + sum(CASE WHEN date < {half} THEN total_secs ELSE 0 END)) AS listen_trend,
        {T} - max(strptime(CAST(date AS VARCHAR), '%Y%m%d')::DATE) AS days_since_last_listen
      FROM '{DATA}/logs_window.parquet'
      WHERE date BETWEEN {Si} AND {Ti}
        AND msno IN (SELECT msno FROM cohort WHERE cohort = '{cohort}')
      GROUP BY msno),
    wk AS (   -- regularity: coefficient of variation of active days over the K full weeks
      SELECT msno,                 -- weeks without activity count as 0 (via sums over K)
        sqrt(greatest(sum(n * n) / {K} - pow(sum(n) / {K}, 2), 0)) / (sum(n) / {K}) AS weekly_cv
      FROM (
        SELECT msno, ({T} - strptime(CAST(date AS VARCHAR), '%Y%m%d')::DATE) // 7 AS w,  -- 0 = last 7 days
               count(DISTINCT date) AS n
        FROM '{DATA}/logs_window.parquet'
        WHERE date BETWEEN {Si} AND {Ti}
          AND msno IN (SELECT msno FROM cohort WHERE cohort = '{cohort}')
        GROUP BY 1, 2)
      WHERE w < {K}
      GROUP BY msno)""" if HAS_LOGS else """
    lg AS (SELECT NULL::VARCHAR AS msno, """ + ", ".join(f"NULL::DOUBLE AS {c}" for c in LOG_COLS) + """ WHERE false),
    wk AS (SELECT NULL::VARCHAR AS msno, NULL::DOUBLE AS weekly_cv WHERE false)"""

    con.execute(f"""
    COPY (
    WITH clean AS (
      -- data-quality rules: payment_plan_days = 0 (with list price 0) are old records without plan
      -- information, not real plans; expiry dates before 2015 are corrupt
      SELECT *, CASE WHEN payment_plan_days > 0 THEN payment_plan_days END AS plan_days_valid,
                CASE WHEN plan_list_price > 0 THEN plan_list_price END AS list_price_valid,
                CASE WHEN edate >= DATE '2015-01-01' THEN edate END AS edate_valid
      FROM tx),
    ranked AS (
      -- one canonical event order (same as data_prep/02_build_sequences.py): date, expiry,
      -- cancel last on ties, so the last row is the latest known state at T
      SELECT *, row_number() OVER (PARTITION BY msno ORDER BY {ORDER_DESC}) AS rn,
             lag(plan_days_valid IGNORE NULLS) OVER (PARTITION BY msno ORDER BY {ORDER_ASC}) AS prev_plan
      FROM clean),
    agg AS (
      SELECT msno,
        count(*) AS n_tx,
        count(*) FILTER (WHERE tdate > {T} - INTERVAL 180 DAY) AS n_tx_180d,
        sum(is_cancel) AS n_cancel,
        avg(is_auto_renew) AS auto_renew_rate,
        avg(CASE WHEN list_price_valid IS NOT NULL THEN (actual_amount_paid < list_price_valid)::INT END) AS discount_rate,
        sum(CASE WHEN actual_amount_paid = 0 THEN 1 ELSE 0 END) AS n_free_tx,
        sum(CASE WHEN plan_days_valid IS NOT NULL AND prev_plan IS NOT NULL AND prev_plan <> plan_days_valid
                 THEN 1 ELSE 0 END) AS n_plan_changes,           -- only between known plans
        count(DISTINCT payment_method_id) AS n_payment_methods,
        sum(actual_amount_paid) AS total_paid,
        {T} - min(tdate) AS days_since_first_tx
      FROM ranked GROUP BY msno),
    last AS (
      SELECT msno, payment_method_id AS last_payment_method, plan_days_valid AS last_plan_days,
             list_price_valid AS last_list_price, actual_amount_paid AS last_paid,
             is_auto_renew AS last_auto_renew, is_cancel AS last_is_cancel,
             {T} - tdate AS days_since_last_tx,
             edate_valid - {T} AS days_to_expire  -- expiry known at T (may lie after T); corrupt -> NULL
      FROM ranked WHERE rn = 1),
    {logs_sql}
    SELECT c.msno, c.is_churn, m.city, m.registered_via, m.gender, m.age,
           CASE WHEN m.reg_date <= {T} THEN {T} - m.reg_date END AS tenure_days,  -- registered after T -> NULL
           (m.msno IS NOT NULL)::INT AS has_member,
           a.* EXCLUDE (msno), l.* EXCLUDE (msno),
           g.* EXCLUDE (msno), w.weekly_cv,
           (g.msno IS NOT NULL)::INT AS has_logs, (a.msno IS NOT NULL)::INT AS has_tx
    FROM (SELECT * FROM cohort WHERE cohort = '{cohort}') c
    LEFT JOIN mem m USING (msno)
    LEFT JOIN agg a USING (msno)
    LEFT JOIN last l USING (msno)
    LEFT JOIN lg g USING (msno)
    LEFT JOIN wk w USING (msno)
    ) TO '{DATA}/features_{cohort}.parquet' (FORMAT parquet)""")
    print(con.sql(f"""SELECT '{cohort}' cohort, count(*) n, avg(is_churn) churn, avg(has_logs) has_logs,
                     avg(has_tx) has_tx FROM '{DATA}/features_{cohort}.parquet'"""))


manifest = {"log_window_days": LOG_WINDOW_DAYS, "has_listening_logs": HAS_LOGS, "cohorts": {}}
for c, v in COHORTS.items():
    build(c, d(v["cutoff"]), starts[c])
    f = f"{DATA}/features_{c}.parquet"
    n, n_unique = con.execute(f"SELECT count(*), count(DISTINCT msno) FROM '{f}'").fetchone()
    assert n == n_unique, f"{c}: duplicate users in feature table"
    max_tx = con.execute(f"SELECT max(transaction_date) FROM '{DATA}/transactions_{c}.parquet'").fetchone()[0]
    assert max_tx is None or max_tx <= int(v["cutoff"].replace("-", "")), f"{c}: transaction after cutoff"
    manifest["cohorts"][c] = {"labels": v["labels"], "cutoff": v["cutoff"], "log_window_start": str(starts[c]),
                              "n_users": n, "max_transaction_date": max_tx,
                              **dict(zip(["churn_rate", "share_has_logs", "share_has_tx"], con.execute(
                                  f"SELECT avg(is_churn), avg(has_logs), avg(has_tx) FROM '{f}'").fetchone()))}
    print(f"{c} done {time.time() - t0:.0f}s")
manifest["data_quality"] = dict(zip(
    ["tx_rows", "tx_plan_days_zero", "tx_expiry_before_2015", "tx_expiry_before_tx_date"], con.execute(f"""
    SELECT count(*), count(*) FILTER (WHERE payment_plan_days = 0),
           count(*) FILTER (WHERE membership_expire_date < 20150101),
           count(*) FILTER (WHERE membership_expire_date < transaction_date)
    FROM '{DATA}/tx_cohort.parquet'""").fetchone()))
manifest["data_quality"]["users_registered_after_cutoff"] = {
    c: con.execute(f"SELECT count(*) FROM mem m JOIN cohort k USING (msno) WHERE k.cohort = '{c}' "
                   f"AND m.reg_date > DATE '{v['cutoff']}'").fetchone()[0] for c, v in COHORTS.items()}
(DATA / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
