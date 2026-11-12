"""Build up to twelve transaction itemsets per cohort user.

An itemset preserves simultaneous business facts without inventing a token order
inside a transaction. Separate itemsets represent strictly later transaction rows.
PAY_AUTO/PAY_MANUAL, FREE and DISC describe non-cancellation transactions; FREE
and DISC may coexist; DISC requires a positive list price. PLAN_30, PLAN_LONG
and PLAN_SHORT describe positive plan durations; zero means an unknown plan.
PLAN_CHANGE compares a positive duration with the most recent positive duration,
skipping unknown plans. METHOD_CHANGE compares adjacent rows. GAP means the new
transaction is over 30 days after the most recent valid expiry; expiries before
2015-01-01 are treated as missing. Comparisons use full history before truncation.
Users without transactions have an empty sequence, not an artificial event.
"""

import os
import time

import duckdb

from config import COHORTS, DATA


def sql_path(path):
    """Quote a filesystem path as a SQL string literal."""
    return "'" + str(path).replace("'", "''") + "'"


def build(con, cohort):
    """Tokenize full history in SQL and retain the last twelve events."""
    source = DATA / f"transactions_{cohort}.parquet"
    features = DATA / f"features_{cohort}.parquet"
    labels = DATA / COHORTS[cohort]["labels"]
    if features.exists():
        users = f"SELECT msno, is_churn FROM read_parquet({sql_path(features)})"
    elif labels.exists():
        users = f"SELECT msno, is_churn FROM read_csv_auto({sql_path(labels)})"
    else:
        users = f"SELECT DISTINCT msno, is_churn FROM read_parquet({sql_path(source)})"
        print(f"WARNING {cohort}: no label roster; users without transactions are unknown")
    con.execute(f"CREATE OR REPLACE TEMP TABLE users AS {users}")
    invalid = con.sql("""SELECT count(*) FROM (
        SELECT msno FROM users GROUP BY msno
        HAVING count(*) <> 1 OR count(is_churn) <> 1
            OR min(is_churn) NOT IN (0, 1) OR msno IS NULL)""").fetchall()[0][0]
    if invalid:
        raise ValueError(f"{cohort}: invalid or duplicate user labels")
    cutoff = int(COHORTS[cohort]["cutoff"].replace("-", ""))
    # Extra keys make ties deterministic without changing the primary ordering.
    ordering = ("transaction_date, membership_expire_date, is_cancel, "
                "payment_method_id, payment_plan_days, plan_list_price, "
                "actual_amount_paid, is_auto_renew")
    output = DATA / f"sequences_{cohort}.parquet"
    con.execute(f"""
        COPY (
          WITH history AS (
            SELECT *, row_number() OVER w AS event_no,
                   count(*) OVER (PARTITION BY msno) AS n_events,
                   lag(CASE WHEN edate >= DATE '2015-01-01' THEN edate END
                       IGNORE NULLS) OVER w AS prev_expiry,
                   lag(CASE WHEN payment_plan_days > 0 THEN payment_plan_days END
                       IGNORE NULLS) OVER w AS prev_plan,
                   lag(payment_method_id) OVER w AS prev_method
            FROM read_parquet({sql_path(source)})
            WHERE transaction_date <= {cutoff}
            WINDOW w AS (PARTITION BY msno ORDER BY {ordering})
          ), events AS (
            SELECT msno, event_no, list_sort([token FOR token IN [
              CASE WHEN is_cancel = 1 THEN 'CANCEL'
                   WHEN is_auto_renew = 1 THEN 'PAY_AUTO' ELSE 'PAY_MANUAL' END,
              CASE WHEN coalesce(is_cancel, 0) = 0 AND actual_amount_paid = 0 THEN 'FREE' END,
              CASE WHEN coalesce(is_cancel, 0) = 0 AND plan_list_price > 0
                        AND actual_amount_paid < plan_list_price
                   THEN 'DISC' END,
              CASE WHEN payment_plan_days = 30 THEN 'PLAN_30'
                   WHEN payment_plan_days > 30 THEN 'PLAN_LONG'
                   WHEN payment_plan_days > 0 THEN 'PLAN_SHORT' END,
              CASE WHEN payment_plan_days > 0 AND prev_plan <> payment_plan_days
                   THEN 'PLAN_CHANGE' END,
              CASE WHEN tdate - prev_expiry > 30 THEN 'GAP' END,
              CASE WHEN prev_method <> payment_method_id THEN 'METHOD_CHANGE' END
            ] IF token IS NOT NULL]) AS tokens
            FROM history WHERE event_no > n_events - 12
          ), sequences AS (
            SELECT msno, list(tokens ORDER BY event_no) AS seq
            FROM events GROUP BY msno
          )
          SELECT u.msno, CAST(u.is_churn AS INTEGER) AS is_churn,
                 coalesce(s.seq, []::VARCHAR[][]) AS seq
          FROM users u LEFT JOIN sequences s USING (msno)
          ORDER BY u.msno
        ) TO {sql_path(output)} (FORMAT PARQUET)
    """)
    stats = con.sql(f"""SELECT count(*), count(*) FILTER (WHERE len(seq) = 0)
                        FROM read_parquet({sql_path(output)})""").fetchall()[0]
    print(f"{cohort}: {stats[0]:,} users, {stats[1]:,} empty sequences -> {output}")


def main():
    """Build both cohorts, or the cohort explicitly selected by COHORT."""
    started = time.perf_counter()
    cohorts = [os.environ["COHORT"]] if os.environ.get("COHORT") else list(COHORTS)
    with duckdb.connect() as con:
        for cohort in cohorts:
            if cohort not in COHORTS:
                raise ValueError(f"Unknown COHORT: {cohort}")
            build(con, cohort)
    print(f"Sequences built in {time.perf_counter() - started:.2f}s")


if __name__ == "__main__":
    main()
