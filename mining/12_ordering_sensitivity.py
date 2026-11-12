"""Measure last-transaction sensitivity to within-day ordering for both cohorts."""

import csv

import duckdb

from common import DATA, REPORT, save_metrics


# These are chronological orders: the LAST row is selected. Reversing them
# below reproduces the feature builder's row_number(... DESC) convention.
STANDARD = [
    ("transaction_date", "ASC"), ("membership_expire_date", "ASC"),
    ("is_cancel", "ASC"), ("payment_method_id", "ASC"),
    ("payment_plan_days", "ASC"), ("plan_list_price", "ASC"),
    ("actual_amount_paid", "ASC"), ("is_auto_renew", "ASC"),
]
RULES = {
    "standard": STANDARD,
    # Cancellation is earlier than a payment on the same day, even if expiry differs.
    "cancel_first": [STANDARD[0], ("is_cancel", "DESC")]
                    + [key for key in STANDARD[1:] if key[0] != "is_cancel"],
    "expiry_desc": [STANDARD[0], ("membership_expire_date", "DESC")]
                   + STANDARD[2:],
    "paid_desc": [STANDARD[0], ("actual_amount_paid", "DESC")]
                 + [key for key in STANDARD[1:] if key[0] != "actual_amount_paid"],
}
FIELDS = {
    "is_auto_renew": "is_auto_renew",
    "is_cancel": "is_cancel",
    "payment_plan_days": "CASE WHEN payment_plan_days > 0 THEN payment_plan_days END",
    "actual_amount_paid": "actual_amount_paid",
    "membership_expire_date": "edate",
}
GROUPS = {"last_is_cancel": "is_cancel", "last_auto_renew": "is_auto_renew"}


def sql_path(path):
    """Quote a path as a SQL literal."""
    return "'" + str(path).replace("'", "''") + "'"


def analyze(con, cohort, rows):
    """Use transaction-bearing users as the denominator for every share."""
    def emit(rule, metric, value):
        rows.append((cohort, rule, metric, value))

    # Older days cannot supply a last row under any of the alternative rules.
    con.execute(f"""CREATE OR REPLACE TEMP TABLE latest AS
        SELECT * FROM read_parquet({sql_path(DATA / f'transactions_{cohort}.parquet')})
        QUALIFY tdate = max(tdate) OVER (PARTITION BY msno)""")
    n_users, tied = con.execute("""SELECT count(*), count(*) FILTER (WHERE n > 1)
        FROM (SELECT msno, count(*) AS n FROM latest GROUP BY msno)""").fetchone()
    share_tied = tied / n_users if n_users else None
    emit("standard", "n_users_with_transactions", n_users)
    emit("standard", "n_users_with_same_day_ties", tied)
    emit("standard", "share_users_with_same_day_ties", share_tied)

    max_changed = dict.fromkeys(FIELDS, 0.0 if n_users else None)
    base_rates = {}
    max_diff = None
    for rule, keys in RULES.items():
        reverse = ", ".join(f"{key} {'DESC' if direction == 'ASC' else 'ASC'} NULLS LAST"
                            for key, direction in keys)
        projection = ", ".join(f"{expr} AS {field}" for field, expr in FIELDS.items())
        con.execute(f"""CREATE OR REPLACE TEMP TABLE selected AS
            SELECT msno, is_churn, {projection} FROM latest
            QUALIFY row_number() OVER (PARTITION BY msno ORDER BY {reverse}) = 1""")
        if rule == "standard":
            con.execute("CREATE OR REPLACE TEMP TABLE baseline AS SELECT * FROM selected")
        changes = con.execute("SELECT " + ", ".join(
            f"avg((s.{field} IS DISTINCT FROM b.{field})::INTEGER)" for field in FIELDS
        ) + " FROM selected s JOIN baseline b USING (msno)").fetchone()
        for field, share in zip(FIELDS, changes):
            emit(rule, f"share_changed_{field}", share)
            if share is not None:
                max_changed[field] = max(max_changed[field], share)

        rule_diffs = []
        for label, field in GROUPS.items():
            # Include empty 0/1 groups explicitly; undefined rates remain null.
            stats = {level: (n, rate) for level, n, rate in con.execute(
                f"SELECT {field}, count(*), avg(is_churn) FROM selected GROUP BY 1"
            ).fetchall()}
            levels = sorted(set(stats) - {None})
            levels = sorted(set(levels) | {0, 1}) + ([None] if None in stats else [])
            for level in levels:
                n, rate = stats.get(level, (0, None))
                suffix = f"{label}_{'missing' if level is None else level}"
                emit(rule, f"n_users_{suffix}", n)
                emit(rule, f"churn_rate_{suffix}", rate)
                key = (label, level)
                if rule == "standard":
                    base_rates[key] = rate
                base = base_rates.get(key)
                diff = abs(rate - base) if rate is not None and base is not None else None
                emit(rule, f"abs_churn_rate_diff_{suffix}", diff)
                if diff is not None:
                    rule_diffs.append(diff)
        rule_max = max(rule_diffs, default=None)
        emit(rule, "max_churn_rate_diff", rule_max)
        if rule_max is not None:
            max_diff = max(max_diff or 0.0, rule_max)

    summary = {
        "n_users_with_transactions": n_users,
        "n_users_with_same_day_ties": tied,
        "share_users_with_same_day_ties": share_tied,
        "max_share_changed_per_field": max_changed,
        "max_churn_rate_diff": max_diff,
    }
    print(f"{cohort}: {tied:,}/{n_users:,} users with last-day ties ({share_tied or 0:.4%})")
    print(f"  max shares changed: {max_changed}")
    print(f"  max churn-rate difference vs standard: {max_diff} (fraction)")
    return summary


def main():
    """Write long-form CSV and per-cohort JSON summaries; all rates are fractions."""
    rows = []
    with duckdb.connect() as con:
        summary = {cohort: analyze(con, cohort, rows) for cohort in ("feb", "mar")}
    with (REPORT / "ordering_sensitivity.csv").open("w", newline="", encoding="utf-8") as out:
        writer = csv.writer(out)
        writer.writerow(("cohort", "rule", "metric", "value"))
        writer.writerows(rows)
    save_metrics("ordering_sensitivity", summary)


if __name__ == "__main__":
    main()
