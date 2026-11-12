"""Generate a tiny synthetic dataset with the KKBox raw schema, for smoke-testing the pipeline.

    python tests/make_synthetic_data.py data_synth      # then DATA_DIR=data_synth python run_all.py

Numbers are random; results on it mean nothing. Only the schema matches the real files.
"""
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

out = Path(sys.argv[1] if len(sys.argv) > 1 else "data_synth")
out.mkdir(exist_ok=True)
rng = np.random.default_rng(0)
N = int(sys.argv[2]) if len(sys.argv) > 2 else 6000
ids = [f"u{i:06d}" for i in range(N)]
ymd = lambda x: int(x.strftime("%Y%m%d"))

pd.DataFrame({
    "msno": ids, "city": rng.integers(1, 22, N), "bd": rng.choice([0, -5, 25, 30, 40, 120], N),
    "gender": rng.choice(["male", "female", None], N), "registered_via": rng.choice([3, 4, 7, 9, 13], N),
    "registration_init_time": [ymd(date(2010, 1, 1) + timedelta(days=int(x))) for x in rng.integers(0, 2500, N)],
}).to_csv(out / "members_v3.csv", index=False)

risk = rng.random(N)                       # latent churn propensity
feb, mar = ids[: N // 2], ids[N // 2:]
churn = (risk + rng.normal(0, 0.25, N) > 0.85).astype(int)
pd.DataFrame({"msno": feb, "is_churn": churn[: N // 2]}).to_csv(out / "train.csv", index=False)
pd.DataFrame({"msno": mar, "is_churn": churn[N // 2:]}).to_csv(out / "train_v2.csv", index=False)

tx, logs = [], []
for i, u in enumerate(ids):
    expire = date(2017, 2, 15) if i < N // 2 else date(2017, 3, 15)
    auto = int(rng.random() > risk[i] * 0.6)
    plan = 30 if rng.random() > 0.1 else 90
    t = expire - timedelta(days=plan)
    for k in range(int(rng.integers(1, 15))):
        price = 149 if plan == 30 else 447
        paid = price if rng.random() > 0.1 else 0
        cancel = int(rng.random() < risk[i] * 0.15)
        p_days = 0 if rng.random() < 0.04 else plan                   # ~4 % records without plan info, as in KKBox
        tx.append((u, rng.choice([36, 38, 39, 41]), p_days, price if p_days else 0, paid, auto, ymd(t),
                   ymd(t + timedelta(days=plan)), cancel))
        t -= timedelta(days=plan)
    if not churn[i]:                       # renewal after expiry (appears only in transactions_v2)
        tx.append((u, 41, 30, 149, 149, auto, ymd(expire), ymd(expire + timedelta(days=30)), 0))
    act = 0.0 if i % 7 == 0 else max(0.05, 0.9 - risk[i] * 0.7)   # ~14 % users without listening logs
    for day in pd.date_range("2016-11-15", "2017-03-31"):
        drop = 0.5 if (churn[i] and day.date() > expire) else 1.0
        if rng.random() < act * drop:
            n = rng.poisson(20, 5)
            logs.append((u, ymd(day.date()), *n, int(n.sum() * 0.8), float(rng.gamma(2, 2000))))

tx = pd.DataFrame(tx, columns=["msno", "payment_method_id", "payment_plan_days", "plan_list_price",
                               "actual_amount_paid", "is_auto_renew", "transaction_date",
                               "membership_expire_date", "is_cancel"])
tx[tx.transaction_date <= 20170228].to_csv(out / "transactions.csv", index=False)
tx[tx.transaction_date >= 20170201].to_csv(out / "transactions_v2.csv", index=False)   # overlaps on purpose
logs = pd.DataFrame(logs, columns=["msno", "date", "num_25", "num_50", "num_75", "num_985", "num_100",
                                   "num_unq", "total_secs"])
logs[logs.date <= 20170228].to_csv(out / "user_logs.csv", index=False)
logs[logs.date >= 20170301].to_csv(out / "user_logs_v2.csv", index=False)
print("synthetic data written to", out, len(tx), "tx", len(logs), "log rows")
