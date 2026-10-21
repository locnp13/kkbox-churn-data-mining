"""Shared loader: reads the feature table produced by data_prep/01_build_features.py."""
import os
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FEATURES = Path(os.environ.get("FEATURES_FILE", ROOT / "data" / "features.parquet"))
FIG = Path(os.environ.get("FIG_DIR", ROOT / "report" / "figures"))
FIG.mkdir(parents=True, exist_ok=True)

NUM = ["age", "n_tx", "n_cancel", "auto_renew_rate", "avg_plan_days", "avg_paid",
       "avg_discount", "active_days", "total_songs", "complete_ratio", "skip_ratio",
       "avg_unique_songs", "avg_secs"]


def load():
    df = pd.read_parquet(FEATURES)
    df["tenure_days"] = (pd.Timestamp("2017-03-01") - df["reg_date"]).dt.days
    # users without logs/transactions are a real segment: keep a flag, fill with 0
    df["has_logs"] = df["active_days"].notna().astype(int)
    for c in NUM:
        df[c] = df[c].fillna(df[c].median() if c == "age" else 0)
    return df
