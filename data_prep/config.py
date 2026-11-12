"""Shared configuration for data preparation.

Two prediction cohorts from the KKBox competition give a clean out-of-time design:

* ``feb`` -- ``train.csv``: users whose membership expires in Feb 2017. Observation cutoff
  2017-01-31, so every feature uses data dated <= 2017-01-31.
* ``mar`` -- ``train_v2.csv``: users whose membership expires in Mar 2017. Cutoff 2017-02-28.

Models are fitted on ``feb`` and evaluated on ``mar`` (out-of-time). Descriptive mining
(clustering, rules, sequences) is run on ``mar``, the most recent cohort.
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("DATA_DIR", ROOT / "data"))

LOG_WINDOW_DAYS = 60  # listening history used for behaviour features

COHORTS = {
    "feb": {"labels": "train.csv", "cutoff": "2017-01-31"},
    "mar": {"labels": "train_v2.csv", "cutoff": "2017-02-28"},
}

# raw files from kaggle.com/c/kkbox-churn-prediction-challenge
RAW_FILES = [
    "train", "train_v2", "members_v3",
    "transactions", "transactions_v2",
    "user_logs", "user_logs_v2",  # user_logs (~28 GB) holds history up to 2017-02-28
]
