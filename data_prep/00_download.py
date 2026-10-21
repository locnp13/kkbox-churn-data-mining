"""Download and extract the KKBox files we use (skips the 28 GB user_logs.csv)."""
import subprocess
import sys
from pathlib import Path

import py7zr

COMP = "kkbox-churn-prediction-challenge"
FILES = ["train_v2", "members_v3", "transactions_v2", "user_logs_v2"]
DATA = Path(__file__).resolve().parents[1] / "data"


def main():
    DATA.mkdir(exist_ok=True)
    for name in FILES:
        archive = DATA / f"{name}.csv.7z"
        if not (DATA / f"{name}.csv").exists():
            if not archive.exists():
                subprocess.run(
                    ["kaggle", "competitions", "download", COMP,
                     "-f", f"{name}.csv.7z", "-p", str(DATA)],
                    check=True,
                )
            with py7zr.SevenZipFile(archive) as z:
                z.extractall(DATA)
            archive.unlink()
            # archives nest the csv in subfolders (e.g. data/churn_comp_refresh/); flatten
            for nested in DATA.rglob(f"{name}.csv"):
                if nested.parent != DATA:
                    nested.rename(DATA / nested.name)
        print("ready:", name, file=sys.stderr)


if __name__ == "__main__":
    main()
