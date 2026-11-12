"""Download and extract the KKBox competition files into data/.

Needs the Kaggle CLI with a token (~/.kaggle/kaggle.json or ~/.kaggle/access_token) and the
competition rules accepted on kaggle.com. ``user_logs.csv.7z`` (~7 GB, ~28 GB extracted) is
never extracted: stream_logs.py filters it while decompressing into data/logs_window.parquet,
so the disk only needs room for the archive. Set SKIP_BIG_LOGS=1 to skip it entirely.
Extraction prefers the 7z executable when available (much faster than py7zr).
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import DATA, RAW_FILES

COMP = "kkbox-churn-prediction-challenge"
SEVEN_ZIP = shutil.which("7z") or next(
    (p for p in [r"C:\Program Files\7-Zip\7z.exe"] if Path(p).exists()), None)


def extract(archive: Path):
    if SEVEN_ZIP:
        subprocess.run([SEVEN_ZIP, "x", "-y", f"-o{DATA}", str(archive)], check=True,
                       stdout=subprocess.DEVNULL)
    else:
        import py7zr
        with py7zr.SevenZipFile(archive) as z:
            z.extractall(DATA)


def unpack_zip(zip_path: Path):
    """Take the needed .7z archives out of Kaggle's "Download All" zip (no re-download)."""
    import zipfile
    with zipfile.ZipFile(zip_path) as z:
        for info in z.infolist():
            name = Path(info.filename).name
            if name.endswith(".csv.7z") and name[:-7] in RAW_FILES and not (DATA / name).exists():
                print("unzipping", name, file=sys.stderr)
                with z.open(info) as src, open(DATA / name, "wb") as dst:
                    shutil.copyfileobj(src, dst, 16 << 20)


def main():
    DATA.mkdir(exist_ok=True)
    if len(sys.argv) > 1:          # python data_prep/00_download.py path/to/kkbox-churn-prediction-challenge.zip
        unpack_zip(Path(sys.argv[1]))
    files = [f for f in RAW_FILES if not (f == "user_logs" and os.environ.get("SKIP_BIG_LOGS"))]
    for name in files:
        if name == "user_logs" and (DATA / "logs_window.parquet").exists():
            print("ready: user_logs (logs_window.parquet)", file=sys.stderr)
            continue
        if (DATA / f"{name}.csv").exists():
            print("ready:", name, file=sys.stderr)
            continue
        archive = DATA / f"{name}.csv.7z"
        if not archive.exists():
            subprocess.run(["kaggle", "competitions", "download", COMP,
                            "-f", f"{name}.csv.7z", "-p", str(DATA)], check=True)
        if name == "user_logs":          # filter while decompressing, never write the 28 GB csv
            from stream_logs import stream
            stream(archive)
            archive.unlink()
            print("ready:", name, file=sys.stderr)
            continue
        extract(archive)
        archive.unlink()
        # some archives nest the csv in a subfolder (e.g. data/churn_comp_refresh/); flatten
        for nested in DATA.rglob(f"{name}.csv"):
            if nested.parent != DATA:
                nested.replace(DATA / nested.name)
        print("ready:", name, file=sys.stderr)


if __name__ == "__main__":
    main()
