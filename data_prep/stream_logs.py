"""Stream-extract user_logs.csv.7z without ever writing the 28 GB CSV to disk.

The archive is decompressed chunk by chunk; each chunk is parsed with pyarrow, filtered to the
listening windows of both cohorts and to cohort users, and appended to
data/logs_window.parquet (a few hundred MB). Disk needed: the ~7 GB archive only.

    python data_prep/stream_logs.py [path/to/user_logs.csv.7z]
"""
import io
import json
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pcsv
import pyarrow.parquet as pq
import py7zr
from py7zr.io import Py7zIO, WriterFactory

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import COHORTS, DATA, LOG_WINDOW_DAYS

COLS = ["msno", "date", "num_25", "num_50", "num_75", "num_985", "num_100", "num_unq", "total_secs"]
TYPES = {"msno": pa.string(), "date": pa.int64(), **{c: pa.int64() for c in COLS[2:-1]},
         "total_secs": pa.float64()}
CHUNK = 64 << 20

starts = [date.fromisoformat(v["cutoff"]) - timedelta(days=LOG_WINDOW_DAYS - 1) for v in COHORTS.values()]
LO = int(min(starts).strftime("%Y%m%d"))
HI = max(int(v["cutoff"].replace("-", "")) for v in COHORTS.values())


def cohort_users():
    ids = []
    for v in COHORTS.values():
        ids.append(pcsv.read_csv(DATA / v["labels"]).column("msno"))
    return pa.chunked_array(ids).unique()


class FilterWriter(Py7zIO):
    def __init__(self, out_path, users):
        self.buf, self.header_done, self.users = bytearray(), False, users
        self.writer, self.out_path = None, out_path
        self.rows_in = self.rows_out = self.bytes = 0
        self.t0 = time.time()

    def _flush_block(self, block: bytes):
        tbl = pcsv.read_csv(io.BytesIO(block),
                            read_options=pcsv.ReadOptions(column_names=COLS),
                            convert_options=pcsv.ConvertOptions(column_types=TYPES))
        self.rows_in += tbl.num_rows
        m = pc.and_(pc.and_(pc.greater_equal(tbl["date"], LO), pc.less_equal(tbl["date"], HI)),
                    pc.is_in(tbl["msno"], value_set=self.users))
        tbl = tbl.filter(m)
        if tbl.num_rows:
            secs = pc.min_element_wise(pc.max_element_wise(tbl["total_secs"], 0.0), 86400.0)
            tbl = tbl.set_column(COLS.index("total_secs"), "total_secs", secs)
            if self.writer is None:
                self.writer = pq.ParquetWriter(self.out_path, tbl.schema)
            self.writer.write_table(tbl)
            self.rows_out += tbl.num_rows

    def write(self, s):
        self.buf += s
        self.bytes += len(s)
        if not self.header_done and b"\n" in self.buf:
            self.buf = self.buf[self.buf.index(b"\n") + 1:]
            self.header_done = True
        if len(self.buf) >= CHUNK:
            cut = self.buf.rindex(b"\n") + 1
            self._flush_block(bytes(self.buf[:cut]))
            del self.buf[:cut]
            print(f"\r  {self.bytes / 2**30:6.1f} GB read, {self.rows_in / 1e6:7.1f}M rows, "
                  f"{self.rows_out / 1e6:6.1f}M kept, {time.time() - self.t0:5.0f}s", end="", flush=True)
        return len(s)

    def close(self):
        if self.buf.strip():
            self._flush_block(bytes(self.buf))
        self.buf = bytearray()
        if self.writer:
            self.writer.close()
        print(f"\n  done: {self.rows_in:,} rows read, {self.rows_out:,} kept -> {self.out_path}")
        (DATA / "logs_stream_stats.json").write_text(json.dumps(
            {"rows_scanned": self.rows_in, "rows_kept": self.rows_out, "date_range": [LO, HI]}))

    def read(self, size=None):
        return b""

    def seek(self, offset, whence=0):
        return 0

    def flush(self):
        pass

    def size(self):
        return self.bytes


class Factory(WriterFactory):
    def __init__(self, out_path, users):
        self.out_path, self.users = out_path, users

    def create(self, filename):
        return FilterWriter(self.out_path, self.users)


def stream(archive: Path, out_path: Path = DATA / "logs_window.parquet"):
    tmp = out_path.with_suffix(".partial")
    print(f"streaming {archive.name}: keep dates {LO}..{HI} for cohort users")
    with py7zr.SevenZipFile(archive) as z:
        z.extract(factory=Factory(tmp, cohort_users()))
    tmp.replace(out_path)


if __name__ == "__main__":
    stream(Path(sys.argv[1]) if len(sys.argv) > 1 else DATA / "user_logs.csv.7z")
