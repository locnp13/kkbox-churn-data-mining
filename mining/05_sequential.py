"""Mine discriminative transaction-itemset sequences with pure NumPy PrefixSpan.

Patterns contain sorted itemsets: {FREE, PAY_AUTO} -> {CANCEL} requires the
first two facts in one transaction and cancellation in a later transaction.
Length counts tokens, capped at four, not elapsed time or transaction count.
The projected database stores ALL possible ending positions as 12-bit masks;
keeping only the earliest ending would lose valid same-itemset extensions.
Support counts distinct users, including empty histories in the denominator.

A deterministic label-stratified sample (SEQ_SAMPLE=300000; 0 means all) is
split 70/30 with seed 42. Mining, Pearson chi-square tests (1 df), BH correction,
ranking and redundancy pruning use training users only. BH covers every pattern
mined at the lowest sensitivity/requested threshold, including neutral patterns.
Sensitivity reuses that family and its q-values; no holdout-based selection.
Redundancy requires a proper sequential subpattern in the same direction with
at least 95% of the longer pattern's log-lift magnitude. After pruning, SEQ_TOP=50
reserves 25 reported patterns per direction (odd limits give churn one extra).
Churn ranks by decreasing training lift, then support; retain ranks by decreasing
absolute training leverage, proportional to support_train * (1 - lift_train),
then increasing lift and decreasing support. This favors broad protective
patterns without letting churn crowd them out. Both directions keep the same
minimum support, q < 0.01 and lift > 1.2 / < 0.8 filters; unused slots are not
transferred between directions. Holdout is descriptive replication, not a test
used for filtering; zero matches yield null confidence/lift.
Reported patterns are also evaluated on an independently sampled February cohort,
using its own base rate. Temporal correlation compares training and February
log-lifts on pairs with positive lift; undefined correlations are null. Direction
agreement uses all reported patterns (missing February matches do not agree).
"""

import json
import math
import os
import time
from pathlib import Path

import duckdb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
MAX_LENGTH = 4
COLUMNS = ["pattern", "len", "support_train", "conf_train", "lift_train", "q_value",
           "support_holdout", "conf_holdout", "lift_holdout",
           "support_feb", "conf_feb", "lift_feb", "direction"]


def stratified_indices(labels, size, rng):
    """Allocate a fixed-size sample proportionally across binary classes."""
    size = min(size, len(labels))
    n_positive = int(round(size * float(labels.mean())))
    chosen = []
    for value, count in ((0, size - n_positive), (1, n_positive)):
        pool = np.flatnonzero(labels == value)
        chosen.append(rng.permutation(pool)[:count])
    return rng.permutation(np.concatenate(chosen))


def split_indices(labels, rng):
    """Keep both classes in each split whenever the input permits it."""
    train, holdout = [], []
    for value in (0, 1):
        pool = rng.permutation(np.flatnonzero(labels == value))
        if len(pool) < 2:
            raise ValueError("Each class needs at least two sampled users for a stratified split")
        n_train = min(len(pool) - 1, max(1, int(round(0.7 * len(pool)))))
        train.extend(pool[:n_train])
        holdout.extend(pool[n_train:])
    return np.asarray(train), np.asarray(holdout)


def encode(sequences, vocabulary=None):
    """Build one uint16 occurrence mask per user and token (at most 12 bits)."""
    if vocabulary is None:
        vocabulary = sorted({str(token) for seq in sequences for event in seq for token in event})
    mapping = {token: i for i, token in enumerate(vocabulary)}
    masks = np.zeros((len(vocabulary), len(sequences)), dtype=np.uint16)
    for user, seq in enumerate(sequences):
        if len(seq) > 12:
            raise ValueError("Input has more than 12 events; rebuild sequences first")
        for pos, event in enumerate(seq):
            if isinstance(event, str):
                raise ValueError("Expected nested itemsets, not a flat token sequence")
            for token in event:
                if str(token) in mapping:
                    masks[mapping[str(token)], user] |= np.uint16(1 << pos)
    return vocabulary, masks


def suffix_mask(ends):
    """Return positions strictly after the earliest possible pattern ending."""
    bits = ends.astype(np.uint32)
    first = bits & (~bits + np.uint32(1))
    return ((~(2 * first - 1)) & 4095).astype(np.uint16)


def mine(masks, labels, minimum):
    """PrefixSpan with canonical itemset and sequence extensions, user support."""
    n_users = len(labels)
    threshold = max(1, math.ceil(minimum * n_users - 1e-12))
    frequent = [i for i, row in enumerate(masks) if np.count_nonzero(row) >= threshold]
    records = []

    def visit(pattern, users, ends, length):
        records.append({"key": pattern, "len": length, "count": len(users),
                        "churn_count": int(labels[users].sum())})
        if length == MAX_LENGTH:
            return
        after = suffix_mask(ends)
        for token in frequent:
            occurrences = masks[token, users]
            # Lexicographic order gives each within-event itemset exactly one path.
            if token > pattern[-1][-1]:
                new_ends = ends & occurrences
                keep = np.flatnonzero(new_ends)
                if len(keep) >= threshold:
                    extended = pattern[:-1] + (pattern[-1] + (token,),)
                    visit(extended, users[keep], new_ends[keep], length + 1)
            new_ends = after & occurrences
            keep = np.flatnonzero(new_ends)
            if len(keep) >= threshold:
                visit(pattern + ((token,),), users[keep], new_ends[keep], length + 1)

    for token in frequent:
        users = np.flatnonzero(masks[token])
        visit(((token,),), users, masks[token, users], 1)
    return records


def add_statistics(records, labels):
    """Pearson chi-square and monotone Benjamini-Hochberg adjusted p-values."""
    total, positives = len(labels), int(labels.sum())
    base = positives / total
    for row in records:
        matched, a = row["count"], row["churn_count"]
        b, c = matched - a, positives - a
        d = total - matched - c
        denominator = matched * (total - matched) * positives * (total - positives)
        statistic = total * (a * d - b * c) ** 2 / denominator if denominator else 0.0
        row.update(support_train=matched / total,
                   support_churn_train=a / positives,
                   support_stay_train=b / (total - positives), conf_train=a / matched,
                   lift_train=(a / matched) / base,
                   p_value=math.erfc(math.sqrt(statistic / 2)))
    if records:
        order = np.argsort([r["p_value"] for r in records])
        adjusted = np.array([records[i]["p_value"] for i in order])
        adjusted *= len(records) / np.arange(1, len(records) + 1)
        adjusted = np.minimum(1, np.minimum.accumulate(adjusted[::-1])[::-1])
        for index, q in zip(order, adjusted):
            records[index]["q_value"] = float(q)


def subpatterns(pattern):
    """Enumerate all proper nonempty subpatterns by deleting tokens/events."""
    found = set()
    pending = [pattern]
    while pending:
        current = pending.pop()
        for i, event in enumerate(current):
            for j in range(len(event)):
                smaller = event[:j] + event[j + 1:]
                candidate = current[:i] + ((smaller,) if smaller else ()) + current[i + 1:]
                if candidate and candidate not in found:
                    found.add(candidate)
                    pending.append(candidate)
    return found


def select_patterns(records, minimum, n_train):
    """Filter significance/effect and remove longer patterns with marginal gain."""
    threshold = math.ceil(minimum * n_train - 1e-12)
    eligible = {r["key"]: r for r in records if r["count"] >= threshold
                and r["q_value"] < 0.01 and (r["lift_train"] > 1.2 or r["lift_train"] < 0.8)}

    def strength(row):
        return abs(math.log(max(row["lift_train"], 1e-12)))

    selected = []
    for key, row in eligible.items():
        retain = row["lift_train"] < 1
        redundant = any(sub in eligible and (eligible[sub]["lift_train"] < 1) == retain
                        and strength(row) <= 1.05 * strength(eligible[sub])
                        for sub in subpatterns(key))
        if not redundant:
            row["direction"] = "retain" if retain else "churn"
            selected.append(row)
    churn = sorted((r for r in selected if r["direction"] == "churn"),
                   key=lambda r: (-r["lift_train"], -r["support_train"],
                                  r["q_value"], r["key"]))
    retain = sorted((r for r in selected if r["direction"] == "retain"),
                    key=lambda r: (-r["support_train"] * (1 - r["lift_train"]),
                                   r["lift_train"], -r["support_train"],
                                   r["q_value"], r["key"]))
    return churn + retain


def match_pattern(pattern, masks):
    """Evaluate a fixed pattern on all holdout users using the same semantics."""
    ends = None
    for event in pattern:
        occurrences = masks[event[0]].copy()
        for token in event[1:]:
            occurrences &= masks[token]
        if ends is None:
            ends = occurrences
        else:
            alive = ends != 0
            ends = occurrences & suffix_mask(ends)
            ends[~alive] = 0
    return ends != 0


def pattern_name(pattern, vocabulary):
    """Render unambiguous simultaneous itemsets and ordered transactions."""
    return " -> ".join("{" + ", ".join(vocabulary[t] for t in event) + "}" for event in pattern)


def load_cohort(source, sample_size, rng):
    """Sample the roster before materializing nested histories; retain empty users."""
    frame = pd.read_parquet(source, columns=["msno", "is_churn"])
    if frame.empty or frame.msno.isna().any() or frame.msno.duplicated().any():
        raise ValueError("Sequences must contain one row per non-null user")
    if not frame.is_churn.isin([0, 1]).all():
        raise ValueError("is_churn must contain binary non-null labels")
    frame = frame.sort_values("msno").reset_index(drop=True)
    n_available = len(frame)
    if sample_size and sample_size < len(frame):
        frame = frame.iloc[stratified_indices(frame.is_churn.to_numpy(), sample_size, rng)]
    with duckdb.connect() as con:
        con.register("sampled_users", frame[["msno"]])
        histories = con.execute("""SELECT s.msno, s.seq FROM read_parquet(?) s
                                   JOIN sampled_users u USING (msno)""", [str(source)]).df()
    return frame.merge(histories, on="msno", how="left", sort=False,
                       validate="one_to_one"), n_available


def main():
    """Write train-selected patterns, holdout estimates, sensitivity and figure."""
    started = time.perf_counter()
    data = Path(os.environ.get("DATA_DIR", ROOT / "data"))
    report = Path(os.environ.get("REPORT_DIR", ROOT / "report"))
    cohort = os.environ.get("COHORT", "mar")
    minimum = float(os.environ.get("SEQ_MIN_SUPPORT", os.environ.get("MIN_SUPPORT", "0.01")))
    sample_size = int(os.environ.get("SEQ_SAMPLE", "300000"))
    top = int(os.environ.get("SEQ_TOP", "50"))
    if not 0 < minimum <= 1 or sample_size < 0 or top < 1:
        raise ValueError("Support must be in (0,1], SEQ_SAMPLE >= 0 and SEQ_TOP >= 1")
    source = data / f"sequences_{cohort}.parquet"
    rng = np.random.default_rng(42)
    frame, n_available = load_cohort(source, sample_size, rng)
    labels = frame.is_churn.to_numpy(dtype=np.int8)
    train, holdout = split_indices(labels, rng)
    vocabulary, masks = encode(frame.seq.to_numpy())
    train_labels, hold_labels = labels[train], labels[holdout]
    train_masks, hold_masks = masks[:, train], masks[:, holdout]
    del masks
    thresholds = [0.005, 0.01, 0.02, 0.05]
    records = mine(train_masks, train_labels, min(minimum, *thresholds))
    add_statistics(records, train_labels)
    selected = select_patterns(records, minimum, len(train))
    reported = ([r for r in selected if r["direction"] == "churn"][:(top + 1) // 2]
                + [r for r in selected if r["direction"] == "retain"][:top // 2])
    feb, n_available_feb = load_cohort(data / "sequences_feb.parquet", sample_size,
                                     np.random.default_rng(42))
    feb_labels = feb.is_churn.to_numpy(dtype=np.int8)
    base_feb = float(feb_labels.mean())
    # Reuse March token IDs: tokens absent in February have zero occurrence masks.
    _, feb_masks = encode(feb.seq.to_numpy(), vocabulary)
    rows = []
    base_holdout = float(hold_labels.mean())
    for row in reported:
        matched = match_pattern(row["key"], hold_masks)
        count = int(matched.sum())
        confidence = float(hold_labels[matched].mean()) if count else None
        matched_feb = match_pattern(row["key"], feb_masks)
        count_feb = int(matched_feb.sum())
        confidence_feb = float(feb_labels[matched_feb].mean()) if count_feb else None
        rows.append({**{key: row[key] for key in COLUMNS if key in row},
                     "pattern": pattern_name(row["key"], vocabulary),
                     "support_holdout": count / len(holdout),
                     "conf_holdout": confidence,
                     "lift_holdout": confidence / base_holdout if count else None,
                     "support_feb": count_feb / len(feb),
                     "conf_feb": confidence_feb,
                     "lift_feb": confidence_feb / base_feb if count_feb and base_feb else None})
    log_pairs = np.asarray([(math.log(r["lift_train"]), math.log(r["lift_feb"]))
                            for r in rows if r["lift_train"] > 0
                            and r["lift_feb"] is not None and r["lift_feb"] > 0])
    corr_lift = None
    if len(log_pairs) >= 2 and np.all(np.ptp(log_pairs, axis=0) > 0):
        corr_lift = float(np.corrcoef(log_pairs.T)[0, 1])
    same_direction = sum(r["lift_feb"] is not None and
                         ((r["direction"] == "churn" and r["lift_feb"] > 1) or
                          (r["direction"] == "retain" and r["lift_feb"] < 1))
                         for r in rows) / len(rows) if rows else None
    sensitivity = []
    for threshold in thresholds:
        count_threshold = math.ceil(threshold * len(train) - 1e-12)
        sensitivity.append({"min_support": threshold,
                            "n_frequent": sum(r["count"] >= count_threshold for r in records),
                            "n_patterns": len(select_patterns(records, threshold, len(train)))})
    (report / "figures").mkdir(parents=True, exist_ok=True)
    (report / "metrics").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=COLUMNS).to_csv(report / "sequential_patterns.csv", index=False)
    pd.DataFrame(sensitivity).to_csv(report / "sequential_sensitivity.csv", index=False)
    displayed = rows[:15]
    fig, ax = plt.subplots(figsize=(12, max(3, len(displayed) * 0.5 + 1.5)))
    if displayed:
        y = np.arange(len(displayed))
        lifts = [r["lift_holdout"] if r["lift_holdout"] is not None else np.nan for r in displayed]
        feb_lifts = [r["lift_feb"] if r["lift_feb"] is not None else np.nan for r in displayed]
        ax.barh(y - 0.18, lifts, height=0.36, color="#4477aa",
                label=f"Tập kiểm định {cohort} (tháng 3)" if cohort == "mar" else f"Tập kiểm định {cohort}")
        ax.barh(y + 0.18, feb_lifts, height=0.36, color="#ee9944", label="Tháng 2")
        ax.legend(loc="best", fontsize=8)
        ax.set_yticks(y, [r["pattern"] for r in displayed], fontsize=8)
        for index, row in enumerate(displayed):
            if row["lift_holdout"] is None:
                ax.text(0.02, index - 0.18, "Không có mẫu khớp ở tập kiểm định", va="center", fontsize=8)
            if row["lift_feb"] is None:
                ax.text(0.02, index + 0.18, "Không đủ dữ liệu tháng 2", va="center", fontsize=8)
        ax.invert_yaxis()
    else:
        ax.text(0.5, 0.5, "Không có mẫu đạt tiêu chí trên tập huấn luyện", ha="center", transform=ax.transAxes)
    ax.axvline(1, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("Lift rời bỏ so với tỷ lệ nền từng tập (chọn và xếp mẫu theo tập huấn luyện)")
    ax.set_ylabel("Mẫu tuần tự")
    ax.set_title(f"Mẫu tuần tự: kiểm định {cohort} (n={len(holdout):,}) và tháng 2 (n={len(feb):,})")
    fig.tight_layout()
    fig.savefig(report / "figures" / "seq_top_patterns.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    metrics = {"cohort": cohort, "n_users": len(frame), "n_available": n_available,
               "n_train": len(train), "n_holdout": len(holdout),
               "n_empty_sequences": int(sum(len(seq) == 0 for seq in frame.seq)),
               "base_rate": float(labels.mean()), "base_rate_train": float(train_labels.mean()),
               "base_rate_holdout": base_holdout, "min_support": minimum,
               "feb_base_rate": base_feb, "n_users_feb": len(feb),
               "n_available_feb": n_available_feb,
               "corr_lift_mar_feb": corr_lift,
               "corr_lift_mar_feb_reference": "lift_train",
               "n_log_lift_pairs_mar_feb": len(log_pairs),
               "share_same_direction_feb": same_direction,
               "max_pattern_length": MAX_LENGTH, "n_tested": len(records),
               "n_selected": len(selected), "n_patterns": len(rows),
               "n_churn_patterns": sum(r["direction"] == "churn" for r in rows),
               "n_retain_patterns": sum(r["direction"] == "retain" for r in rows),
               "seed": 42, "test": "Pearson chi-square (1 df), BH FDR over mined family",
               "sensitivity_fdr": "Shared family at lowest requested/sensitivity support",
               "runtime_s": round(time.perf_counter() - started, 3),
               "top5": [{**row, "support_churn_train": chosen["support_churn_train"],
                         "support_stay_train": chosen["support_stay_train"]}
                        for row, chosen in zip(rows[:5], reported[:5])]}
    with (report / "metrics" / "sequential.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, ensure_ascii=False, indent=2, allow_nan=False)
    print(f"{len(frame):,} users; {len(records):,} frequent patterns; "
          f"{len(selected):,} selected; {len(rows)} reported; {metrics['runtime_s']:.2f}s")
    print(f"Outputs: {report.resolve()}")


if __name__ == "__main__":
    main()
