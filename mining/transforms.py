"""Feature transforms shared by several mining scripts (so other cohorts get identical treatment)."""
import numpy as np
import pandas as pd

FEATS = ["n_tx", "n_cancel", "auto_renew_rate", "discount_rate", "last_plan_days", "last_paid",
         "days_to_expire", "tenure_days", "active_days", "total_hours", "complete_ratio",
         "skip_ratio", "avg_unique_songs", "listen_trend", "days_since_last_listen", "weekly_cv"]
LOGT = ["n_tx", "n_cancel", "last_plan_days", "last_paid", "tenure_days", "active_days",
        "total_hours", "avg_unique_songs", "days_since_last_listen"]


def prepare(d):
    """Feature matrix before scaling. Users without logs get explicit 'inactive' values."""
    raw = d[FEATS].copy()
    nolog = d.has_logs == 0
    raw.loc[nolog, ["complete_ratio", "skip_ratio", "listen_trend"]] = 0
    raw.loc[nolog, "days_since_last_listen"] = 60
    raw.loc[nolog, "weekly_cv"] = 3.0          # maximal irregularity
    raw["days_to_expire"] = raw.days_to_expire.clip(-60, 400)
    raw[LOGT] = np.log1p(raw[LOGT].clip(lower=0))
    return raw


def itemise(d, ref):
    """Map users to items. Quantile cut-points are taken from ``ref`` (the train split)."""
    act_q = ref.loc[ref.has_logs == 1, "active_days"].quantile([1 / 3, 2 / 3]).values
    skip_q = ref.loc[ref.has_logs == 1, "skip_ratio"].quantile(2 / 3)
    logs = d.has_logs == 1
    it = pd.DataFrame(index=d.index)
    it["renew"] = d.last_auto_renew.map({1: "auto_renew=on", 0: "auto_renew=off"}).fillna("no_tx")
    it["cancel"] = np.where(d.last_is_cancel == 1, "last_tx=cancel",
                   np.where(d.n_cancel > 0, "cancelled_before", "never_cancelled"))
    it["plan"] = pd.cut(d.last_plan_days, [-1, 7, 30, 10_000],
                        labels=["plan<=7d", "plan=30d", "plan>30d"]).astype(str).replace("nan", "plan=none")
    it["price"] = np.select([d.last_paid == 0, d.last_paid < d.last_list_price],
                            ["paid=free", "paid=discount"], "paid=full")
    it["tenure"] = pd.cut(d.tenure_days, [-1, 365, 3 * 365, 6 * 365, 1e6],
                          labels=["tenure<1y", "tenure1-3y", "tenure3-6y", "tenure>6y"]).astype(str)
    it["via"] = "via=" + d.registered_via.astype("Int64").astype(str)
    it["gender"] = "gender=" + d.gender.astype(str)
    it["activity"] = np.where(~logs, "no_logs", pd.cut(d.active_days, [-1, act_q[0], act_q[1], 1e6],
                              labels=["active=low", "active=mid", "active=high"]).astype(str))
    it["idle"] = np.where(~logs, None, pd.cut(d.days_since_last_listen, [-1, 3, 14, 1e6],
                          labels=["idle<=3d", "idle4-14d", "idle>14d"]).astype(str))
    it["trend"] = np.where(~logs, None, pd.cut(d.listen_trend, [-99, -0.5, 0.5, 99],
                           labels=["trend=falling", "trend=flat", "trend=rising"]).astype(str))
    it["skip"] = np.where(logs & (d.skip_ratio > skip_q), "skip=high", None)
    it["expire"] = pd.cut(d.days_to_expire, [-1e6, 0, 31, 1e6],
                          labels=["expired_at_T", "expires<=31d", "expires>31d"]).astype(str).replace("nan", None)
    if "cluster" in d:
        it["segment"] = np.where(d.cluster.isna(), None, "segment=" + d.cluster.astype("Int64").astype(str))
    it = it.astype(object).where(it.notna(), None)
    return pd.concat([pd.get_dummies(it[c].dropna(), prefix="", prefix_sep="").reindex(d.index, fill_value=False)
                      for c in it.columns], axis=1).astype(bool)
