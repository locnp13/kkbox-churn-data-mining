"""Decision tree + random forest as a supporting method: which features matter and why."""
import sys
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import FIG, NUM, load

SEED = 42
df = load()
# no leakage columns: only behaviour/profile features, never derived from the churn label
X = pd.concat([df[NUM + ["tenure_days", "has_logs"]],
               pd.get_dummies(df[["gender", "registered_via"]].astype(str))], axis=1)
y = df["is_churn"]
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, stratify=y, random_state=SEED)

tree = DecisionTreeClassifier(max_depth=4, min_samples_leaf=200, random_state=SEED).fit(Xtr, ytr)
print(export_text(tree, feature_names=list(X.columns)))
rf = RandomForestClassifier(300, min_samples_leaf=20, n_jobs=-1, random_state=SEED).fit(Xtr, ytr)
print(classification_report(yte, rf.predict(Xte)))
print("ROC-AUC rf:", round(roc_auc_score(yte, rf.predict_proba(Xte)[:, 1]), 3))
imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
print(imp.head(15).round(3).to_string())
imp.to_csv(FIG.parent / "feature_importance.csv")
