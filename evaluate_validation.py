"""Extra check: score the saved models on UJIIndoorLoc's separate validationData.csv
(collected later than the training data, so it is a harder, more realistic test).
Models are NOT retrained or tuned on this file.
Usage: python evaluate_validation.py validationData.csv
"""
import glob
import os
import sys
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score,
                             recall_score, f1_score, matthews_corrcoef)
from common import NO_SIGNAL

path = sys.argv[1] if len(sys.argv) > 1 else "validationData.csv"
feat = [c for c in pd.read_csv("prepared_data.csv", nrows=0).columns if c != "zone"]

raw = pd.read_csv(path)
zone = "B" + raw["BUILDINGID"].astype(str) + "_F" + raw["FLOOR"].astype(str)
X = raw[feat].replace(100, NO_SIGNAL)
ok = (X > NO_SIGNAL).any(axis=1)                 # same rule as training: drop rows with no AP heard
X, y = X[ok].reset_index(drop=True), zone[ok].reset_index(drop=True)
print(f"validation rows used: {len(X)} of {len(raw)}")

rows = []
for p in sorted(glob.glob("model/*.joblib")):
    name = os.path.basename(p)[:-7].replace("_", " ").title()
    m = joblib.load(p)
    known = y.isin(m.classes_)
    Xk, yk = X[known], y[known]
    pred = m.predict(Xk)
    proba = m.predict_proba(Xk)
    present = [c for c in m.classes_ if c in set(yk)]          # AUC only over zones that occur
    idx = [list(m.classes_).index(c) for c in present]
    pp = proba[:, idx] / proba[:, idx].sum(axis=1, keepdims=True)
    auc = roc_auc_score(yk, pp, multi_class="ovr", average="weighted", labels=present)
    kw = dict(average="weighted", zero_division=0)
    rows.append({"ML Model Name": name, "Accuracy": accuracy_score(yk, pred), "AUC": auc,
                 "Precision": precision_score(yk, pred, **kw), "Recall": recall_score(yk, pred, **kw),
                 "F1": f1_score(yk, pred, **kw), "MCC": matthews_corrcoef(yk, pred)})

out = pd.DataFrame(rows).round(4)
out.to_csv("validation_metrics.csv", index=False)
print(out.to_string(index=False))
