"""Shared helpers: signal feature engineering, metrics, noise injection."""
import time
import warnings
import numpy as np
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score,
                             recall_score, f1_score, matthews_corrcoef)

NO_SIGNAL = -105          # replaces the dataset's "100 = not detected" placeholder
TARGET = "zone"           # building + floor label, e.g. B1_F2


def add_signal_features(X):
    """Append 4 physically meaningful features to the raw RSSI columns:
    number of detected APs, strongest RSSI, mean RSSI of detected APs, spread."""
    X = np.asarray(X, dtype=float)
    detected = X > NO_SIGNAL
    n_det = detected.sum(axis=1)
    strongest = X.max(axis=1)
    masked = np.where(detected, X, np.nan)
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)   # all-undetected rows give empty slices
        mean_det = np.nan_to_num(np.nanmean(masked, axis=1), nan=NO_SIGNAL)
        std_det = np.nan_to_num(np.nanstd(masked, axis=1), nan=0.0)
    return np.column_stack([X, n_det, strongest, mean_det, std_det])


def add_noise(X, sigma, drop_frac=0.0, seed=0):
    """Simulate real-world RSSI fluctuation: gaussian noise on detected APs
    plus random AP dropout. Undetected APs stay at NO_SIGNAL."""
    rng = np.random.default_rng(seed)
    X = np.asarray(X, dtype=float).copy()
    detected = X > NO_SIGNAL
    X = np.where(detected, X + rng.normal(0, sigma, X.shape), X)
    if drop_frac > 0:
        X[detected & (rng.random(X.shape) < drop_frac)] = NO_SIGNAL
    return np.clip(X, NO_SIGNAL, 0)


def evaluate(model, X, y):
    """Six required metrics + inference time (ms per 1000 rows)."""
    classes = model.classes_
    t0 = time.perf_counter()
    pred = model.predict(X)
    ms = (time.perf_counter() - t0) / max(len(X), 1) * 1000 * 1000
    proba = model.predict_proba(X)
    if len(classes) == 2:
        auc = roc_auc_score(y, proba[:, 1])
    else:
        auc = roc_auc_score(y, proba, multi_class="ovr", average="weighted", labels=classes)
    avg = "binary" if len(classes) == 2 else "weighted"
    kw = dict(average=avg, zero_division=0)
    if avg == "binary":
        kw["pos_label"] = classes[1]
    return {
        "Accuracy": accuracy_score(y, pred), "AUC": auc,
        "Precision": precision_score(y, pred, **kw),
        "Recall": recall_score(y, pred, **kw),
        "F1": f1_score(y, pred, **kw),
        "MCC": matthews_corrcoef(y, pred),
        "Inference ms/1k rows": ms,
    }


def split_zone(label):
    """'B1_F2' -> (1, 2): building number, floor number."""
    b, f = str(label).split("_")
    return int(b[1:]), int(f[1:])


def level_accuracies(y_true, y_pred):
    """Accuracy at building level, floor level (floor number only) and exact zone."""
    t = np.array([split_zone(z) for z in y_true])
    p = np.array([split_zone(z) for z in y_pred])
    return {"Building": float((t[:, 0] == p[:, 0]).mean()),
            "Floor": float((t[:, 1] == p[:, 1]).mean()),
            "Exact zone": float((t == p).all(axis=1).mean())}
