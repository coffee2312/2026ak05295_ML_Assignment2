"""SignalScope - indoor zone locator (Streamlit app).
Loads five saved classifiers and scores them on an uploaded test file. No retraining happens here."""
import glob
import html
import io
import os

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch, Rectangle
from sklearn.metrics import confusion_matrix, classification_report, recall_score

from common import (TARGET, NO_SIGNAL, add_noise, evaluate, add_signal_features,  # noqa: F401 (needed to unpickle)
                    split_zone, level_accuracies)

BASE = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------- palette
# Built from a signal-strength colour ramp: weak = dim purple, strong = bright peach.
PAPER, LILAC, WHITE = "#F7F4F9", "#EDE7F2", "#FFFFFF"
INK, MUTED, LINE = "#1E1033", "#6B5F7A", "#D9CFE3"
MAGENTA, ROSE, PEACH = "#8C2981", "#DE4968", "#FE9F6D"
OK, FIXED, NOTE, FAIL, WAIT = "#1E8E6E", "#3B5BDB", "#D99A00", "#D1344E", "#C9BFD6"
RAMP = [(0.0, "#5B2C83"), (0.40, "#A8317F"), (0.68, "#E8506A"), (0.86, "#FE9F6D"), (1.0, "#FCFDBF")]
CMAP = LinearSegmentedColormap.from_list("signal", [PAPER, "#FBD9C6", PEACH, ROSE, MAGENTA, INK])
FLOORS = {0: 4, 1: 4, 2: 5}          # floors per building in UJIIndoorLoc


def ramp(f):
    """Colour for a signal strength between 0 (weakest) and 1 (strongest)."""
    f = min(max(float(f), 0.0), 1.0)
    for (a, ca), (b, cb) in zip(RAMP, RAMP[1:]):
        if f <= b:
            t = (f - a) / (b - a)
            ra = [int(ca[i:i + 2], 16) for i in (1, 3, 5)]
            rb = [int(cb[i:i + 2], 16) for i in (1, 3, 5)]
            return "#%02x%02x%02x" % tuple(round(x + (y - x) * t) for x, y in zip(ra, rb))
    return RAMP[-1][1]


RAMP_CSS = "linear-gradient(90deg, " + ", ".join(f"{c} {int(p * 100)}%" for p, c in RAMP) + ")"


# ---------------------------------------------------------------- charts (matplotlib)
def _axes(ax):
    ax.set_facecolor(PAPER)
    for s in ax.spines.values():
        s.set_color(LINE)
    ax.tick_params(colors=INK)


def confusion_figure(cm, labels, normalize=False):
    data = np.asarray(cm, dtype=float)
    if normalize:
        data = data / np.maximum(data.sum(axis=1, keepdims=True), 1)
    n = len(labels)
    fig, ax = plt.subplots(figsize=(8.2, 6.6), facecolor=PAPER)
    ax.imshow(data, cmap=CMAP, vmin=0, vmax=max(data.max(), 1e-9))
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=60, ha="right", color=INK, fontsize=8)
    ax.set_yticklabels(labels, color=INK, fontsize=8)
    ax.set_xlabel("Predicted zone", color=MUTED)
    ax.set_ylabel("Actual zone", color=MUTED)
    thr = data.max() * 0.5
    for i in range(n):
        for j in range(n):
            v = data[i, j]
            if v > 0:
                txt = f"{v * 100:.0f}" if normalize else f"{int(v)}"
                ax.text(j, i, txt, ha="center", va="center", fontsize=7.5,
                        color=PAPER if v > thr else INK, fontweight="bold" if i == j else "normal")
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks(np.arange(-.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-.5, n, 1), minor=True)
    ax.grid(which="minor", color=PAPER, linewidth=1.2)
    ax.tick_params(which="minor", length=0)
    fig.tight_layout()
    return fig


def zone_recall_figure(labels, recalls, supports):
    fig, ax = plt.subplots(figsize=(7.5, 4.6), facecolor=PAPER)
    _axes(ax)
    cols = [OK if r >= .95 else NOTE if r >= .80 else FAIL for r in recalls]
    y = np.arange(len(labels))[::-1]
    ax.barh(y, recalls, color=cols, height=.62)
    ax.set_yticks(y); ax.set_yticklabels(labels, color=INK, fontsize=8)
    ax.set_xlim(0, 1.3)
    ax.set_xticks([0, .25, .5, .75, 1.0])
    ax.set_xlabel("Recall: share of that zone's rows the model finds", color=MUTED)
    for yi, r, n in zip(y, recalls, supports):
        ax.text(r + .015, yi, f"{r:.2f}  (n={n})", va="center", color=INK, fontsize=7.5)
    ax.legend(handles=[Patch(color=OK, label="0.95 or more"), Patch(color=NOTE, label="0.80 to 0.95"),
                       Patch(color=FAIL, label="below 0.80")], loc="lower right", fontsize=7,
              facecolor=WHITE, edgecolor=LINE, labelcolor=INK)
    fig.tight_layout()
    return fig


def building_figure(pred_zone, actual_zone):
    """Cross-section of the three buildings: predicted floor filled, actual floor outlined."""
    fig, ax = plt.subplots(figsize=(7.6, 4.3), facecolor=PAPER)
    ax.set_facecolor(PAPER); ax.axis("off")
    ax.set_xlim(-0.3, 8.9); ax.set_ylim(-1.4, 5.9)
    correct = pred_zone == actual_zone
    for b, nf in FLOORS.items():
        x0 = b * 3.0
        ax.text(x0 + 1.15, -0.55, f"Building {b}", ha="center", color=MUTED, fontsize=9)
        for f in range(nf):
            z = f"B{b}_F{f}"
            face, edge, lw = WHITE, LINE, 1.0
            if z == pred_zone:
                face = OK if correct else FAIL
            if z == actual_zone:
                edge, lw = MAGENTA, 3.4
            ax.add_patch(Rectangle((x0, f), 2.3, 0.86, facecolor=face, edgecolor=edge, linewidth=lw))
            ax.text(x0 + 1.15, f + 0.43, f"floor {f}", ha="center", va="center", fontsize=8,
                    color=WHITE if z == pred_zone else MUTED)
    ax.legend(handles=[Patch(facecolor=OK, label="predicted (correct)"),
                       Patch(facecolor=FAIL, label="predicted (wrong)"),
                       Patch(facecolor=WHITE, edgecolor=MAGENTA, linewidth=2.5, label="actual")],
              loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=3, fontsize=8,
              facecolor=WHITE, edgecolor=LINE, labelcolor=INK)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------- CSV format checker
CHECK_ORDER = ["File", "Zone label", "RSSI columns", "Cell values", "Value range",
               "Zone names", "Row count", "Silent fingerprints"]


def expected_columns(models):
    """Names of the RSSI columns the saved models were trained on."""
    try:
        return [str(c) for c in next(iter(models.values())).feature_names_in_]
    except Exception:
        return None


def verify_csv(raw, names, classes, note=None):
    """Check an uploaded table against what the saved models need, and fix what can be fixed safely.
    Returns (clean_dataframe_or_None, checks) where checks is a list of (state, title, detail);
    state is 'ok', 'fix' (fixed automatically), 'warn' or 'fail'."""
    checks = []
    df = raw.copy()
    df.columns = [str(c).strip() for c in df.columns]
    checks.append(("fix" if note else "ok", "File",
                   f"{len(df):,} rows, {df.shape[1]} columns" + (f"; {note}" if note else "")))

    zone = None
    if TARGET in df.columns:
        zone = df[TARGET].astype(str).str.strip()
        checks.append(("ok", "Zone label", "Column 'zone' found"))
    elif {"BUILDINGID", "FLOOR"} <= set(df.columns):
        b = pd.to_numeric(df["BUILDINGID"], errors="coerce")
        f = pd.to_numeric(df["FLOOR"], errors="coerce")
        if b.isna().any() or f.isna().any():
            checks.append(("fail", "Zone label", "BUILDINGID or FLOOR has empty or non-numeric cells."))
        else:
            zone = "B" + b.astype(int).astype(str) + "_F" + f.astype(int).astype(str)
            checks.append(("fix", "Zone label", "Built from BUILDINGID and FLOOR (raw UCI file)"))
    else:
        checks.append(("fail", "Zone label",
                       "No 'zone' column found. Add one with values like B1_F2, or include BUILDINGID and FLOOR."))

    num = None
    missing = [c for c in names if c not in df.columns]
    if missing:
        checks.append(("fail", "RSSI columns",
                       f"{len(missing)} of {len(names)} access point columns are missing "
                       f"(for example {', '.join(missing[:3])}). Column names must match exactly."))
    else:
        in_file_order = [c for c in df.columns if c in set(names)]
        extra = df.shape[1] - len(names) - (1 if TARGET in df.columns else 0)
        detail, state = f"All {len(names)} access point columns found", "ok"
        if in_file_order != list(names):
            detail, state = detail + "; reordered to match the models", "fix"
        if extra > 0:
            detail += f"; {extra} other columns ignored"
        checks.append((state, "RSSI columns", detail))

        block = df[list(names)]
        num = block.apply(pd.to_numeric, errors="coerce")
        bad = int(num.isna().sum().sum() - block.isna().sum().sum())
        if bad:
            checks.append(("fail", "Cell values",
                           f"{bad:,} cells are not numbers. Signal columns must hold dBm values only."))
            num = None
        else:
            empty = int(num.isna().sum().sum())
            n100 = int((num == 100).sum().sum())
            notes = []
            if empty:
                num = num.fillna(NO_SIGNAL)
                notes.append(f"{empty:,} empty cells set to {NO_SIGNAL}")
            if n100:
                num = num.mask(num == 100, NO_SIGNAL)
                notes.append(f"{n100:,} values of 100 (the UCI code for not detected) set to {NO_SIGNAL}")
            checks.append(("fix" if notes else "ok", "Cell values",
                           "; ".join(notes) if notes else "All cells are numbers, none empty"))
            lo, hi = float(num.min().min()), float(num.max().max())
            if lo < NO_SIGNAL or hi > 0:
                checks.append(("fail", "Value range",
                               f"Values must lie between {NO_SIGNAL} and 0 dBm; this file runs from {lo:.0f} to {hi:.0f}."))
                num = None
            else:
                checks.append(("ok", "Value range", f"All values between {lo:.0f} and {hi:.0f} dBm"))

    if zone is not None:
        unknown = sorted(set(zone) - set(classes))
        if unknown:
            checks.append(("fail", "Zone names",
                           f"{len(unknown)} unknown zone names (for example {', '.join(unknown[:4])}). "
                           f"Known zones run from {classes[0]} to {classes[-1]}."))
        else:
            checks.append(("ok", "Zone names", f"{zone.nunique()} of {len(classes)} known zones appear"))

    n = len(df)
    if n == 0:
        checks.append(("fail", "Row count", "The file has no data rows."))
    elif n < 30:
        checks.append(("warn", "Row count", f"Only {n} rows. Scores from so few rows are unstable."))
    else:
        checks.append(("ok", "Row count", f"{n:,} rows"))

    if num is not None:
        silent = int((~(num > NO_SIGNAL).any(axis=1)).sum())
        if silent:
            checks.append(("warn", "Silent fingerprints",
                           f"{silent:,} fingerprints hear no access point at all. They are kept, "
                           "but the models can only guess for them."))
        else:
            checks.append(("ok", "Silent fingerprints", "Every fingerprint hears at least one access point"))

    clean = None
    if num is not None and zone is not None and not any(c[0] == "fail" for c in checks):
        clean = num.copy()
        clean[TARGET] = zone.values
    checks.sort(key=lambda c: CHECK_ORDER.index(c[1]))
    return clean, checks


def waiting_checks(names, classes):
    return [("wait", "File", "A comma-separated .csv with a header row"),
            ("wait", "Zone label", "A column named zone with values like B1_F2, or BUILDINGID and FLOOR from the raw UCI file"),
            ("wait", "RSSI columns", f"The {len(names)} access point columns, {names[0]} to {names[-1]}, named exactly"),
            ("wait", "Cell values", f"Numbers only. A 100 (not detected in UCI files) is changed to {NO_SIGNAL}"),
            ("wait", "Value range", f"Between {NO_SIGNAL} and 0 dBm"),
            ("wait", "Zone names", f"Only the {len(classes)} known zones, {classes[0]} to {classes[-1]}"),
            ("wait", "Row count", "At least one row; 30 or more for stable scores"),
            ("wait", "Silent fingerprints", "Rows that hear no access point are flagged, not removed")]


GLYPH = {"ok": "&#10003;", "fix": "&#10003;", "warn": "!", "fail": "&#10005;", "wait": "&ndash;"}


def checklist_html(checks):
    rows = "".join(
        f'<div class="crow"><span class="ico {s}">{GLYPH[s]}</span><div><div class="ct">{html.escape(t)}</div>'
        f'<div class="cd">{html.escape(d)}</div></div></div>' for s, t, d in checks)
    legend = ('<div class="clegend"><span class="ico ok">&#10003;</span> passed '
              '<span class="ico fix">&#10003;</span> fixed for you '
              '<span class="ico warn">!</span> worth a look '
              '<span class="ico fail">&#10005;</span> needs your attention</div>')
    return f'<div class="checks">{rows}</div>{legend}'


# ---------------------------------------------------------------- page look
st.set_page_config(page_title="SignalScope | Indoor Zone Locator", page_icon="📶", layout="wide",
                   initial_sidebar_state="expanded")

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Archivo:wght@500;700;800&display=swap');
:root {color-scheme: light;}
.stApp {background: @@PAPER; color: @@INK;}
.block-container {padding-top: 1.4rem; max-width: 1250px;}
h2, h3, h4 {font-family: 'Archivo', 'Segoe UI', system-ui, sans-serif; color: @@INK; letter-spacing: -.2px;}
[data-testid="stSidebar"] {background: @@LILAC; border-right: 1px solid @@LINE;}
[data-testid="stSidebar"] [data-testid="stVerticalBlockBorderWrapper"] {background: @@WHITE; border-radius: 12px;}
.sbname {font-family: 'Archivo', 'Segoe UI', system-ui, sans-serif; font-weight: 800; font-size: 1.5rem;
         letter-spacing: -.4px; color: @@INK; margin: 0 0 14px 2px;}
.sbhead {display: flex; align-items: center; gap: 8px; font-family: 'Archivo', 'Segoe UI', system-ui, sans-serif;
         font-weight: 700; font-size: 1.02rem; color: @@INK; margin-bottom: 6px;}
.dot {width: 10px; height: 10px; border-radius: 50%; display: inline-block; background: @@WAIT;}
.dot.ok {background: @@OK;} .dot.fix {background: @@FIXED;} .dot.warn {background: @@NOTE;} .dot.fail {background: @@FAIL;}
.sbnote {font-size: .8rem; color: @@MUTED; margin: 2px 0 2px 0;}
.badge {border-radius: 8px; padding: 7px 10px; font-size: .84rem; font-weight: 600; margin-top: 6px;}
.badge.wait {background: @@LILAC; color: @@MUTED;} .badge.ok {background: #DCF2E8; color: #0F5C45;}
.badge.fix {background: #E2E8FB; color: #243C9E;} .badge.fail {background: #FADADF; color: #8F1B2F;}
.ready {margin-top: 14px; padding: 10px 12px; border-radius: 12px; background: @@WHITE; border: 1px solid @@LINE;}
.ready .rr {display: flex; align-items: center; gap: 8px; font-size: .82rem; color: @@MUTED; padding: 3px 0;}
.ready .rr b {color: @@INK; margin-left: auto; font-weight: 600; text-align: right;}
.hero {background: @@INK; border-radius: 16px; padding: 28px 32px 18px; margin-bottom: 18px; color: @@PAPER;}
.hero h1 {font-family: 'Archivo', 'Segoe UI', system-ui, sans-serif; font-weight: 800; font-size: 2.9rem;
          letter-spacing: -1px; line-height: 1.05; margin: 0; padding: 0; color: @@PAPER;}
.hero .lead {margin: 8px 0 0 0; font-size: 1.08rem; color: #E4D9EF;}
.hero .sub {margin: 2px 0 0 0; font-size: .92rem; color: #A99BBE;}
.hero svg.strip {display: block; width: 100%; height: 66px; margin-top: 20px;}
.hero .key {display: flex; align-items: center; gap: 10px; margin-top: 8px; font-size: .78rem; color: #A99BBE;}
.hero .key i {display: block; width: 150px; height: 8px; border-radius: 4px; background: @@RAMPCSS;}
.hero .cap {margin-top: 6px; font-size: .84rem; color: #CBBEDD;}
.checks {background: @@WHITE; border: 1px solid @@LINE; border-radius: 12px; padding: 4px 6px;}
.crow {display: flex; gap: 12px; align-items: flex-start; padding: 9px 10px; border-bottom: 1px solid #EEE7F3;}
.crow:last-child {border-bottom: none;}
.ico {flex: 0 0 24px; width: 24px; height: 24px; border-radius: 50%; text-align: center; line-height: 24px;
      color: #FFFFFF; font-weight: 700; font-size: .8rem; display: inline-block;}
.ico.ok {background: @@OK;} .ico.fix {background: @@FIXED;} .ico.warn {background: @@NOTE;}
.ico.fail {background: @@FAIL;} .ico.wait {background: @@WAIT;}
.ct {font-weight: 700; color: @@INK; font-size: .93rem;} .cd {color: @@MUTED; font-size: .85rem; line-height: 1.4;}
.clegend {font-size: .8rem; color: @@MUTED; margin: 10px 2px 0 2px;}
.clegend .ico {width: 18px; height: 18px; line-height: 18px; font-size: .62rem; flex: none; margin: 0 4px 0 12px;}
.clegend .ico:first-child {margin-left: 0;}
.fixhint {border-left: 5px solid @@FAIL; background: @@WHITE; padding: 10px 14px; border-radius: 0 10px 10px 0;
          margin: 12px 0; font-size: .92rem;}
.plain {border-left: 5px solid @@MAGENTA; background: @@WHITE; padding: 12px 16px; border-radius: 0 10px 10px 0;
        margin: 6px 0 14px 0; font-size: 1.02rem; line-height: 1.55;}
.ros {display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); border-top: 3px solid @@INK;
      margin: 10px 0 6px 0;}
.ro {padding: 12px 14px 8px 0;} .ro + .ro {border-left: 1px solid @@LINE; padding-left: 16px;}
.rl {font-size: .85rem; color: @@MUTED; font-weight: 600;}
.rv {font-family: 'Archivo', 'Segoe UI', system-ui, sans-serif; font-weight: 800; font-size: 2.15rem; line-height: 1.2;
     color: @@INK; font-variant-numeric: tabular-nums;}
.rm {height: 5px; border-radius: 3px; background: #E5DCEC; margin: 4px 0 6px 0; overflow: hidden;}
.rm i {display: block; height: 100%; background: @@MAGENTA;}
.rh {font-size: .78rem; color: @@MUTED; line-height: 1.3;}
.banner {border-left: 5px solid @@NOTE; background: #FBF0D4; padding: 8px 14px; border-radius: 6px;
         color: #6F4B00; font-size: .9rem; margin: 6px 0 12px 0;}
.foot {color: @@MUTED; font-size: .8rem; text-align: center; margin-top: 30px; border-top: 1px solid @@LINE; padding-top: 12px;}
button[data-baseweb="tab"] {font-weight: 600;}
button[data-baseweb="tab"][aria-selected="true"] {color: @@MAGENTA;}
div[data-baseweb="tab-highlight"] {background-color: @@MAGENTA;}
</style>
"""
TOKENS = {"PAPER": PAPER, "LILAC": LILAC, "WHITE": WHITE, "INK": INK, "MUTED": MUTED, "LINE": LINE,
          "MAGENTA": MAGENTA, "OK": OK, "FIXED": FIXED, "NOTE": NOTE, "FAIL": FAIL, "WAIT": WAIT,
          "RAMPCSS": RAMP_CSS}
for _k in sorted(TOKENS, key=len, reverse=True):
    CSS = CSS.replace("@@" + _k, TOKENS[_k])
st.markdown(CSS, unsafe_allow_html=True)

MODEL_ORDER = ["Logistic Regression", "Decision Tree", "KNN", "Naive Bayes", "Support Vector Machine"]
FILE_NAMES = {"logistic_regression": "Logistic Regression", "decision_tree": "Decision Tree", "knn": "KNN",
              "naive_bayes": "Naive Bayes", "support_vector_machine": "Support Vector Machine"}
HINTS = {"Accuracy": "rows given the exact right zone", "AUC": "how well the right zone is ranked",
         "Precision": "when it names a zone, how often it is right", "Recall": "share of true rows it finds",
         "F1": "balance of precision and recall", "MCC": "overall quality, from -1 to 1"}


@st.cache_resource
def load_models():
    found = {}
    for p in glob.glob(os.path.join(BASE, "model", "*.joblib")):
        key = os.path.basename(p)[:-7]
        found[FILE_NAMES.get(key, key.replace("_", " ").title())] = joblib.load(p)
    ordered = {n: found[n] for n in MODEL_ORDER if n in found}
    ordered.update({n: m for n, m in found.items() if n not in ordered})
    return ordered


@st.cache_data(show_spinner=False)
def read_upload(raw_bytes):
    df = pd.read_csv(io.BytesIO(raw_bytes))
    note = None
    if df.shape[1] == 1 and ";" in str(df.columns[0]):
        df = pd.read_csv(io.BytesIO(raw_bytes), sep=";")
        note = "semicolon separators detected and handled"
    return df, note


@st.cache_data(show_spinner=False)
def run_model(name, X, y, _model):
    return evaluate(_model, X, y), _model.predict(X)


@st.cache_data(show_spinner="Scoring all five models on this data...")
def score_all(X, y, _models):
    return pd.DataFrame({n: evaluate(m, X, y) for n, m in _models.items()}).T


def readouts(items, hints=None):
    out = '<div class="ros">'
    for label, v in items:
        missing = v is None or (isinstance(v, float) and np.isnan(v))
        txt = "n/a" if missing else f"{v:.4f}"
        width = 0 if missing else max(min(v, 1.0), 0.0) * 100
        hint = (hints or {}).get(label, "")
        out += (f'<div class="ro"><div class="rl">{label}</div><div class="rv">{txt}</div>'
                f'<div class="rm"><i style="width:{width:.1f}%"></i></div><div class="rh">{hint}</div></div>')
    return out + "</div>"


def strip_svg(values):
    """One bar per access point: taller and brighter means a stronger signal."""
    bars = []
    for i, v in enumerate(values):
        if v > NO_SIGNAL:
            f = (v - NO_SIGNAL) / (-NO_SIGNAL)
            h = 10 + 38 * f
            bars.append(f'<rect x="{i}" y="{50 - h:.1f}" width="0.85" height="{h:.1f}" fill="{ramp(f)}"/>')
        else:
            bars.append(f'<rect x="{i}" y="48" width="0.85" height="2" fill="#44335C"/>')
    return f'<svg class="strip" viewBox="0 0 {len(values)} 50" preserveAspectRatio="none">{"".join(bars)}</svg>'


def hero_html(values, caption):
    return (f'<div class="hero"><h1>SignalScope</h1>'
            f'<p class="lead">Predicts the building and floor from WiFi signal strengths (UJIIndoorLoc)</p>'
            f'<p class="sub">Models are loaded from the model folder. Nothing is trained in this app.</p>'
            f'{strip_svg(values)}'
            f'<div class="key"><span>weak</span><i></i><span>strong signal</span></div>'
            f'<div class="cap">{caption}</div></div>')


def plain_words(choice, y, pred, lv, labels, cm, rec, sup):
    n, hit = len(y), int((y.values == pred).sum())
    parts = [f"<b>{choice}</b> placed <b>{hit:,} of {n:,}</b> fingerprints ({hit / n:.1%}) in the exact right zone. "
             f"It named the right building {lv['Building']:.1%} of the time and the right floor {lv['Floor']:.1%}."]
    seen = [(z, r, s) for z, r, s in zip(labels, rec, sup) if s >= 5]
    if seen:
        z, r, s = min(seen, key=lambda t: t[1])
        if r < 0.999:
            parts.append(f"Weakest zone: <b>{z}</b> ({r:.0%} of {s} rows found).")
    off = np.array(cm, dtype=float)
    np.fill_diagonal(off, 0)
    if off.max() > 0:
        i, j = np.unravel_index(off.argmax(), off.shape)
        parts.append(f"Most common error: <b>{labels[i]}</b> predicted as <b>{labels[j]}</b> ({int(off[i, j])} rows).")
    return " ".join(parts)


models = load_models()
if not models:
    st.error("No saved models found in the model/ folder. Run train_models.py first.")
    st.stop()
names = expected_columns(models)
classes = list(next(iter(models.values())).classes_)
if names is None:
    st.error("Could not read the expected column list from the saved models.")
    st.stop()

# ---------------------------------------------------------------- left panel
with st.sidebar:
    st.markdown('<div class="sbname">SignalScope</div>', unsafe_allow_html=True)
    with st.container(border=True):
        data_head = st.empty()
        up = st.file_uploader("Upload a CSV", type="csv", label_visibility="collapsed")
        badge_slot = st.empty()
    with st.container(border=True):
        st.markdown('<div class="sbhead">Model</div>', unsafe_allow_html=True)
        choice = st.radio("Model", list(models), label_visibility="collapsed")
    with st.container(border=True):
        st.markdown('<div class="sbhead">Signal conditions</div>', unsafe_allow_html=True)
        sigma = st.slider("RSSI noise (dB, std dev)", 0, 12, 0)
        drop = st.slider("Access point dropout (%)", 0, 50, 0)
        st.markdown('<div class="sbnote">Optional. Adds random noise to the readings and drops some access points '
                    'before scoring.</div>', unsafe_allow_html=True)
    ready_slot = st.empty()

# ---------------------------------------------------------------- check the upload
state, checks, clean = "wait", None, None
if up is not None:
    try:
        raw_df, note = read_upload(up.getvalue())
        clean, checks = verify_csv(raw_df, names, classes, note)
        if clean is None:
            state = "fail"
        elif any(c[0] in ("fix", "warn") for c in checks):
            state = "fix"
        else:
            state = "ok"
    except Exception as exc:
        checks = [("fail", "File", f"This file could not be read as a CSV ({type(exc).__name__}). "
                                   "Export it again as comma-separated text with a header row.")]
        state = "fail"

n_fail = sum(1 for c in (checks or []) if c[0] == "fail")
data_head.markdown(f'<div class="sbhead"><span class="dot {state}"></span>Your data</div>', unsafe_allow_html=True)
if state == "wait":
    badge_slot.markdown('<div class="badge wait">No file yet</div>', unsafe_allow_html=True)
elif state == "fail":
    badge_slot.markdown(f'<div class="badge fail">&#10005; {n_fail} problem{"s" if n_fail != 1 else ""} to fix</div>',
                        unsafe_allow_html=True)
else:
    extra = "" if state == "ok" else ", fixed for you"
    badge_slot.markdown(f'<div class="badge {state}">&#10003; {len(clean):,} rows ready{extra}</div>',
                        unsafe_allow_html=True)

stress = "off" if not (sigma or drop) else f"{sigma} dB, {drop}% dropout"
status_rows = {"wait": ("wait", "waiting for a CSV"), "fail": ("fail", "needs fixes"),
               "ok": ("ok", "verified"), "fix": ("fix", "verified")}[state]
ready_slot.markdown(
    '<div class="ready">'
    f'<div class="rr"><span class="dot {status_rows[0]}"></span>Data<b>{status_rows[1]}</b></div>'
    f'<div class="rr"><span class="dot {"ok" if clean is not None else "wait"}"></span>Model<b>{html.escape(choice)}</b></div>'
    f'<div class="rr"><span class="dot {"wait" if clean is None else "warn" if (sigma or drop) else "ok"}"></span>Signal<b>{stress}</b></div>'
    f'<div class="rr"><span class="dot {"ok" if clean is not None else "wait"}"></span>Results'
    f'<b>{"on screen" if clean is not None else "not yet"}</b></div></div>', unsafe_allow_html=True)

# ---------------------------------------------------------------- main page
if state in ("wait", "fail"):
    st.markdown(hero_html([NO_SIGNAL] * len(names), "No data loaded yet. Upload a CSV to see a scan here."),
                unsafe_allow_html=True)
    st.markdown("### Format check")
    if state == "wait":
        st.markdown("Upload a CSV in the left panel. It is checked against the points below, and small problems "
                    "are corrected automatically.")
        st.markdown(checklist_html(waiting_checks(names, classes)), unsafe_allow_html=True)
        sample = os.path.join(BASE, "test_data.csv")
        if os.path.exists(sample):
            with open(sample, "rb") as fh:
                st.download_button("Download sample file (test_data.csv)", fh.read(),
                                   file_name="test_data.csv", mime="text/csv")
    else:
        st.markdown(f"The file failed {n_fail} check{'s' if n_fail != 1 else ''}, so no scores were calculated.")
        st.markdown(checklist_html(checks), unsafe_allow_html=True)
        st.markdown('<div class="fixhint">Fix the items marked in red and upload again. The layout should match '
                    '<code>test_data.csv</code>.</div>', unsafe_allow_html=True)
    st.stop()

X, y = clean[names], clean[TARGET]
if sigma or drop:
    X = pd.DataFrame(add_noise(X, sigma, drop / 100, seed=1), columns=names)
model = models[choice]
res, pred = run_model(choice, X, y, model)
labels = list(model.classes_)
cm = confusion_matrix(y, pred, labels=labels)
rec = recall_score(y, pred, labels=labels, average=None, zero_division=0)
sup = [int((y == z).sum()) for z in labels]
lv = level_accuracies(y, pred)

vals0 = X.iloc[0].values
st.markdown(hero_html(vals0, f"Example scan (row 0): it hears {int((vals0 > NO_SIGNAL).sum())} of {len(names)} "
                              f"access points, strongest {vals0.max():.0f} dBm. Taller bars are stronger signals."),
            unsafe_allow_html=True)

if sigma or drop:
    st.markdown(f'<div class="banner">Noise {sigma} dB and {drop}% access point dropout are applied. '
                'The scores below include this.</div>', unsafe_allow_html=True)
with st.expander(f"Format check passed ({len(checks)} checks)"):
    st.markdown(checklist_html(checks), unsafe_allow_html=True)

left, right = st.columns([5, 1])
left.markdown(f"#### {choice}, scored on {len(y):,} rows")
out = pd.DataFrame({"actual_zone": y.values, "predicted_zone": pred, "correct": y.values == pred})
right.download_button("Download predictions", out.to_csv(index=False), file_name="signalscope_predictions.csv",
                      mime="text/csv")
st.markdown(f'<div class="plain">{plain_words(choice, y, pred, lv, labels, cm, rec, sup)}</div>',
            unsafe_allow_html=True)
st.markdown(readouts([(k, res[k]) for k in ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]], HINTS),
            unsafe_allow_html=True)
st.caption(f"Prediction speed: {res['Inference ms/1k rows']:.1f} ms per 1,000 rows. "
           "Precision, recall and F1 are weighted over the 13 zones; AUC is one-vs-rest.")

t_cm, t_rep, t_lb, t_bf, t_row = st.tabs(["Confusion matrix", "Per-zone report", "Leaderboard",
                                           "Building & floor", "One row"])

with t_cm:
    view = st.radio("Show", ["Counts", "% of each actual zone"], horizontal=True)
    fig = confusion_figure(cm, labels, normalize=view != "Counts")
    st.pyplot(fig)
    plt.close(fig)

with t_rep:
    c1, c2 = st.columns([1, 1])
    rep = pd.DataFrame(classification_report(y, pred, output_dict=True, zero_division=0)).T.round(3)
    c1.markdown("**Classification report**")
    c1.dataframe(rep)
    c2.markdown("**Recall for each zone**")
    fig = zone_recall_figure(labels, rec, sup)
    c2.pyplot(fig)
    plt.close(fig)

with t_lb:
    comp = score_all(X, y, models).rename(columns={"Inference ms/1k rows": "Speed (ms per 1k rows)"})
    metric_cols = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]
    styled = (comp.style.format({**{c: "{:.4f}" for c in metric_cols}, "Speed (ms per 1k rows)": "{:.1f}"})
              .highlight_max(subset=metric_cols, color="#D7EFE4")
              .highlight_min(subset=["Speed (ms per 1k rows)"], color="#D7EFE4"))
    st.markdown("**All five models on the same rows** (green marks the best value in each column)")
    st.dataframe(styled)
    best = comp["Accuracy"].idxmax()
    st.success(f"Best accuracy here: {best} ({comp.loc[best, 'Accuracy']:.4f})")
    st.bar_chart(comp[["Accuracy", "F1", "MCC"]], color=[MAGENTA, ROSE, "#8A94A6"])

with t_bf:
    st.markdown("**Is the building right? The floor? Both?**")
    st.markdown(readouts([("Building correct", lv["Building"]), ("Floor correct", lv["Floor"]),
                          ("Exact zone correct", lv["Exact zone"])]), unsafe_allow_html=True)

with t_row:
    st.markdown("**Check one row**")
    st.caption("Each row is one WiFi scan. Pick a row to compare the model's prediction with the actual zone.")
    idx = int(st.number_input("Row number (0 is the first row)", min_value=0, max_value=len(X) - 1, value=0, step=1))
    row = X.iloc[[idx]]
    guess, truth = model.predict(row)[0], y.iloc[idx]
    gb, gf = split_zone(guess)
    tb, tf = split_zone(truth)
    a, b = st.columns([1, 1])
    a.markdown(f"Predicted: **building {gb}, floor {gf}**")
    a.markdown(f"Actual: **building {tb}, floor {tf}**")
    a.markdown("The model got this one **right**." if guess == truth else "The model got this one **wrong**.")
    top = pd.Series(model.predict_proba(row)[0], index=model.classes_).sort_values(ascending=False).head(3)
    a.caption("Probability for its top three zones (1.0 = completely sure)")
    a.bar_chart(top, color=MAGENTA)
    fig = building_figure(guess, truth)
    b.pyplot(fig)
    plt.close(fig)

st.markdown('<div class="foot">Data: UJIIndoorLoc (UCI).</div>', unsafe_allow_html=True)
