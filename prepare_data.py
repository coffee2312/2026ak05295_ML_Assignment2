"""Turn UJIIndoorLoc trainingData.csv into a compact, cleaned dataset.
Usage: python prepare_data.py trainingData.csv
"""
import sys
import numpy as np
import pandas as pd
from common import NO_SIGNAL, TARGET

APS_PER_BUILDING = None  # None = keep every AP ever detected (~465 of 520); or an int, e.g. 10 per building
SAMPLE_ROWS = None  # None = use all rows; set an int for a stratified sample
SEED = 7            # change this: it makes your split/sample unique

src = sys.argv[1] if len(sys.argv) > 1 else "trainingData.csv"
df = pd.read_csv(src)
wap_cols = [c for c in df.columns if c.startswith("WAP")]

df[wap_cols] = df[wap_cols].replace(100, NO_SIGNAL)
# Pick APs per building so every building keeps usable signal coverage
if APS_PER_BUILDING is None:
    keep = [c for c in wap_cols if (df[c] > NO_SIGNAL).any()]   # drop never-detected APs
else:
    keep = []
    for b, grp in df.groupby("BUILDINGID"):
        rate = (grp[wap_cols] > NO_SIGNAL).mean().sort_values(ascending=False)
        keep += [c for c in rate.index[:APS_PER_BUILDING] if c not in keep]

zone = "B" + df["BUILDINGID"].astype(str) + "_F" + df["FLOOR"].astype(str)
out = pd.concat([df[keep], zone.rename(TARGET)], axis=1)
out = out[(out[keep] > NO_SIGNAL).any(axis=1)]          # drop rows with no AP seen

if SAMPLE_ROWS and len(out) > SAMPLE_ROWS:
    frac = SAMPLE_ROWS / len(out)
    out = out.groupby(TARGET).sample(frac=frac, random_state=SEED)
out = out.reset_index(drop=True)
out.to_csv("prepared_data.csv", index=False)
print(f"Saved prepared_data.csv: {out.shape[0]} rows, {len(keep)} AP features")
print(out[TARGET].value_counts())
