# -*- coding: utf-8 -*-
"""Figure 2 data: capture-vs-capacity curves for the model and zero-cost rules.
Run with SYSTEM python (pandas/numpy/sklearn). Writes fig2_data.json."""
import os, json
import numpy as np, pandas as pd

BASE = r"D:\02XIHO Files\Daily Work\03Scientific Research\00Article_File\00Student\01单任务论文\PublicHealth_Submission"
V4 = os.path.join(BASE, "_Archive_旧版", "04_中间产物", "Intermediate", "Intermediate", "v4")
ARCH = os.path.join(BASE, "_Archive_旧版", "04_中间产物", "Intermediate", "Intermediate")
V5 = os.path.join(BASE, "04_Data_and_Results", "v5")

pv = pd.read_csv(os.path.join(V4, "predictions_val.csv"))
print("predictions_val columns:", list(pv.columns))
p = np.load(os.path.join(V4, "pred_val_LightGBM.npy"))
if p.ndim == 2: p = p[:, 1]
y = pv["y"].values.astype(int)
print("n=", len(y), "events=", y.sum(), "model AUC=",
      round(__import__("sklearn.metrics", fromlist=["roc_auc_score"]).roc_auc_score(y, p), 4))

if "baseline_n_disease" in pv.columns:
    cnt = pv["baseline_n_disease"].values.astype(int)
else:
    dv = pd.read_csv(os.path.join(ARCH, "dataset_validation_2015_2018.csv"), encoding="utf-8-sig")
    if "ID" in pv.columns:
        m = pv[["ID"]].merge(dv[["ID", "baseline_n_disease"]], on="ID", how="left")
        cnt = m["baseline_n_disease"].values.astype(float)
    else:
        cnt = dv["baseline_n_disease"].values.astype(float)
print("baseline count aligned:", np.isnan(cnt).sum() if cnt.dtype == float else 0, "mean=", np.nanmean(cnt))

caps = np.round(np.arange(0.05, 0.55, 0.05), 2)
def capture_curve(score, y, caps):
    order = np.argsort(-score)          # highest score first
    ys = y[order]
    n = len(y)
    out = []
    for c in caps:
        k = int(round(c * n))
        out.append(float(ys[:k].sum() / y.sum()))
    return out

curves = {
    "model": capture_curve(p, y, caps),
    "count_reversed": capture_curve(-cnt.astype(float), y, caps),   # fewest conditions first
    "count_natural": capture_curve(cnt.astype(float), y, caps),
}
rnd = [float(c) for c in caps]

from sklearn.metrics import roc_auc_score
rules = {}
d1 = pd.read_csv(os.path.join(ARCH, "dataset_derivation_2011_2013.csv"), encoding="utf-8-sig")
pt = pd.read_csv(os.path.join(V4, "predictions_test.csv"))
pte = np.load(os.path.join(V4, "pred_test_LightGBM.npy"))
if pte.ndim == 2: pte = pte[:, 1]
yte = pt["y"].values.astype(int)
if "baseline_n_disease" in pt.columns:
    cnt_te = pt["baseline_n_disease"].values.astype(float)
else:
    mte = pt[["ID"]].merge(d1[["ID", "baseline_n_disease"]], on="ID", how="left")
    assert mte["baseline_n_disease"].notna().all()
    cnt_te = mte["baseline_n_disease"].values.astype(float)

def pack(yv, model_p, c):
    return {
        "model": round(float(roc_auc_score(yv, model_p)), 4),
        "count_natural": round(float(roc_auc_score(yv, c)), 4),
        "count_reversed": round(float(roc_auc_score(yv, -c)), 4),
    }

data = {
    "caps": list(caps),
    "curves": curves,
    "random": rnd,
    "internal_test": pack(yte, pte, cnt_te),
    "temporal_validation": pack(y, p, cnt.astype(float)),
    "capture_top20": {
        "model_test": float((pte >= np.quantile(pte, 0.8)).sum() and 0),  # placeholder
    },
}
def top20(score, yv):
    k = int(round(0.2 * len(yv)))
    idx = np.argsort(-score)[:k]
    return float(yv[idx].sum() / yv.sum())
data["capture_top20"] = {
    "model_test": round(top20(pte, yte), 4),
    "count_reversed_test": round(top20(-cnt_te, yte), 4),
    "model_val": round(top20(p, y), 4),
    "count_reversed_val": round(top20(-cnt.astype(float), y), 4),
}
with open(os.path.join(V5, "fig2_data.json"), "w", encoding="utf-8") as fh:
    json.dump(data, fh, indent=1)
print(json.dumps({k: v for k, v in data.items() if k != "curves"}, indent=1))
print("curves model@0.2:", curves["model"][3], "count_rev@0.2:", curves["count_reversed"][3],
      "natural@0.2:", curves["count_natural"][3], "random:", rnd[3])
print("wrote fig2_data.json")
