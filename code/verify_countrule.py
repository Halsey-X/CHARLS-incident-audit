# -*- coding: utf-8 -*-
"""Decisive check: is the zero-cost "baseline condition count" rule being tested
in its best orientation?  Compute AUC and top-20% capture for BOTH orientations,
plus the ceiling-effect table d(incident rate)/d(baseline count)."""
import csv, os, statistics as st

BASE = r"D:\02XIHO Files\Daily Work\03Scientific Research\00Article_File\00Student\01单任务论文\PublicHealth_Submission"
INT = os.path.join(BASE, "_Archive_旧版", "04_中间产物", "Intermediate", "Intermediate")

def load(fn):
    with open(os.path.join(INT, fn), encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))

def auc(scores, labels):
    """Mann-Whitney AUC with ties handled by average rank."""
    pairs = sorted(zip(scores, labels))
    n = len(pairs)
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and pairs[j + 1][0] == pairs[i][0]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[k] = avg
        i = j + 1
    n1 = sum(1 for _, y in pairs if y == 1)
    n0 = n - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    r1 = sum(r for r, (_, y) in zip(ranks, pairs) if y == 1)
    return (r1 - n1 * (n1 + 1) / 2.0) / (n1 * n0)

def capture_at_k(scores, labels, k_frac):
    """Fraction of all cases captured when the top k_frac by score are screened."""
    order = sorted(range(len(scores)), key=lambda i: -scores[i])
    k = int(round(len(scores) * k_frac))
    tot = sum(labels)
    hit = sum(labels[i] for i in order[:k])
    return hit / tot if tot else float("nan")

for name, fn in [("derivation 2011-13 (n=2,638)", "dataset_derivation_2011_2013.csv"),
                 ("validation 2015-18 (n=3,740)", "dataset_validation_2015_2018.csv")]:
    rows = load(fn)
    y = [int(float(r["y"])) for r in rows]
    cnt = [float(r["baseline_n_disease"]) for r in rows]
    print("=" * 78)
    print(name, " events =", sum(y), " (%.1f%%)" % (100 * sum(y) / len(y)))
    print("  baseline_n_disease distribution:")
    for c in sorted(set(cnt)):
        idx = [i for i, v in enumerate(cnt) if v == c]
        ev = sum(y[i] for i in idx)
        print("     count=%2d  n=%5d (%5.2f%%)  events=%5d  rate=%6.2f%%"
              % (c, len(idx), 100 * len(idx) / len(cnt), ev, 100 * ev / len(idx)))
    a_nat = auc(cnt, y)
    a_rev = auc([-v for v in cnt], y)
    print("  AUC count, natural (more = higher risk) : %.4f" % a_nat)
    print("  AUC count, REVERSED (fewer = higher risk): %.4f" % a_rev)
    for kf in (0.10, 0.20, 0.30):
        print("  top-%d%% capture: natural %.4f | reversed %.4f | random %.4f"
              % (kf * 100, capture_at_k(cnt, y, kf), capture_at_k([-v for v in cnt], y, kf), kf))

# model capture, using the archived per-sample predictions
pr = load("predictions_all_models.csv")
val = [r for r in pr if r["split"] == "temporal_validation"]
test = [r for r in pr if r["split"] == "internal_test"]
print("=" * 78)
print("predictions_all_models.csv: splits =", sorted(set(r["split"] for r in pr)),
      "| sizes:", {s: sum(1 for r in pr if r["split"] == s) for s in set(r["split"] for r in pr)})
for sname, sub in [("internal_test", test), ("temporal_validation", val)]:
    if not sub:
        continue
    yy = [int(float(r["y_true"])) for r in sub]
    for m in ["p_LightGBM", "p_Logistic", "p_XGBoost", "p_SVM", "p_RandomForest"]:
        sc = [float(r[m]) for r in sub]
        print("  %-16s %-14s AUC=%.4f  top20%%=%.4f" % (sname, m, auc(sc, yy), capture_at_k(sc, yy, 0.20)))
