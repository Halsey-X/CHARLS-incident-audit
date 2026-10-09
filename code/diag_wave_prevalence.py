# -*- coding: utf-8 -*-
"""Wave-by-wave disease prevalence: is apparent 'incidence' driven by between-wave
measurement/checklist changes rather than by real onset?"""
import os, pandas as pd, numpy as np
BASE = r"D:\02XIHO Files\Daily Work\03Scientific Research\00Article_File\00Student\01单任务论文\PublicHealth_Submission"
V5 = os.path.join(BASE, "04_Data_and_Results", "v5")
DISEASES = ["高血压","糖尿病","癌症","肺部疾病","心脏病","中风","精神心理疾病","关节炎",
            "血脂异常","肝脏疾病","肾脏疾病","消化系统疾病","哮喘","记忆力障碍"]

def load(fn):
    df = pd.read_csv(os.path.join(V5, fn)); df.columns = [str(c).strip() for c in df.columns]
    df = df.rename(columns={"数据年份": "wave", "调查年份": "wave",
                            "脑卒中": "中风", "精神疾病": "精神心理疾病", "记忆障碍": "记忆力障碍"})
    df["ID"] = pd.to_numeric(df["ID"], errors="coerce").astype("Int64")
    for c in DISEASES:
        df[c] = df[c].map(lambda v: 1 if str(v).strip() in ("是", "1", "1.0") else 0)
    n_null = df[DISEASES].isna().sum().sum()
    return df, n_null

print("=" * 108)
for fn in ["raw_wave_2011_2013_2015.csv", "raw_wave_2015_2018_2020.csv"]:
    df, _ = load(fn)
    waves = sorted(df.wave.dropna().unique())
    print(fn)
    print("  %-12s %s" % ("disease", "  ".join("w%-7d" % w for w in waves)))
    for c in DISEASES:
        cells = []
        for w in waves:
            s = df.loc[df.wave == w, c]
            cells.append("%6.2f%%" % (100 * s.mean()))
        print("  %-12s %s" % (c, "  ".join(cells)))
    print("  %-12s %s" % ("n", "  ".join("%-8d" % (df.wave == w).sum() for w in waves)))
    # how many people change condition count between adjacent waves
    print("  -- within-person consistency (same ID across the two adjacent waves) --")
    for wa, wb in zip(waves, waves[1:]):
        a = df[df.wave == wa].set_index("ID")[DISEASES]
        b = df[df.wave == wb].set_index("ID")[DISEASES]
        common = a.index.intersection(b.index)
        if len(common) < 50: continue
        a, b = a.loc[common], b.loc[common]
        newd = ((a == 0) & (b == 1)).sum(axis=1)
        lost = ((a == 1) & (b == 0)).sum(axis=1)
        print("     %d->%d  n=%4d | mean new %+.2f | mean lost %+.2f | unchanged-set %.1f%% | any new %.1f%%"
              % (wa, wb, len(common), newd.mean(), lost.mean(),
                 100 * ((newd == 0) & (lost == 0)).mean(), 100 * (newd > 0).mean()))
    print()
