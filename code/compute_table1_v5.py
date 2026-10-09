# -*- coding: utf-8 -*-
"""Table 1 — baseline characteristics of the derivation and temporal-validation cohorts.
Run with SYSTEM python (pandas)."""
import os, json
import pandas as pd
import numpy as np

BASE = r"D:\02XIHO Files\Daily Work\03Scientific Research\00Article_File\00Student\01单任务论文\PublicHealth_Submission"
ARCH = os.path.join(BASE, "_Archive_旧版", "04_中间产物", "Intermediate", "Intermediate")
V5 = os.path.join(BASE, "04_Data_and_Results", "v5")

CONT = [("年龄","Age, years"),("睡眠","Sleep duration, h"),("BMI","Body mass index, kg/m2"),
        ("收缩压","Systolic BP, mmHg"),("舒张压","Diastolic BP, mmHg"),("血糖","Fasting glucose, mmol/L"),
        ("抑郁","Depressive symptoms (CES-D10 score)"),("baseline_n_disease","Chronic conditions at baseline, n")]
BIN = [("性别","Male sex"),("婚姻","Married or partnered"),("教育","Education (primary or above)"),
       ("退休","Retired"),("吸烟","Current smoking"),("饮酒","Current drinking"),
       ("锻炼","Regular physical exercise"),("自评健康","Self-rated health as fair or poor")]

def ms(s):
    s = pd.to_numeric(s, errors="coerce")
    return "%s (%s)" % (round(s.mean(),1), round(s.std(),1))

def pct(s, pos):
    s = pd.to_numeric(s, errors="coerce")
    return "%.1f" % (100*s.apply(lambda v: str(v).strip() in pos).mean())

rows = []
out = {}
for cohort, fn, byr in (("derivation","dataset_derivation_2011_2013.csv",2011),
                        ("validation","dataset_validation_2015_2018.csv",2015)):
    d = pd.read_csv(os.path.join(ARCH, fn), encoding="utf-8-sig")
    rec = {"n": len(d), "baseline_year": byr}
    for col, label in CONT:
        rec[label] = ms(d[col])
    for col, label in BIN:
        pos = ["男","1","1.0","已婚","有配偶","是","差","一般或差"] if col!="自评健康" else ["差","一般或差"]
        # generic: treat coded 1/是 as positive; for 自评健康 use 差/一般或差
        if col == "自评健康":
            pos = ["差","一般或差"]
        elif col == "教育":
            pos = ["小学","初中","高中","大专及以上","小学及以上","1","1.0"]
        rec[label] = pct(d[col], pos)
    rec["Depressive symptoms (CES-D10 >=10), %"] = "%.1f" % (100*(pd.to_numeric(d["抑郁"],errors="coerce")>=10).mean())
    out[cohort] = rec

for k, v in out.items():
    print("="*20, k, "n=", v["n"])
    for kk, vv in v.items():
        print("   %-46s %s" % (kk, vv))

with open(os.path.join(V5, "table1_baseline.json"), "w", encoding="utf-8") as fh:
    json.dump(out, fh, ensure_ascii=False, indent=1)
print("\nwrote table1_baseline.json")
