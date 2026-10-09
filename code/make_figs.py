import openpyxl, os
import numpy as np, pandas as pd
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.metrics import brier_score_loss
import lightgbm as lgb
import shap
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

BASE = r"D:\02XIHO Files\Daily Work\03Scientific Research\00Article_File\00Student\01单任务论文\多任务原始数据"
OUT  = r"D:\02XIHO Files\Daily Work\03Scientific Research\00Article_File\00Student\01单任务论文\PublicHealth_Submission"
DISEASES=["高血压","糖尿病","癌症","肺部疾病","心脏病","中风","精神心理疾病","关节炎","血脂异常","肝脏疾病","肾脏疾病","消化系统疾病","哮喘","记忆力障碍"]
ALIGN={"脑卒中":"中风","精神疾病":"精神心理疾病","记忆障碍":"记忆力障碍"}
EN={"年龄":"Age","性别":"Sex","婚姻":"Marital","退休":"Retired","吸烟":"Smoking","饮酒":"Drinking",
    "锻炼":"Exercise","血糖":"Glucose","慢病_高血压":"Hypertension","慢病_糖尿病":"Diabetes","慢病_癌症":"Cancer",
    "慢病_肺部疾病":"LungDisease","慢病_心脏病":"HeartDisease","慢病_中风":"Stroke","慢病_精神心理疾病":"PsychDisorder",
    "慢病_关节炎":"Arthritis","慢病_血脂异常":"Dyslipidemia","慢病_肝脏疾病":"LiverDisease","慢病_肾脏疾病":"KidneyDisease",
    "慢病_消化系统疾病":"GastroDisease","慢病_哮喘":"Asthma","慢病_记忆力障碍":"MemoryImpair"}
def flag(v):
    if v is None: return np.nan
    return 1.0 if str(v).strip()=="是" else 0.0
def load(f):
    p=os.path.join(BASE,f); wb=openpyxl.load_workbook(p,read_only=True,data_only=True)
    ws=wb[wb.sheetnames[0]]; rows=list(ws.iter_rows(values_only=True)); wb.close()
    hdr=list(rows[0]); return [dict(zip(hdr,r)) for r in rows[1:] if r and any(c is not None for c in r)]
def build(f):
    data=load(f)
    for r in data:
        r["_y"]=(r.get("调查年份") if f=="15-18外部验证.xlsx" else r.get("数据年份"))
        r["_n"]=sum(flag(r.get(c)) for c in DISEASES if r.get(c) is not None)
    by=defaultdict(list)
    for r in data: by[r["ID"]].append(r)
    out=[]
    for uid,recs in by.items():
        recs=sorted([x for x in recs if x["_y"] is not None], key=lambda x:x["_y"])
        if len(recs)<2: continue
        b,fo=recs[0],recs[1]
        if fo["_n"]<1: continue
        fe={}
        fe["年龄"]=b.get("年龄")
        fe["性别"]=1.0 if str(b.get("性别")).strip()=="男性" else 0.0
        fe["婚姻"]=1.0 if str(b.get("婚姻状况")).strip()=="已婚" else 0.0
        fe["退休"]=1.0 if str(b.get("退休状态") if f=="11-13" else b.get("是否退休")).strip()=="是" else 0.0
        fe["吸烟"]=1.0 if str(b.get("吸烟行为") if f=="11-13" else b.get("现在是否吸烟")).strip()=="是" else 0.0
        fe["饮酒"]=1.0 if str(b.get("饮酒行为") if f=="11-13" else b.get("现在是否饮酒")).strip()=="是" else 0.0
        fe["锻炼"]=1.0 if str(b.get("体育锻炼") if f=="11-13" else b.get("锻炼情况")).strip()=="是" else 0.0
        fe["血糖"]=b.get("血糖")
        for c in DISEASES: fe["慢病_"+c]=flag(b.get(ALIGN.get(c,c)))
        fe["_y"]=1.0 if fo["_n"]>=2 else 0.0
        out.append(fe)
    return pd.DataFrame(out)
dt=build("2011_2013关联规则.xlsx"); dv=build("15-18外部验证.xlsx")
F=["年龄","性别","婚姻","退休","吸烟","饮酒","锻炼","血糖"]+["慢病_"+c for c in DISEASES]
_med=pd.concat([dt[F],dv[F]]).median()
for d in [dt,dv]:
    for c in F: d[c]=d[c].astype(float).fillna(_med[c])
Xtr,Xte,ytr,yte=train_test_split(dt[F].values,dt["_y"].values,test_size=0.2,random_state=42,stratify=dt["_y"].values)
Xv,yv=dv[F].values,dv["_y"].values
m=lgb.LGBMClassifier(objective="binary",n_estimators=400,learning_rate=0.03,num_leaves=31,random_state=42,verbose=-1).fit(Xtr,ytr)
pte=m.predict_proba(Xte)[:,1]; pvv=m.predict_proba(Xv)[:,1]

# Fig7 校准
plt.figure(figsize=(4.4,3.8))
for nm,p,yt in [("Internal test",pte,yte),("Temporal validation",pvv,yv)]:
    bins=np.linspace(0,1,10); idx=np.digitize(p,bins); fp_,mp=[],[]
    for b in range(1,10):
        msk=idx==b
        if msk.sum()>0: fp_.append(yt[msk].mean()); mp.append(p[msk].mean())
    plt.plot(mp,fp_,"o-",ms=4,label=f"{nm} (Brier={brier_score_loss(yt,p):.3f})")
plt.plot([0,1],[0,1],"k--",lw=0.8); plt.xlabel("Predicted risk"); plt.ylabel("Observed multimorbidity")
plt.title("Figure 7. Calibration curves"); plt.legend(fontsize=7); plt.tight_layout()
plt.savefig(os.path.join(OUT,"Figure7_Calibration.png"),dpi=200); plt.close()

# SHAP
expl=shap.TreeExplainer(m); sv=expl.shap_values(Xv)
if isinstance(sv,list): sv=sv[1]
imp=np.abs(sv).mean(0); order=np.argsort(-imp)

# Fig8 SHAP 全局重要性 (bar, top12)
top=order[:12][::-1]
plt.figure(figsize=(4.6,4.0))
plt.barh(range(len(top)), imp[top], color="#4c72b0")
plt.yticks(range(len(top)), [EN.get(F[j],F[j]) for j in top], fontsize=7)
plt.xlabel("mean |SHAP value| (impact on model output)")
plt.title("Figure 8. SHAP global feature importance"); plt.tight_layout()
plt.savefig(os.path.join(OUT,"Figure8_SHAP_importance.png"),dpi=200); plt.close()

# Fig9 SHAP 依赖 (top3)
top3=order[:3]
fig,axes=plt.subplots(1,3,figsize=(11,3.4))
for ax,j in enumerate(top3):
    c=F[j]
    axes[ax].scatter(Xv[:,j],sv[:,j],c=yv,cmap="coolwarm",s=5,alpha=0.5)
    axes[ax].set_xlabel(EN.get(c,c)); axes[ax].set_ylabel("SHAP value")
    axes[ax].set_title(f"({j+1}) {EN.get(c,c)}",fontsize=9)
plt.tight_layout(); plt.savefig(os.path.join(OUT,"Figure9_SHAP.png"),dpi=160); plt.close()

# Fig10 决策曲线
def nb(yt,p,thrs):
    o={}
    for t in thrs:
        pred=(p>=t).astype(int); tp=((pred==1)&(yt==1)).sum(); fp=((pred==1)&(yt==0)).sum(); n=len(yt)
        o[t]=(tp-fp*(t/(1-t)))/n
    return o
thrs=np.round(np.arange(0.05,0.96,0.05),2); nbv=nb(yv,pvv,thrs)
plt.figure(figsize=(4.4,3.8))
plt.plot([0,1],[0,0],"k--",lw=0.8,label="Screen none")
plt.plot(thrs,[yv.mean()]*len(thrs),":",color="gray",lw=0.8,label="Screen all")
plt.plot(thrs,[nbv[t] for t in thrs],"-",color="#b04c4c",lw=1.8,label="Risk-based targeting")
plt.xlabel("Threshold probability"); plt.ylabel("Net benefit")
plt.title("Figure 10. Decision curve analysis"); plt.legend(fontsize=7); plt.tight_layout()
plt.savefig(os.path.join(OUT,"Figure10_DecisionCurve.png"),dpi=200); plt.close()
print("FIGURES OK: Fig7, Fig8(importance), Fig9(dependence), Fig10(decision curve)")
