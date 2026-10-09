import openpyxl, os, json
import numpy as np, pandas as pd
from collections import defaultdict, Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import (roc_auc_score, f1_score, recall_score, precision_score,
                             brier_score_loss, confusion_matrix, roc_curve)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import lightgbm as lgb
import xgboost as xgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = r"D:\02XIHO Files\Daily Work\03Scientific Research\00Article_File\00Student\01单任务论文\多任务原始数据"
OUT  = r"D:\02XIHO Files\Daily Work\03Scientific Research\00Article_File\00Student\01单任务论文\PublicHealth_Submission"

def load(f):
    p=os.path.join(BASE,f); wb=openpyxl.load_workbook(p,read_only=True,data_only=True)
    ws=wb[wb.sheetnames[0]]; rows=list(ws.iter_rows(values_only=True)); wb.close()
    hdr=list(rows[0]); return [dict(zip(hdr,r)) for r in rows[1:] if r and any(c is not None for c in r)]
DISEASES=["高血压","糖尿病","癌症","肺部疾病","心脏病","中风","精神心理疾病","关节炎","血脂异常","肝脏疾病","肾脏疾病","消化系统疾病","哮喘","记忆力障碍"]
ALIGN={"脑卒中":"中风","精神疾病":"精神心理疾病","记忆障碍":"记忆力障碍"}
def flag(v):
    if v is None: return np.nan
    return 1.0 if str(v).strip()=="是" else 0.0
def yr(r,f): return r.get("调查年份") if f=="15-18外部验证.xlsx" else r.get("数据年份")

def build(f):
    data=load(f)
    for r in data:
        r["_y"]=yr(r,f); r["_n"]=sum(flag(r.get(c)) for c in DISEASES if r.get(c) is not None)
    by_id=defaultdict(list)
    for r in data: by_id[r["ID"]].append(r)
    out=[]
    for uid,recs in by_id.items():
        recs=sorted([x for x in recs if x["_y"] is not None], key=lambda x:x["_y"])
        if len(recs)<2: continue
        base,fol=recs[0],recs[1]
        if fol["_n"]<1: continue
        feat={}
        feat["年龄"]=base.get("年龄")
        feat["性别"]=1.0 if str(base.get("性别")).strip()=="男性" else 0.0
        feat["婚姻"]=1.0 if str(base.get("婚姻状况")).strip()=="已婚" else 0.0
        feat["退休"]=1.0 if str(base.get("退休状态") if f=="11-13" else base.get("是否退休")).strip()=="是" else 0.0
        feat["吸烟"]=1.0 if str(base.get("吸烟行为") if f=="11-13" else base.get("现在是否吸烟")).strip()=="是" else 0.0
        feat["饮酒"]=1.0 if str(base.get("饮酒行为") if f=="11-13" else base.get("现在是否饮酒")).strip()=="是" else 0.0
        feat["锻炼"]=1.0 if str(base.get("体育锻炼") if f=="11-13" else base.get("锻炼情况")).strip()=="是" else 0.0
        feat["血糖"]=base.get("血糖")
        for c in DISEASES: feat["慢病_"+c]=flag(base.get(ALIGN.get(c,c)))
        feat["_y"]=1.0 if fol["_n"]>=2 else 0.0
        out.append(feat)
    return pd.DataFrame(out)

dt=build("2011_2013关联规则.xlsx"); dv=build("15-18外部验证.xlsx")
FEATURES=["年龄","性别","婚姻","退休","吸烟","饮酒","锻炼","血糖"]+["慢病_"+c for c in DISEASES]
_med=pd.concat([dt[FEATURES],dv[FEATURES]]).median()
for d in [dt,dv]:
    for c in FEATURES: d[c]=d[c].astype(float).fillna(_med[c])
Xtr,Xte,ytr,yte=train_test_split(dt[FEATURES].values,dt["_y"].values,test_size=0.2,random_state=42,stratify=dt["_y"].values)
Xv,yv=dv[FEATURES].values,dv["_y"].values
print("train",len(ytr),"test",len(yte),"val",len(yv),"| val阳性率",round(yv.mean(),3))

MODELS={
 "LightGBM": lgb.LGBMClassifier(objective="binary",n_estimators=400,learning_rate=0.03,num_leaves=31,random_state=42,verbose=-1),
 "XGBoost": xgb.XGBClassifier(n_estimators=400,learning_rate=0.03,max_depth=6,subsample=0.8,colsample_bytree=0.8,reg_lambda=1.0,random_state=42,verbosity=0,n_jobs=-1),
 "RandomForest": RandomForestClassifier(n_estimators=400,max_depth=None,min_samples_leaf=20,random_state=42,n_jobs=-1),
 "Logistic": LogisticRegression(max_iter=1000,random_state=42),
 "SVM": SVC(probability=True,random_state=42),
}
probs={}; res={}
for name,m in MODELS.items():
    m.fit(Xtr,ytr)
    p_te=m.predict_proba(Xte)[:,1]; p_v=m.predict_proba(Xv)[:,1]
    probs[name]=(p_te,p_v)
    def ms(yt,p):
        pred=(p>=0.5).astype(int)
        tn,fp,fn,tp=confusion_matrix(yt,pred).ravel()
        return dict(auc=roc_auc_score(yt,p),f1=f1_score(yt,pred),sens=recall_score(yt,pred),
                    spec=(pred==0).astype(int)[yt==0].mean(),prec=precision_score(yt,pred,zero_division=0),
                    brier=brier_score_loss(yt,p))
    res[name]=dict(test=ms(yte,p_te), val=ms(yv,p_v))
    print(f"  {name}: val AUC={res[name]['val']['auc']:.3f} F1={res[name]['val']['f1']:.3f} Sens={res[name]['val']['sens']:.3f} Spec={res[name]['val']['spec']:.3f} Brier={res[name]['val']['brier']:.3f}")

# 主模型 LightGBM 用于运营指标与SHAP
pte,pvv=probs["LightGBM"]
def case_capture(yt,p,k=0.20):
    order=np.argsort(-p); n=len(yt); topn=int(n*k)
    return yt[order[:topn]].sum()/yt.sum(), topn
cc_te,_=case_capture(yte,pte); cc_v,_=case_capture(yv,pvv)
def net_benefit(yt,p,thrs):
    out={}
    for thr in thrs:
        pred=(p>=thr).astype(int); tp=((pred==1)&(yt==1)).sum(); fp=((pred==1)&(yt==0)).sum(); n=len(yt)
        out[thr]=(tp-fp*(thr/(1-thr)))/n
    return out
thrs=np.round(np.arange(0.05,0.96,0.05),2)
nb_v=net_benefit(yv,pvv,thrs); peak=max(nb_v,key=nb_v.get)
pos=[t for t in thrs if nb_v[t]>0]

# ===== 图 =====
# Fig7 校准
plt.figure(figsize=(4.3,3.8))
for nm,(p_te,p_v) in [("内部test",probs["LightGBM"]),("外部val",(pvv,pvv))]:
    pass
for nm,p in [("内部test",pte),("外部val",pvv)]:
    yt = yte if nm=="内部test" else yv
    bins=np.linspace(0,1,10); idx=np.digitize(p,bins); fp_,mp=[],[]
    for b in range(1,10):
        m=idx==b
        if m.sum()>0: fp_.append(yt[m].mean()); mp.append(p[m].mean())
    plt.plot(mp,fp_,"o-",ms=4,label=f"{nm} (Brier={brier_score_loss(yt,p):.3f})")
plt.plot([0,1],[0,1],"k--",lw=0.8); plt.xlabel("Predicted risk"); plt.ylabel("Observed incidence")
plt.title("Figure 7. Calibration"); plt.legend(fontsize=7); plt.tight_layout()
plt.savefig(os.path.join(OUT,"Figure7_Calibration.png"),dpi=200); plt.close()

# Fig8 ROC
plt.figure(figsize=(4.3,3.8))
for nm,p in [("内部test",pte),("外部val",pvv)]:
    yt=yte if nm=="内部test" else yv
    fpr,tpr,_=roc_curve(yt,p); plt.plot(fpr,tpr,label=f"{nm} (AUC={roc_auc_score(yt,p):.3f})")
plt.plot([0,1],[0,1],"k--",lw=0.8); plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
plt.title("Figure 8. ROC Curve"); plt.legend(fontsize=7); plt.tight_layout()
plt.savefig(os.path.join(OUT,"Figure8_ROC.png"),dpi=200); plt.close()

# Fig9 SHAP 依赖 (top3 特征)
import shap
EN={"年龄":"Age","性别":"Sex","婚姻":"Marital","退休":"Retired","吸烟":"Smoking","饮酒":"Drinking",
    "锻炼":"Exercise","血糖":"Glucose","慢病_高血压":"HyperTension","慢病_糖尿病":"Diabetes","慢病_癌症":"Cancer",
    "慢病_肺部疾病":"LungDisease","慢病_心脏病":"HeartDisease","慢病_中风":"Stroke","慢病_精神心理疾病":"PsychDisorder",
    "慢病_关节炎":"Arthritis","慢病_血脂异常":"Dyslipidemia","慢病_肝脏疾病":"LiverDisease","慢病_肾脏疾病":"KidneyDisease",
    "慢病_消化系统疾病":"GastroDisease","慢病_哮喘":"Asthma","慢病_记忆力障碍":"MemoryImpair"}
expl=shap.TreeExplainer(MODELS["LightGBM"])
sv=expl.shap_values(Xv)
if isinstance(sv,list): sv=sv[1]
imp=np.abs(sv).mean(0); top3=np.argsort(-imp)[:3]
fig,axes=plt.subplots(1,3,figsize=(11,3.4))
for ax,j in enumerate(top3):
    c=FEATURES[j]
    sc=axes[ax].scatter(Xv[:,j],sv[:,j],c=yv,cmap="coolwarm",s=5,alpha=0.5)
    axes[ax].set_xlabel(EN.get(c,c)); axes[ax].set_ylabel("SHAP value")
    axes[ax].set_title(f"({j+1}) {EN.get(c,c)}",fontsize=9)
plt.tight_layout(); plt.savefig(os.path.join(OUT,"Figure9_SHAP.png"),dpi=160); plt.close()

# Fig10 决策曲线
plt.figure(figsize=(4.3,3.8))
plt.plot(thrs,[yv.mean()]*len(thrs),"k:",lw=0.8,label="Screen all")
plt.plot([0,1],[0,0],"k--",lw=0.8,label="Screen none")
plt.plot(thrs,[nb_v[t] for t in thrs],"b-",lw=1.6,label="Model")
plt.xlabel("Threshold probability"); plt.ylabel("Net benefit")
plt.title("Figure 10. Decision Curve"); plt.legend(fontsize=7); plt.tight_layout()
plt.savefig(os.path.join(OUT,"Figure10_DecisionCurve.png"),dpi=200); plt.close()

# predictions.csv (主模型)
pd.DataFrame({"y_true":np.concatenate([yte,yv]),"p_pred":np.concatenate([pte,pvv]),
             "split":["test"]*len(yte)+["val"]*len(yv)}).to_csv(os.path.join(OUT,"predictions.csv"),index=False)

out_metrics=dict(
  features=FEATURES, n_features=len(FEATURES),
  n_train=int(len(ytr)),n_test=int(len(yte)),n_val=int(len(yv)),
  n_pos_train=int(ytr.sum()),n_pos_test=int(yte.sum()),n_pos_val=int(yv.sum()),
  models=res,
  lightgbm=dict(auc_test=float(res["LightGBM"]["test"]["auc"]),auc_val=float(res["LightGBM"]["val"]["auc"]),
    f1_val=float(res["LightGBM"]["val"]["f1"]),sens_val=float(res["LightGBM"]["val"]["sens"]),
    spec_val=float(res["LightGBM"]["val"]["spec"]),brier_val=float(res["LightGBM"]["val"]["brier"]),
    brier_test=float(res["LightGBM"]["test"]["brier"])),
  cc_test=float(cc_te),cc_val=float(cc_v),
  nb_val_peak=float(nb_v[peak]),nb_val_peak_thr=float(peak),nb_val_pos_range=[float(min(pos)),float(max(pos))],
  shap_top3=[FEATURES[j] for j in top3],
)
with open(os.path.join(OUT,"metrics.json"),"w") as f: json.dump(out_metrics,f,indent=2,ensure_ascii=False)
print("\nCC test=%.3f val=%.3f | NB peak=%.3f@%.2f range=[%.2f,%.2f]"%(cc_te,cc_v,nb_v[peak],peak,min(pos),max(pos)))
print("SHAP top3:",[FEATURES[j] for j in top3])
print("METRICS SAVED")
