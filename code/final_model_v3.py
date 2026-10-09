"""
最终建模 v3 —— 结局：随访波次新增 >=1 种慢性病（incident condition）
 + 全量评估：判别(AUC+95%CI)、校准(截距/斜率/Brier)、增量价值、亚组、预算-捕获、决策曲线
 + 输出无标题的 PNG/SVG/EPS 图件与全部表格
"""
import os, json, warnings
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, f1_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import lightgbm as lgb
import xgboost as xgb
from scipy.stats import norm

warnings.filterwarnings("ignore")
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                     "axes.edgecolor": "#333333", "axes.linewidth": 0.8,
                     "svg.fonttype": "none", "ps.fonttype": 42})

BASE = r'D:\02XIHO Files\Daily Work\03Scientific Research\00Article_File\00Student\01单任务论文\PublicHealth_Submission'
INTER = os.path.join(BASE, '04_Data_and_Results', 'Intermediate')
DATA = os.path.join(BASE, '04_Data_and_Results')
TAB = os.path.join(BASE, '03_Tables')
FIG = os.path.join(BASE, '02_Figures', '02-2_ResultFigures')
os.makedirs(TAB, exist_ok=True); os.makedirs(FIG, exist_ok=True)
os.makedirs(os.path.join(INTER, 'v3'), exist_ok=True)

DISEASES = ['高血压','糖尿病','癌症','肺部疾病','心脏病','中风','精神心理疾病','关节炎',
            '血脂异常','肝脏疾病','肾脏疾病','消化系统疾病','哮喘','记忆力障碍']
FEATURES = ['年龄','性别','婚姻','教育','退休','吸烟','饮酒','锻炼','睡眠','BMI',
            '收缩压','舒张压','血糖','抑郁','自评健康'] + ['慢病_'+c for c in DISEASES]
EN = {'年龄':'Age','性别':'Sex','婚姻':'Married','教育':'Education','退休':'Retired',
      '吸烟':'Smoking','饮酒':'Alcohol','锻炼':'Exercise','睡眠':'Sleep duration',
      'BMI':'BMI','收缩压':'Systolic BP','舒张压':'Diastolic BP','血糖':'Glucose',
      '抑郁':'Depression (CES-D10)','自评健康':'Self-rated health'}
for c in DISEASES:
    EN['慢病_'+c] = {'高血压':'Hypertension','糖尿病':'Diabetes','癌症':'Cancer',
        '肺部疾病':'Lung disease','心脏病':'Heart disease','中风':'Stroke',
        '精神心理疾病':'Psychiatric disorder','关节炎':'Arthritis','血脂异常':'Dyslipidaemia',
        '肝脏疾病':'Liver disease','肾脏疾病':'Kidney disease','消化系统疾病':'Digestive disease',
        '哮喘':'Asthma','记忆力障碍':'Memory disorder'}[c]
FN = [EN.get(f, f) for f in FEATURES]

d1 = pd.read_csv(os.path.join(INTER, 'dataset_derivation_2011_2013.csv'))
d2 = pd.read_csv(os.path.join(INTER, 'dataset_validation_2015_2018.csv'))
for d in (d1, d2):
    d['y'] = ((d['followup_n_disease'] - d['baseline_n_disease']) >= 1).astype(int)

# 统一缺失值填充
for c in FEATURES:
    med = pd.concat([d1[c], d2[c]]).astype(float).median()
    d1[c] = d1[c].astype(float).fillna(med); d2[c] = d2[c].astype(float).fillna(med)

X = d1[FEATURES].astype(float).values; Y = d1['y'].values
Xv = d2[FEATURES].astype(float).values; Yv = d2['y'].values
Xtr, Xte, Ytr, Yte = train_test_split(X, Y, test_size=0.2, random_state=42, stratify=Y)
print("n_train=%d (events %d) n_test=%d (events %d) n_val=%d (events %d)" %
      (len(Ytr), Ytr.sum(), len(Yte), Yte.sum(), len(Yv), Yv.sum()))

# ---------------- 模型 ----------------
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

MODELS = {
 "LightGBM": lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31,
                                max_depth=6, reg_lambda=0.1, random_state=42, verbose=-1),
 "XGBoost": xgb.XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=6,
                              reg_lambda=0.1, random_state=42, eval_metric='logloss', verbosity=0),
 "RandomForest": RandomForestClassifier(n_estimators=400, random_state=42, n_jobs=-1),
 "Logistic": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=42)),
 "SVM": make_pipeline(StandardScaler(), SVC(probability=True, random_state=42)),
}

def delong_ci(y, p, alpha=0.05):
    y = np.asarray(y, int); p = np.asarray(p, float)
    order = np.argsort(p); ps, ys = p[order], y[order]
    n1 = int(ys.sum()); n0 = len(ys) - n1
    if n1 == 0 or n0 == 0: return np.nan, np.nan, np.nan
    ranks = np.empty(len(ps)); i = 0
    while i < len(ps):
        j = i
        while j+1 < len(ps) and ps[j+1] == ps[i]: j += 1
        ranks[i:j+1] = (i+j+2)/2.0; i = j+1
    r1 = ranks[ys == 1]; r0 = ranks[ys == 0]
    V10 = (r1 - np.searchsorted(r0, r1, side='left')/1.0)/n0
    V10 = np.array([(ranks[ys == 1][k] - np.searchsorted(r0, ranks[ys == 1][k], side='left'))/n0
                    for k in range(n1)])
    V01 = np.array([1 - np.searchsorted(r1, r, side='right')/n1 for r in r0])
    auc = roc_auc_score(y, p)
    se = np.sqrt(V10.var(ddof=1)/n1 + V01.var(ddof=1)/n0)
    z = norm.ppf(1-alpha/2)
    return auc, max(0., auc-z*se), min(1., auc+z*se)

def perf(y, p, thr=0.5):
    y = np.asarray(y, int)
    a, lo, hi = delong_ci(y, p)
    pred = (np.asarray(p) >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0,1]).ravel()
    lp = np.log(np.clip(p,1e-6,1-1e-6)/(1-np.clip(p,1e-6,1-1e-6))).reshape(-1,1)
    try:
        lr = LogisticRegression(C=1e6, max_iter=1000).fit(lp, y); itc, slp = lr.intercept_[0], lr.coef_[0][0]
    except Exception: itc, slp = np.nan, np.nan
    return dict(auc=round(a,4), auc_ci_low=round(lo,4), auc_ci_high=round(hi,4),
                f1=round(f1_score(y,pred),4), sensitivity=round(tp/(tp+fn),4) if tp+fn else np.nan,
                specificity=round(tn/(tn+fp),4) if tn+fp else np.nan,
                brier=round(float(np.mean((np.asarray(p)-y)**2)),4),
                calib_intercept=round(itc,4), calib_slope=round(slp,4))

res = {}
for nm, m in MODELS.items():
    m.fit(Xtr, Ytr)
    pte = m.predict_proba(Xte)[:,1]; pva = m.predict_proba(Xv)[:,1]
    res[nm] = dict(test=perf(Yte,pte), val=perf(Yv,pva))
    print("%-13s TEST auc=%.4f[%.4f-%.4f] | VAL auc=%.4f[%.4f-%.4f] brier=%.4f slope=%.3f" %
          (nm, res[nm]['test']['auc'], res[nm]['test']['auc_ci_low'], res[nm]['test']['auc_ci_high'],
           res[nm]['val']['auc'], res[nm]['val']['auc_ci_low'], res[nm]['val']['auc_ci_high'],
           res[nm]['val']['brier'], res[nm]['val']['calib_slope']))
    np.save(os.path.join(INTER,'v3','pred_val_%s.npy'%nm), pva)
    np.save(os.path.join(INTER,'v3','pred_test_%s.npy'%nm), pte)

MAIN = "LightGBM"
pv = np.load(os.path.join(INTER,'v3','pred_val_%s.npy'%MAIN))
pt = np.load(os.path.join(INTER,'v3','pred_test_%s.npy'%MAIN))

# ---------------- 增量价值 ----------------
def auc_only(y,s):
    try: return roc_auc_score(np.asarray(y,int), np.asarray(s,float))
    except Exception: return np.nan
inc = []
for nm,(yy,pp,dd) in [('internal_test',(Yte,pt,d1.iloc[train_test_split(np.arange(len(d1)),test_size=0.2,random_state=42,stratify=Y)[1]])),
                      ('temporal_validation',(Yv,pv,d2))]:
    af = auc_only(yy,pp)
    row = dict(sample=nm, auc_model=round(af,4))
    for lab, s in [('baseline_condition_count', dd['baseline_n_disease'].astype(float).values),
                   ('age', dd['年龄'].astype(float).values),
                   ('systolic_bp', dd['收缩压'].astype(float).values),
                   ('depression_score', dd['抑郁'].astype(float).values),
                   ('bmi', dd['BMI'].astype(float).values)]:
        row['auc_'+lab] = round(auc_only(yy,s),4)
        row['delta_'+lab] = round(af - auc_only(yy,s),4)
    inc.append(row)
inc_df = pd.DataFrame(inc)
inc_df.to_csv(os.path.join(TAB,'TableS9_incremental_value.csv'), index=False)
print("\n增量价值:\n", inc_df.to_string(index=False))

# ---------------- 捕获率-预算曲线 + 资源节约 ----------------
def cap(y,p,k):
    y=np.asarray(y,float); idx=np.argsort(-np.asarray(p))[:int(round(len(y)*k))]
    return float(y[idx].sum()/y.sum())
budget=[]
for k in [0.05,0.10,0.15,0.20,0.25,0.30,0.40,0.50,0.60,0.80,1.00]:
    cv, ct = cap(Yv,pv,k), cap(Yte,pt,k)
    budget.append(dict(budget_fraction=k, capture_validation=round(cv,4), capture_test=round(ct,4),
                       lift_validation=round(cv/k,3), lift_test=round(ct/k,3)))
bud=pd.DataFrame(budget)
bud.to_csv(os.path.join(TAB,'TableS3_capture_by_budget.csv'), index=False)
c20 = cap(Yv,pv,0.20)
save = 1 - 0.20/c20 if c20 > 0 else np.nan
print("\n预算-捕获:\n", bud.to_string(index=False))
print("top20%%捕获=%.4f  等效资源节约=%.1f%%" % (c20, 100*save))

# ---------------- 决策曲线 ----------------
def nbv(y,p,t):
    n=len(y); f=(np.asarray(p)>=t).astype(int)
    tp=((f==1)&(y==1)).sum(); fp=((f==1)&(y==0)).sum()
    return float(tp/n - fp/n*(t/(1-t)))
thrs=[0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90]
nbr=[dict(threshold=t, nb_model=round(nbv(Yv,pv,t),4),
          nb_screen_all=round(nbv(Yv,np.ones_like(pv),t),4), nb_screen_none=0.0) for t in thrs]
pd.DataFrame(nbr).to_csv(os.path.join(TAB,'TableS2_net_benefit_by_threshold.csv'), index=False)
nbmax=max(nbr,key=lambda r:r['nb_model'])

# ---------------- 亚组 ----------------
def bmi_g(b): return 'Normal (<24)' if b<24 else ('Overweight (24-28)' if b<28 else 'Obese (>=28)')
def edu_g(e): return 'Primary or below' if float(e)<=1 else ('Junior high' if float(e)==2 else 'Senior high or above')
subs=[]
def add(label, grp, mask):
    yy,pp = Yv[mask], pv[mask]
    if len(yy)<100 or yy.sum()<20 or (1-yy).sum()<20:
        subs.append(dict(subgroup=label,group=grp,n=int(len(yy)),events=int(yy.sum()),auc=np.nan,
                         ci_low=np.nan,ci_high=np.nan,note='not estimable')); return
    a,lo,hi = delong_ci(yy,pp)
    subs.append(dict(subgroup=label,group=grp,n=int(len(yy)),events=int(yy.sum()),
                     auc=round(a,4),ci_low=round(lo,4),ci_high=round(hi,4),note=''))
for g,m in [('45-59',d2['年龄']<60),('>=60',d2['年龄']>=60)]: add('Age',g,m.values)
for g,m in [('Male',d2['性别']==1),('Female',d2['性别']==0)]: add('Sex',g,m.values)
for g in ['Normal (<24)','Overweight (24-28)','Obese (>=28)']: add('BMI',g,(d2['BMI'].map(bmi_g)==g).values)
for g,m in [('CES-D<10',d2['抑郁']<10),('CES-D>=10',d2['抑郁']>=10)]: add('Depression',g,m.values)
for g in ['Primary or below','Junior high','Senior high or above']: add('Education',g,(d2['教育'].map(edu_g)==g).values)
subs_df=pd.DataFrame(subs); subs_df.to_csv(os.path.join(TAB,'TableS7_subgroup_AUC.csv'),index=False)

# 按基线病种数分层
strat=[]
for lab,m in [('0',d2['baseline_n_disease']==0),('1',d2['baseline_n_disease']==1),('>=2',d2['baseline_n_disease']>=2)]:
    mask=m.values; yy,pp=Yv[mask],pv[mask]
    if len(yy)<50 or yy.sum()<10 or (1-yy).sum()<10:
        strat.append(dict(baseline_conditions=lab,n=int(len(yy)),events=int(yy.sum()),
                          event_rate=np.nan,auc=np.nan)); continue
    a,lo,hi=delong_ci(yy,pp)
    strat.append(dict(baseline_conditions=lab,n=int(len(yy)),events=int(yy.sum()),
                      event_rate=round(float(yy.mean()),4),auc=round(a,4),
                      ci_low=round(lo,4),ci_high=round(hi,4)))
pd.DataFrame(strat).to_csv(os.path.join(TAB,'TableS8_performance_by_baseline_burden.csv'),index=False)

# 基线特征表
def desc(d, name):
    rows=[]
    rows.append(dict(sample=name, variable='N', value=len(d)))
    rows.append(dict(sample=name, variable='Age, mean (SD)',
                     value='%.1f (%.1f)'%(d['年龄'].mean(), d['年龄'].std())))
    rows.append(dict(sample=name, variable='Male, n (%)', value='%d (%.1f)'%((d['性别']==1).sum(),100*(d['性别']==1).mean())))
    rows.append(dict(sample=name, variable='BMI, mean (SD)', value='%.2f (%.2f)'%(d['BMI'].mean(), d['BMI'].std())))
    rows.append(dict(sample=name, variable='Systolic BP, mean (SD)', value='%.1f (%.1f)'%(d['收缩压'].mean(), d['收缩压'].std())))
    rows.append(dict(sample=name, variable='Glucose, mean (SD)', value='%.1f (%.1f)'%(d['血糖'].mean(), d['血糖'].std())))
    rows.append(dict(sample=name, variable='CES-D10, mean (SD)', value='%.1f (%.1f)'%(d['抑郁'].mean(), d['抑郁'].std())))
    rows.append(dict(sample=name, variable='>=1 baseline condition, n (%)',
                     value='%d (%.1f)'%((d['baseline_n_disease']>=1).sum(),100*(d['baseline_n_disease']>=1).mean())))
    rows.append(dict(sample=name, variable='>=2 baseline conditions, n (%)',
                     value='%d (%.1f)'%((d['baseline_n_disease']>=2).sum(),100*(d['baseline_n_disease']>=2).mean())))
    rows.append(dict(sample=name, variable='Outcome: incident condition, n (%)',
                     value='%d (%.1f)'%(d['y'].sum(),100*d['y'].mean())))
    return rows
pd.DataFrame(desc(d1,'derivation_2011_2013')+desc(d2,'validation_2015_2018')).to_csv(
    os.path.join(TAB,'Table1_baseline_characteristics.csv'), index=False)

# ---------------- SHAP ----------------
import shap
expl = shap.TreeExplainer(MODELS[MAIN])
sv = expl.shap_values(Xv)
if isinstance(sv, list): sv = sv[1]
imp = np.abs(sv).mean(0)
shap_imp = pd.DataFrame({'feature': FN, 'mean_abs_shap': np.round(imp,5)}).sort_values(
    'mean_abs_shap', ascending=False).reset_index(drop=True)
shap_imp.to_csv(os.path.join(TAB,'TableS10_SHAP_importance.csv'), index=False)
np.save(os.path.join(INTER,'v3','shap_values_val.npy'), sv)
top3 = shap_imp.head(3)['feature'].tolist()
print("\nSHAP top5:\n", shap_imp.head(5).to_string(index=False))

# ---------------- 出图（无标题） ----------------
def save3(fig, name):
    for ext, kw in [('png', dict(dpi=300)), ('svg', {}), ('eps', {})]:
        fig.savefig(os.path.join(FIG, '%s.%s' % (name, ext)), bbox_inches='tight', **kw)
    plt.close(fig)

# Fig7 校准
fig, axes = plt.subplots(1,2, figsize=(7.2,3.1))
for ax,(yy,pp,tag) in zip(axes, [(Yte,pt,'Internal test'), (Yv,pv,'Temporal validation')]):
    frac_pos, mean_pred = [], []
    bins = np.linspace(0,1,11)
    for i in range(10):
        m = (pp>=bins[i]) & (pp<bins[i+1]) if i<9 else (pp>=bins[i])
        if m.sum()>0:
            frac_pos.append(yy[m].mean()); mean_pred.append(pp[m].mean())
    ax.plot([0,1],[0,1],'--',color='#999999',lw=1)
    ax.plot(mean_pred, frac_pos, 'o-', color='#1f4e79', lw=1.6, ms=5)
    ax.set_xlim(0,1); ax.set_ylim(0,1)
    ax.set_xlabel('Predicted probability'); ax.set_ylabel('Observed proportion')
    ax.text(0.05,0.92,tag, transform=ax.transAxes, fontsize=9, color='#333333')
    ax.text(0.05,0.82,'Brier = %.3f'%np.mean((pp-yy)**2), transform=ax.transAxes, fontsize=8, color='#555555')
    ax.grid(alpha=.2)
save3(fig,'Figure7_Calibration')

# Fig8 SHAP 重要性
ordr = np.argsort(imp)[::-1][:15]
fig, ax = plt.subplots(figsize=(5.6,4.4))
ax.barh(range(len(ordr))[::-1], imp[ordr], color='#2e75b6')
ax.set_yticks(range(len(ordr))[::-1]); ax.set_yticklabels([FN[i] for i in ordr], fontsize=8)
ax.set_xlabel('Mean |SHAP value|'); ax.grid(axis='x', alpha=.25)
save3(fig,'Figure8_SHAP_importance')

# Fig9 SHAP 依赖
idxs = [FEATURES.index(EN_REV) for EN_REV in [ [k for k,vv in EN.items() if vv==t][0] for t in top3 ]]
fig, axes = plt.subplots(1,3, figsize=(10.5,3.0))
for ax,j in zip(axes, idxs):
    ax.scatter(Xv[:,j], sv[:,j], c=Yv, cmap='coolwarm', s=5, alpha=.45, linewidths=0)
    ax.axhline(0, color='#999999', lw=.8)
    ax.set_xlabel(FN[j], fontsize=8); ax.set_ylabel('SHAP value', fontsize=8)
    ax.grid(alpha=.2)
save3(fig,'Figure9_SHAP_dependence')

# Fig10 决策曲线（含靶向 vs 随机，预算受限视角）
grid = np.linspace(0.02, 0.95, 200)
fig, ax = plt.subplots(figsize=(5.8,4.0))
ax.plot(grid, [nbv(Yv,pv,t) for t in grid], color='#c00000', lw=1.8, label='Risk-based targeting')
ax.plot(grid, [nbv(Yv,np.ones_like(pv),t) for t in grid], color='#7f7f7f', lw=1.4, ls='--', label='Screen all')
ax.axhline(0, color='#000000', lw=1.0, label='Screen none')
ax.set_xlabel('Threshold probability'); ax.set_ylabel('Net benefit')
ax.set_xlim(0,0.95); ax.legend(frameon=False, fontsize=8); ax.grid(alpha=.2)
save3(fig,'Figure10_DecisionCurve')

# FigS1 ROC
fig, ax = plt.subplots(figsize=(4.8,4.4))
for nm, col in [('LightGBM','#c00000'), ('XGBoost','#2e75b6'), ('RandomForest','#548235'),
                ('Logistic','#7030a0'), ('SVM','#ed7d31')]:
    p = np.load(os.path.join(INTER,'v3','pred_val_%s.npy'%nm))
    fpr,tpr,_ = roc_curve(Yv,p)
    ax.plot(fpr,tpr,color=col,lw=1.5,label='%s (AUC %.3f)'%(nm,res[nm]['val']['auc']))
ax.plot([0,1],[0,1],'--',color='#999999',lw=1)
ax.set_xlabel('1 - Specificity'); ax.set_ylabel('Sensitivity')
ax.legend(frameon=False, fontsize=7, loc='lower right'); ax.grid(alpha=.2)
save3(fig,'FigureS1_ROC')

# FigS2 预算-捕获曲线
fig, ax = plt.subplots(figsize=(5.0,4.0))
ks = bud['budget_fraction'].values
ax.plot(ks, bud['capture_validation'].values, 'o-', color='#c00000', lw=1.6, ms=4, label='Risk-based targeting')
ax.plot(ks, ks, '--', color='#7f7f7f', lw=1.3, label='Random allocation')
ax.set_xlabel('Screening budget (fraction screened)'); ax.set_ylabel('Proportion of incident cases captured')
ax.set_xlim(0,1); ax.set_ylim(0,1); ax.legend(frameon=False, fontsize=8); ax.grid(alpha=.2)
save3(fig,'FigureS2_capture_vs_budget')

# ---------------- 汇总 ----------------
out = dict(
 version='v3',
 outcome_definition='Incident chronic condition: >=1 new condition reported at the follow-up wave relative to baseline wave',
 predictors=FN, n_predictors=len(FN),
 n_train=int(len(Ytr)), n_test=int(len(Yte)), n_val=int(len(Yv)),
 n_events_train=int(Ytr.sum()), n_events_test=int(Yte.sum()), n_events_val=int(Yv.sum()),
 event_rate_train=round(float(Ytr.mean()),4), event_rate_test=round(float(Yte.mean()),4),
 event_rate_val=round(float(Yv.mean()),4),
 epv=round(float(Ytr.sum())/len(FN),1),
 models=res, main_model=MAIN,
 lightgbm=res[MAIN],
 capture_top20_val=round(cap(Yv,pv,0.20),4), capture_top20_test=round(cap(Yte,pt,0.20),4),
 capture_top10_val=round(cap(Yv,pv,0.10),4),
 resource_saving_vs_random=round(float(1-0.20/cap(Yv,pv,0.20)),4),
 nb_peak=nbmax['nb_model'], nb_peak_thr=nbmax['threshold'],
 nb_vs_screen_all_at_peak=round(nbmax['nb_model']-nbmax['nb_screen_all'],4),
 shap_top5=shap_imp.head(5).to_dict('records'),
 incremental_value=inc_df.to_dict('records'),
 budget_curve=budget,
 followup_interval=dict(derivation_years=2, validation_years=3),
)
json.dump(out, open(os.path.join(DATA,'metrics_v3.json'),'w',encoding='utf-8'),
          ensure_ascii=False, indent=2, default=lambda o: float(o) if hasattr(o,'item') or isinstance(o,(np.floating,)) else str(o))
pd.DataFrame({'y_true':np.concatenate([Yte,Yv]),
              'p_pred':np.concatenate([pt,pv]),
              'split':['test']*len(Yte)+['val']*len(Yv)}).to_csv(
    os.path.join(INTER,'v3','predictions_v3.csv'), index=False)
print("\n=== V3 SUMMARY ===")
print(json.dumps({k:out[k] for k in ['n_train','n_test','n_val','event_rate_test','event_rate_val',
      'epv','capture_top20_val','capture_top20_test','resource_saving_vs_random',
      'nb_peak','nb_peak_thr','nb_vs_screen_all_at_peak']}, indent=2))
print("FIGURES+TABLES SAVED")
