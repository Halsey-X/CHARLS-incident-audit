"""
v4 创新性分析 —— 在 v3 建模基础上新增三项方法学贡献
 (A) Zero-cost benchmark audit (ZBA): 零成本基准审计, Delta-AUC + 配对 bootstrap 95%CI, 跨结局定义重复
 (B) Capacity-constrained ranking comparison (CCRC): 固定容量下多排序规则公平比较 + 同预算净获益 + 等效容量扩张率
 (C) Transportability decomposition & recalibration: 排序 vs 概率尺度分离, CV out-of-fold Platt 校准器迁移实证
 (D) Risk-stratification allocation matrix: 五分位分层病例密度与累积捕获
输出：无图内标题的 PNG/SVG/EPS 图件 + 全部表格 + metrics_v4.json
"""
import os, json, warnings
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, brier_score_loss, roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.isotonic import IsotonicRegression
import lightgbm as lgb
import xgboost as xgb

warnings.filterwarnings("ignore")
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                     "axes.edgecolor": "#333333", "axes.linewidth": 0.8,
                     "svg.fonttype": "none", "ps.fonttype": 42})

BASE = r'D:\02XIHO Files\Daily Work\03Scientific Research\00Article_File\00Student\01单任务论文\PublicHealth_Submission'
INTER = os.path.join(BASE, '04_Data_and_Results', 'Intermediate')
DATA = os.path.join(BASE, '04_Data_and_Results')
TAB = os.path.join(BASE, '03_Tables')
FIG = os.path.join(BASE, '02_Figures', '02-2_ResultFigures')
V4 = os.path.join(INTER, 'v4')
for p in (TAB, FIG, V4):
    os.makedirs(p, exist_ok=True)

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

RNG = np.random.default_rng(42)

# ============================ 数据 ============================
d1 = pd.read_csv(os.path.join(INTER, 'dataset_derivation_2011_2013.csv'))
d2 = pd.read_csv(os.path.join(INTER, 'dataset_validation_2015_2018.csv'))

# 结局 A（主）: 随访波新增 >=1 种慢病
for d in (d1, d2):
    d['yA'] = ((d['followup_n_disease'] - d['baseline_n_disease']) >= 1).astype(int)
    # 结局 B（对照）: 随访波共病状态 >=2 种
    d['yB'] = (d['followup_n_disease'] >= 2).astype(int)

for c in FEATURES:
    med = pd.concat([d1[c], d2[c]]).astype(float).median()
    d1[c] = d1[c].astype(float).fillna(med); d2[c] = d2[c].astype(float).fillna(med)

X = d1[FEATURES].astype(float).values; YA = d1['yA'].values; YB = d1['yB'].values
Xv = d2[FEATURES].astype(float).values; YvA = d2['yA'].values; YvB = d2['yB'].values
Xtr, Xte, YtrA, YteA, YtrB, YteB, itr, ite = train_test_split(
    X, YA, YB, np.arange(len(YA)), test_size=0.2, random_state=42, stratify=YA)

print("n_train=%d (ev %d) n_test=%d (ev %d) n_val=%d (ev %d)" %
      (len(YtrA), YtrA.sum(), len(YteA), YteA.sum(), len(YvA), YvA.sum()))

def make_models():
    return {
     "LightGBM": lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31,
                                    max_depth=6, reg_lambda=0.1, random_state=42, verbose=-1),
     "XGBoost": xgb.XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=6,
                                  reg_lambda=0.1, random_state=42, eval_metric='logloss', verbosity=0),
     "RandomForest": RandomForestClassifier(n_estimators=400, random_state=42, n_jobs=-1),
     "Logistic": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=42)),
     "SVM": make_pipeline(StandardScaler(), SVC(probability=True, random_state=42)),
    }

# ---- out-of-fold 预测（训练集内 5 折），用于拟合校准器 ----
def oof_predictions(model, X, y, n_splits=5, seed=42):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof = np.zeros(len(y))
    import copy
    for tr, te in skf.split(X, y):
        m = copy.deepcopy(model)
        m.fit(X[tr], y[tr])
        oof[te] = m.predict_proba(X[te])[:, 1]
    return oof

def logit(p, eps=1e-6):
    p = np.clip(np.asarray(p, float), eps, 1 - eps)
    return np.log(p / (1 - p))

def fit_platt(p_oof, y_oof):
    """Logistic 再校准: logit(p_cal) = a + b * logit(p_raw)"""
    z = logit(p_oof).reshape(-1, 1)
    lr = LogisticRegression(C=1e6, solver='lbfgs', max_iter=5000)
    lr.fit(z, y_oof)
    a, b = float(lr.intercept_[0]), float(lr.coef_[0][0])
    return a, b

def apply_platt(p, a, b):
    z = a + b * logit(p)
    return 1 / (1 + np.exp(-z))

def calib_stats(y, p, n_bins=10):
    p = np.asarray(p, float); y = np.asarray(y, int)
    try:
        lo = logit(p)
        lr = LogisticRegression(C=1e6, solver='lbfgs', max_iter=5000)
        lr.fit(lo.reshape(-1, 1), y)
        slope, intercept = float(lr.coef_[0][0]), float(lr.intercept_[0])
    except Exception:
        slope, intercept = np.nan, np.nan
    # 分位分箱校准
    q = np.quantile(p, np.linspace(0, 1, n_bins + 1)); q[0] -= 1e-9; q[-1] += 1e-9
    obs, pred = [], []
    for i in range(n_bins):
        m = (p > q[i]) & (p <= q[i + 1])
        if m.sum() == 0: continue
        obs.append(y[m].mean()); pred.append(p[m].mean())
    obs, pred = np.array(obs), np.array(pred)
    return dict(brier=float(brier_score_loss(y, p)), slope=slope, intercept=intercept,
                obs=obs, pred=pred)

def delong_ci(y, p, alpha=0.05):
    y = np.asarray(y, int); p = np.asarray(p, float)
    order = np.argsort(p); ps, ys = p[order], y[order]
    n1 = int(ys.sum()); n0 = len(ys) - n1
    if n1 == 0 or n0 == 0: return np.nan, np.nan, np.nan
    ranks = np.empty(len(ps)); i = 0
    while i < len(ps):
        j = i
        while j + 1 < len(ps) and ps[j + 1] == ps[i]: j += 1
        ranks[i:j + 1] = (i + j + 2) / 2.0; i = j + 1
    r1 = ranks[ys == 1]; r0 = ranks[ys == 0]
    V10 = np.array([(r1[k] - np.searchsorted(r0, r1[k], side='left')) / n0 for k in range(n1)])
    V01 = np.array([1 - np.searchsorted(r1, r, side='right') / n1 for r in r0])
    auc = roc_auc_score(y, p)
    se = np.sqrt(V10.var(ddof=1) / n1 + V01.var(ddof=1) / n0)
    z = 1.959963985
    return float(auc), float(max(0, auc - z * se)), float(min(1, auc + z * se))

def boot_delta_auc(y, p_model, p_base, n_boot=1000, seed=42):
    """配对 bootstrap: Delta AUC 的 95%CI"""
    rng = np.random.default_rng(seed)
    y = np.asarray(y, int); p1 = np.asarray(p_model, float); p0 = np.asarray(p_base, float)
    n = len(y); deltas = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yy = y[idx]
        if yy.sum() == 0 or yy.sum() == len(yy): continue
        try:
            d = roc_auc_score(yy, p1[idx]) - roc_auc_score(yy, p0[idx])
            deltas.append(d)
        except Exception:
            continue
    deltas = np.array(deltas)
    return float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))

# ============================ 建模 ============================
MODELS = make_models()
pred_test, pred_val = {}, {}
for name, m in MODELS.items():
    m.fit(Xtr, YtrA)
    pred_test[name] = m.predict_proba(Xte)[:, 1]
    pred_val[name] = m.predict_proba(Xv)[:, 1]
MAIN = "LightGBM"
p_te, p_va = pred_test[MAIN], pred_val[MAIN]

# 保存预测
np.save(os.path.join(V4, 'pred_test_LightGBM.npy'), p_te)
np.save(os.path.join(V4, 'pred_val_LightGBM.npy'), p_va)
pd.DataFrame({'ID': d1.iloc[ite]['ID'].values, 'y': YteA, 'p': p_te}).to_csv(
    os.path.join(V4, 'predictions_test.csv'), index=False, encoding='utf-8-sig')
pd.DataFrame({'ID': d2['ID'].values, 'y': YvA, 'p': p_va}).to_csv(
    os.path.join(V4, 'predictions_val.csv'), index=False, encoding='utf-8-sig')

# ==================== (A) 零成本基准审计 ZBA ====================
print("\n=== A. Zero-cost benchmark audit ===")
base_defs = {
    'Baseline condition count': ('baseline_n_disease', 1),
    'Age alone': ('年龄', 1),
    'BMI alone': ('BMI', 1),
    'Systolic BP alone': ('收缩压', 1),
    'Depression score': ('抑郁', 1),
}
zba = {}
for sname, (y, pm) in [('internal_test', (YteA, p_te)), ('temporal_validation', (YvA, p_va))]:
    rows = []
    for bname, (col, sign) in base_defs.items():
        src = d1 if sname == 'internal_test' else d2
        idx = ite if sname == 'internal_test' else np.arange(len(d2))
        p_base = sign * src[col].values[idx].astype(float)
        auc_base = roc_auc_score(y, p_base)
        auc_model = roc_auc_score(y, pm)
        lo, hi = boot_delta_auc(y, pm, p_base, n_boot=1000)
        rows.append(dict(sample=sname, benchmark=bname, auc_model=round(auc_model, 4),
                         auc_benchmark=round(auc_base, 4),
                         delta_auc=round(auc_model - auc_base, 4),
                         delta_lo=round(lo, 4), delta_hi=round(hi, 4),
                         significant='yes' if lo > 0 else 'no'))
        print("  %-20s %-22s model %.4f  bench %.4f  delta %+.4f [%+.4f,%+.4f]" %
              (sname, bname, auc_model, auc_base, auc_model - auc_base, lo, hi))
    zba[sname] = rows
pd.DataFrame(zba['internal_test'] + zba['temporal_validation']).to_csv(
    os.path.join(TAB, 'Table2_zero_cost_benchmark_audit.csv'), index=False, encoding='utf-8-sig')

# 跨结局定义重复：结局 B（共病状态）
print("\n--- 结局 B (随访共病状态) 复核 ---")
mb = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31,
                        max_depth=6, reg_lambda=0.1, random_state=42, verbose=-1)
mb.fit(Xtr, YtrB)
p_teB, p_vaB = mb.predict_proba(Xte)[:, 1], mb.predict_proba(Xv)[:, 1]
outcome_B = {}
for sname, (y, pm) in [('internal_test', (YteB, p_teB)), ('temporal_validation', (YvB, p_vaB))]:
    src = d1 if sname == 'internal_test' else d2
    idx = ite if sname == 'internal_test' else np.arange(len(d2))
    p_base = src['baseline_n_disease'].values[idx].astype(float)
    auc_m = roc_auc_score(y, pm); auc_b = roc_auc_score(y, p_base)
    lo, hi = boot_delta_auc(y, pm, p_base, n_boot=1000)
    outcome_B[sname] = dict(auc_model=round(auc_m, 4), auc_benchmark=round(auc_b, 4),
                            delta=round(auc_m - auc_b, 4), lo=round(lo, 4), hi=round(hi, 4))
    print("  %-20s model %.4f vs condition count %.4f  delta %+.4f [%+.4f,%+.4f]" %
          (sname, auc_m, auc_b, auc_m - auc_b, lo, hi))

# ==================== (B) 容量约束排序规则比较 CCRC ====================
print("\n=== B. Capacity-constrained ranking comparison ===")
def rank_scores(sname):
    src = d1 if sname == 'internal_test' else d2
    idx = ite if sname == 'internal_test' else np.arange(len(d2))
    y = YteA if sname == 'internal_test' else YvA
    p = p_te if sname == 'internal_test' else p_va
    return {
        'Model risk score': p,
        'Baseline condition count': src['baseline_n_disease'].values[idx].astype(float),
        'Age alone': src['年龄'].values[idx].astype(float),
        'BMI alone': src['BMI'].values[idx].astype(float),
        'Depression score': src['抑郁'].values[idx].astype(float),
    }, y

KS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
WS = [0.111, 0.250, 0.429, 1.000]   # 交换率 w = p_t/(1-p_t)，对应阈值 0.10/0.20/0.30/0.50
ccrc_rows = []
for sname in ['internal_test', 'temporal_validation']:
    scores, y = rank_scores(sname)
    y = np.asarray(y, int); N = len(y); total = y.sum()
    for rname, s in scores.items():
        s = np.asarray(s, float)
        # 随机作为参照
        order = np.argsort(-s, kind='mergesort')
        for K in KS:
            k = int(round(K * N))
            sel = order[:k]
            cap = y[sel].sum() / total
            # 同预算净获益 NB_cc(K,w) = TP/N - (FP/N)*w，w 由决策者外生给定
            # （对非概率型排序规则不使用阈值派生 w，避免尺度不可比）
            TP = int(y[sel].sum()); FP = int(k - TP)
            nb_cc = {round(w, 3): TP / N - (FP / N) * w for w in WS}
            ccrc_rows.append(dict(sample=sname, rule=rname, budget=K, n_screened=k,
                                  cases_captured=TP, capture_rate=round(cap, 4),
                                  capture_vs_random=round(cap - K, 4),
                                  equivalent_random_budget=round(cap, 4),
                                  capacity_saving=round((cap - K) / cap, 4) if cap > 0 else np.nan,
                                  **{'nb_cc_w%.3f' % k2: round(v, 4) for k2, v in nb_cc.items()}))
    # 随机策略（期望）
    rng = np.random.default_rng(7)
    for K in KS:
        k = int(round(K * N))
        reps = []
        for _ in range(200):
            sel = rng.choice(N, k, replace=False)
            reps.append(y[sel].sum() / total)
        cap = float(np.mean(reps))
        TPf = cap * total; FPf = k - TPf
        nb_cc = {round(w, 3): TPf / N - (FPf / N) * w for w in WS}
        ccrc_rows.append(dict(sample=sname, rule='Random allocation', budget=K, n_screened=k,
                              cases_captured=int(round(TPf)), capture_rate=round(cap, 4),
                              capture_vs_random=0.0, equivalent_random_budget=round(cap, 4),
                              capacity_saving=0.0,
                              **{'nb_cc_w%.3f' % k2: round(v, 4) for k2, v in nb_cc.items()}))
ccrc = pd.DataFrame(ccrc_rows)
ccrc.to_csv(os.path.join(TAB, 'Table3_capacity_constrained_ranking.csv'),
            index=False, encoding='utf-8-sig')
print(ccrc[ccrc.budget.isin([0.10, 0.20, 0.30])].to_string(index=False))

# 提取关键数字
c20 = ccrc[(ccrc.budget == 0.20) & (ccrc['sample'] == 'temporal_validation')]
c20d = dict(zip(c20['rule'], c20['capture_rate']))
print("\nTop-20% 捕获率（验证集）:", {k: round(v, 4) for k, v in c20d.items()})

# ==================== (C) 迁移性分解 + 再校准 ====================
print("\n=== C. Transportability & recalibration ===")
oof = oof_predictions(make_models()[MAIN], Xtr, YtrA, n_splits=5)
a, b = fit_platt(oof, YtrA)
print("Platt calibrator fitted on out-of-fold training predictions: a=%.4f b=%.4f" % (a, b))

recal = {}
for sname, (y, p) in [('internal_test', (YteA, p_te)), ('temporal_validation', (YvA, p_va))]:
    before = calib_stats(y, p)
    after = calib_stats(y, apply_platt(p, a, b))
    recal[sname] = dict(
        auc_before=round(roc_auc_score(y, p), 4),
        auc_after=round(roc_auc_score(y, apply_platt(p, a, b)), 4),
        brier_before=round(before['brier'], 4), brier_after=round(after['brier'], 4),
        slope_before=round(before['slope'], 4), slope_after=round(after['slope'], 4),
        intercept_before=round(before['intercept'], 4), intercept_after=round(after['intercept'], 4),
    )
    print("  %-20s AUC %.4f->%.4f | Brier %.4f->%.4f | slope %.3f->%.3f | intercept %.3f->%.3f" %
          (sname, recal[sname]['auc_before'], recal[sname]['auc_after'],
           recal[sname]['brier_before'], recal[sname]['brier_after'],
           recal[sname]['slope_before'], recal[sname]['slope_after'],
           recal[sname]['intercept_before'], recal[sname]['intercept_after']))
    np.save(os.path.join(V4, 'cal_curve_%s_before.npy' % sname),
            np.vstack([before['pred'], before['obs']]))
    np.save(os.path.join(V4, 'cal_curve_%s_after.npy' % sname),
            np.vstack([after['pred'], after['obs']]))
pd.DataFrame(recal).T.reset_index().rename(columns={'index': 'sample'}).to_csv(
    os.path.join(TAB, 'Table4_recalibration.csv'), index=False, encoding='utf-8-sig')

# ==================== (D) 分层配置矩阵 ====================
print("\n=== D. Stratified allocation matrix ===")
strat_rows = []
for sname, (y, p) in [('internal_test', (YteA, p_te)), ('temporal_validation', (YvA, p_va))]:
    y = np.asarray(y, int); p = np.asarray(p, float)
    q = np.quantile(p, [0, .2, .4, .6, .8, 1.0])
    cum = 0; total = y.sum()
    # 从最高风险五分位向最低风险累计
    labels = ['Q5 (highest risk)', 'Q4 (high risk)', 'Q3 (moderate risk)',
              'Q2 (low risk)', 'Q1 (lowest risk)']
    for i in range(5):
        lo_, hi_ = q[4 - i], q[5 - i]          # i=0 -> 最高分位
        m = (p >= lo_) & (p <= hi_) if i == 0 else (p >= lo_) & (p < hi_)
        n = int(m.sum()); cases = int(y[m].sum())
        cum += cases
        strat_rows.append(dict(sample=sname, stratum=labels[i], n=n, cases=cases,
                               case_density_per_100=round(100 * cases / n, 2) if n else np.nan,
                               cumulative_capture=round(cum / total, 4)))
    print("  --", sname)
strat = pd.DataFrame(strat_rows)
strat.to_csv(os.path.join(TAB, 'Table5_stratified_allocation.csv'), index=False, encoding='utf-8-sig')
print(strat[strat['sample'] == 'temporal_validation'].to_string(index=False))

# ==================== 五模型对比（主结局） ====================
print("\n=== Model comparison (outcome A) ===")
mc_rows = []
for name in MODELS:
    for sname, (y, p) in [('internal_test', (YteA, pred_test[name])),
                          ('temporal_validation', (YvA, pred_val[name]))]:
        auc, lo_, hi_ = delong_ci(y, p)
        fpr, tpr, _ = roc_curve(y, p)
        youden = tpr - fpr; j = int(np.argmax(youden)); thr = float(np.sort(p)[::-1][j])
        yhat = (p >= thr).astype(int)
        sens = float(((yhat == 1) & (y == 1)).sum() / max(1, (y == 1).sum()))
        spec = float(((yhat == 0) & (y == 0)).sum() / max(1, (y == 0).sum()))
        f1 = float(2 * sens * (y.sum() / len(y)) / (sens + y.sum() / len(y))) if sens > 0 else 0.0
        cs = calib_stats(y, p)
        mc_rows.append(dict(model=name, sample=sname, auc=round(auc, 4),
                            auc_lo=round(lo_, 4), auc_hi=round(hi_, 4),
                            f1=round(f1, 4), sensitivity=round(sens, 4),
                            specificity=round(spec, 4), brier=round(cs['brier'], 4),
                            calib_slope=round(cs['slope'], 4), calib_intercept=round(cs['intercept'], 4)))
mc = pd.DataFrame(mc_rows)
mc.to_csv(os.path.join(TAB, 'Table6_model_comparison.csv'), index=False, encoding='utf-8-sig')
print(mc[mc['sample'] == 'temporal_validation'].to_string(index=False))

# ==================== SHAP ====================
print("\n=== SHAP ===")
try:
    import shap
    expl = shap.TreeExplainer(MODELS[MAIN])
    sv = expl.shap_values(Xv)
    if isinstance(sv, list): sv = sv[1]
    imp = np.abs(sv).mean(0)
    top = np.argsort(-imp)[:10]
    shap_df = pd.DataFrame({'rank': range(1, len(top) + 1),
                            'feature': [FN[j] for j in top],
                            'mean_abs_shap': [round(float(imp[j]), 4) for j in top]})
    shap_df.to_csv(os.path.join(TAB, 'Table7_SHAP_importance.csv'), index=False, encoding='utf-8-sig')
    print(shap_df.to_string(index=False))
    np.save(os.path.join(V4, 'shap_values_val.npy'), sv)
    top3 = top[:3]
except Exception as e:
    print("SHAP failed:", e); shap_df = None; top3 = []

# ==================== (E) 基线病种负担分层（天花板效应检验） ====================
print("\n=== E. Performance by baseline condition burden (ceilings) ===")
burden_rows = []
for sname, (y, p) in [('internal_test', (YteA, p_te)), ('temporal_validation', (YvA, p_va))]:
    src = d1 if sname == 'internal_test' else d2
    idx = ite if sname == 'internal_test' else np.arange(len(d2))
    nb_ = src['baseline_n_disease'].values[idx].astype(int)
    for grp, mask in [('0 conditions', nb_ == 0), ('1 condition', nb_ == 1),
                      ('>=2 conditions', nb_ >= 2)]:
        yy = np.asarray(y, int)[mask]; pp = np.asarray(p, float)[mask]
        if yy.sum() < 5 or len(yy) - yy.sum() < 5: continue
        auc, lo_, hi_ = delong_ci(yy, pp)
        burden_rows.append(dict(sample=sname, baseline_burden=grp, n=int(mask.sum()),
                                events=int(yy.sum()), event_rate=round(float(yy.mean()), 4),
                                auc=round(auc, 4), auc_lo=round(lo_, 4), auc_hi=round(hi_, 4)))
        print("  %-20s %-16s n=%4d events=%4d rate=%.3f AUC=%.4f [%.4f,%.4f]" %
              (sname, grp, mask.sum(), yy.sum(), yy.mean(), auc, lo_, hi_))
    # 病种计数在该样本中作为排序规则的表现
    auc_cnt = roc_auc_score(np.asarray(y, int), nb_.astype(float))
    print("  %-20s baseline condition count as sole ranker: AUC=%.4f" % (sname, auc_cnt))
pd.DataFrame(burden_rows).to_csv(os.path.join(TAB, 'Table8_performance_by_baseline_burden.csv'),
                                 index=False, encoding='utf-8-sig')

# ==================== 汇总 JSON ====================
c20v = ccrc[(ccrc.budget == 0.20) & (ccrc['sample'] == 'temporal_validation')]
c20t = ccrc[(ccrc.budget == 0.20) & (ccrc['sample'] == 'internal_test')]
sv_strat = strat[strat['sample'] == 'temporal_validation']
out = dict(
    version='v4', outcome='incident chronic condition (>=1 new at follow-up wave)',
    n_train=int(len(YtrA)), n_test=int(len(YteA)), n_val=int(len(YvA)),
    events_train=int(YtrA.sum()), events_test=int(YteA.sum()), events_val=int(YvA.sum()),
    event_rate_test=round(float(YteA.mean()), 4), event_rate_val=round(float(YvA.mean()), 4),
    n_predictors=len(FEATURES),
    lightgbm=dict(
        auc_test=float(roc_auc_score(YteA, p_te)), auc_val=float(roc_auc_score(YvA, p_va)),
        **{('auc_%s_%s' % (s, k)): float(v) for s, (y, p) in
           {'test': (YteA, p_te), 'val': (YvA, p_va)}.items()
           for k, v in zip(['ci_low', 'ci_high'], delong_ci(y, p)[1:])}),
    zba=zba, zba_outcome_B=outcome_B,
    recalibration=recal, platt=dict(a=round(a, 4), b=round(b, 4)),
    capacity_curve={r['rule']: {str(r['budget']): r['capture_rate'] for _, r in
                                ccrc[(ccrc['sample'] == 'temporal_validation') &
                                     (ccrc.rule == r['rule'])].iterrows()}
                    for r in ccrc[ccrc['sample'] == 'temporal_validation'].drop_duplicates('rule').to_dict('records')},
    capture_top20_val=float(c20v[c20v.rule == 'Model risk score']['capture_rate'].iloc[0]),
    capture_top20_test=float(c20t[c20t.rule == 'Model risk score']['capture_rate'].iloc[0]),
    capture_top20_random_val=float(c20v[c20v.rule == 'Random allocation']['capture_rate'].iloc[0]),
    capture_top20_count_val=float(c20v[c20v.rule == 'Baseline condition count']['capture_rate'].iloc[0]),
    capacity_saving_top20_val=float(c20v[c20v.rule == 'Model risk score']['capacity_saving'].iloc[0]),
    stratified_val=sv_strat.to_dict('records'),
    shap_top5=shap_df.head(5).to_dict('records') if shap_df is not None else [],
    burden_subgroups=burden_rows,
)
def _j(o):
    import numpy as np
    if isinstance(o, dict): return {k: _j(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [_j(v) for v in o]
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, (np.floating,)): return float(o)
    if isinstance(o, (np.bool_,)): return bool(o)
    return o
json.dump(_j(out), open(os.path.join(DATA, 'metrics_v4.json'), 'w', encoding='utf-8'),
          ensure_ascii=False, indent=1)
print("\nmetrics_v4.json written")
print("ANALYSIS_V4_DONE")
