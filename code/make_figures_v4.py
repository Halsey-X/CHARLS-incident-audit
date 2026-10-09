"""
v4 图件 —— 全部无图内标题；输出 PNG / SVG / EPS
核心新图：Figure11 零成本基准审计森林图、Figure12 容量约束排序比较、Figure13 再校准前后
"""
import os, json, warnings
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

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
os.makedirs(FIG, exist_ok=True)

M = json.load(open(os.path.join(DATA, 'metrics_v4.json'), encoding='utf-8'))
p_te = np.load(os.path.join(V4, 'pred_test_LightGBM.npy'))
p_va = np.load(os.path.join(V4, 'pred_val_LightGBM.npy'))
te = pd.read_csv(os.path.join(V4, 'predictions_test.csv'))
va = pd.read_csv(os.path.join(V4, 'predictions_val.csv'))
y_te, y_va = te['y'].values.astype(int), va['y'].values.astype(int)
sv = np.load(os.path.join(V4, 'shap_values_val.npy'))
if sv.ndim == 3: sv = sv[:, :, 1]

d2 = pd.read_csv(os.path.join(INTER, 'dataset_validation_2015_2018.csv'))
d1 = pd.read_csv(os.path.join(INTER, 'dataset_derivation_2011_2013.csv'))

DISEASES = ['高血压','糖尿病','癌症','肺部疾病','心脏病','中风','精神心理疾病','关节炎',
            '血脂异常','肝脏疾病','肾脏疾病','消化系统疾病','哮喘','记忆力障碍']
BASE_F = ['年龄','性别','婚姻','教育','退休','吸烟','饮酒','锻炼','睡眠','BMI',
          '收缩压','舒张压','血糖','抑郁','自评健康']
FEATURES = BASE_F + ['慢病_' + c for c in DISEASES]
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

CMAP = {"model": "#1f4e79", "bench": "#c00000", "random": "#7f7f7f",
        "teal": "#2e8b8b", "orange": "#e08b2f", "purple": "#6a51a3"}

def save(fig, name, w=6.4, h=4.2):
    fig.set_size_inches(w, h)
    for ext in ("png", "svg", "eps"):
        fig.savefig(os.path.join(FIG, "%s.%s" % (name, ext)),
                    dpi=300, bbox_inches="tight", format=ext)
    plt.close(fig)
    print("  ->", name)

# ---------------- Figure 7 校准曲线（含再校准后） ----------------
def calcurve(y, p, n_bins=10):
    q = np.quantile(p, np.linspace(0, 1, n_bins + 1)); q[0] -= 1e-9; q[-1] += 1e-9
    obs, pred = [], []
    for i in range(n_bins):
        m = (p > q[i]) & (p <= q[i + 1])
        if m.sum() == 0: continue
        obs.append(y[m].mean()); pred.append(p[m].mean())
    return np.array(pred), np.array(obs)

a, b = M['platt']['a'], M['platt']['b']
def platt(p):
    z = a + b * np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
    return 1 / (1 + np.exp(-z))

fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.6))
for ax, (y, p, tit) in zip(axes, [(y_te, p_te, "Internal test"), (y_va, p_va, "Temporal validation")]):
    pr, ob = calcurve(y, p)
    pr2, ob2 = calcurve(y, platt(p))
    ax.plot([0, 1], [0, 1], ls="--", lw=0.9, color="#999999")
    ax.plot(pr, ob, "o-", color=CMAP["bench"], ms=4, lw=1.2, label="Before recalibration")
    ax.plot(pr2, ob2, "s-", color=CMAP["model"], ms=4, lw=1.2, label="After recalibration")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("Predicted probability"); ax.set_ylabel("Observed proportion")
    ax.set_title(tit, fontsize=9.5)
    ax.legend(frameon=False, fontsize=7.5, loc="upper left")
    ax.grid(alpha=0.18, lw=0.5)
plt.tight_layout(); save(fig, "Figure7_Calibration", 7.6, 3.6)

# ---------------- Figure 8 SHAP 重要性 ----------------
imp = np.abs(sv).mean(0)
order = np.argsort(imp)[::-1][:12]
fig, ax = plt.subplots()
ax.barh(range(len(order))[::-1], imp[order][::-1], color=CMAP["model"], height=0.68)
ax.set_yticks(range(len(order))[::-1])
ax.set_yticklabels([FN[j] for j in order][::-1], fontsize=8)
ax.set_xlabel("Mean |SHAP value| (average impact on model output)")
ax.grid(axis="x", alpha=0.18, lw=0.5)
plt.tight_layout(); save(fig, "Figure8_SHAP_importance", 6.2, 3.9)

# ---------------- Figure 9 SHAP 依赖 ----------------
top3 = np.argsort(-imp)[:3]
fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.1))
for ax, j in zip(axes, top3):
    xv = d2[FEATURES[j]].astype(float).values
    sc = ax.scatter(xv, sv[:, j], c=y_va, cmap="coolwarm", s=4, alpha=0.45,
                    vmin=-0.2, vmax=1.2)
    ax.axhline(0, color="#999999", lw=0.7, ls=":")
    ax.set_xlabel(FN[j], fontsize=8.5); ax.set_ylabel("SHAP value", fontsize=8.5)
    ax.grid(alpha=0.15, lw=0.5)
plt.tight_layout(); save(fig, "Figure9_SHAP_dependence", 9.6, 3.1)

# ---------------- Figure 10 决策曲线 ----------------
def dca(y, p, grid):
    N = len(y); out = []
    for t in grid:
        w = t / (1 - t)
        sel = p >= t
        tp = ((sel) & (y == 1)).sum(); fp = ((sel) & (y == 0)).sum()
        out.append(tp / N - fp / N * w)
    return np.array(out)

grid = np.arange(0.05, 0.96, 0.01)
prev = y_va.mean()
nb_all = prev - (1 - prev) * (grid / (1 - grid))
fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.6))
for ax, (y, p, tit) in zip(axes, [(y_te, p_te, "Internal test"), (y_va, p_va, "Temporal validation")]):
    nb = dca(y, p, grid)
    ax.plot(grid, nb, color=CMAP["model"], lw=1.5, label="Risk-based targeting")
    ax.plot(grid, np.zeros_like(grid), color="#333333", lw=1.0, label="Screen none")
    ax.plot(grid, y.mean() - (1 - y.mean()) * (grid / (1 - grid)),
            color=CMAP["bench"], lw=1.0, ls="--", label="Screen all")
    ax.set_xlim(0.05, 0.95); ax.set_ylim(-0.05, max(0.5, float(np.nanmax(nb)) + 0.06))
    ax.set_xlabel("Threshold probability"); ax.set_ylabel("Net benefit")
    ax.set_title(tit, fontsize=9.5)
    ax.legend(frameon=False, fontsize=7.5)
    ax.grid(alpha=0.18, lw=0.5)
plt.tight_layout(); save(fig, "Figure10_DecisionCurve", 7.6, 3.6)

# ---------------- Figure 11 零成本基准审计（核心新图） ----------------
zba = M['zba']; zb = M['zba_outcome_B']
fig, axes = plt.subplots(1, 2, figsize=(8.4, 4.0), gridspec_kw={"width_ratios": [1.25, 1]})
ax = axes[0]
labels, deltas, los, his = [], [], [], []
for r in zba['temporal_validation']:
    labels.append(r['benchmark'])
    deltas.append(r['delta_auc']); los.append(r['delta_lo']); his.append(r['delta_hi'])
for r in zba['internal_test']:
    labels.append(r['benchmark'] + "\n(internal test)")
    deltas.append(r['delta_auc']); los.append(r['delta_lo']); his.append(r['delta_hi'])
yy = np.arange(len(labels))[::-1]
cols = [CMAP["model"] if "internal" not in l else CMAP["teal"] for l in labels]
for i, (d, lo, hi, c) in enumerate(zip(deltas, los, his, cols)):
    ax.plot([lo, hi], [yy[i], yy[i]], color=c, lw=1.4)
    ax.plot([lo, lo], [yy[i] - 0.12, yy[i] + 0.12], color=c, lw=1.0)
    ax.plot([hi, hi], [yy[i] - 0.12, yy[i] + 0.12], color=c, lw=1.0)
    ax.plot(d, yy[i], "o", color=c, ms=5)
ax.axvline(0, color="#333333", lw=1.0, ls="--")
ax.set_yticks(yy); ax.set_yticklabels(labels, fontsize=7.2)
ax.set_xlabel("$\\Delta$AUC of the model over the zero-cost rule (95% CI)")
ax.set_xlim(-0.1, 0.48); ax.grid(axis="x", alpha=0.18, lw=0.5)
ax.text(0.02, len(labels) - 0.4, "Outcome: incident condition", fontsize=8,
        color=CMAP["model"], style="italic")

ax2 = axes[1]
names = ["Incident condition\n(temporal validation)", "Multimorbidity status\n(temporal validation)",
         "Multimorbidity status\n(internal test)"]
vals = [zb['temporal_validation']['delta'], zb['temporal_validation']['delta'] * 0,
        zb['internal_test']['delta']]
rows = [
    ("Incident condition", zb['temporal_validation']['delta'], None),
]
dA = zba['temporal_validation'][0]
rows = [("Incident condition", dA['delta_auc'], (dA['delta_lo'], dA['delta_hi'])),
        ("Multimorbidity status", zb['temporal_validation']['delta'],
         (zb['temporal_validation']['lo'], zb['temporal_validation']['hi'])),
        ("Multimorbidity status (test)", zb['internal_test']['delta'],
         (zb['internal_test']['lo'], zb['internal_test']['hi']))]
yy2 = np.arange(len(rows))[::-1]
for i, (nm, d, ci) in enumerate(rows):
    c = CMAP["model"] if d > 0 else CMAP["bench"]
    if ci: ax2.plot([ci[0], ci[1]], [yy2[i], yy2[i]], color=c, lw=1.4)
    ax2.plot(d, yy2[i], "o", color=c, ms=6)
ax2.axvline(0, color="#333333", lw=1.0, ls="--")
ax2.set_yticks(yy2); ax2.set_yticklabels([r[0] for r in rows], fontsize=7.2)
ax2.set_xlabel("$\\Delta$AUC vs baseline condition count (95% CI)")
ax2.set_xlim(-0.05, 0.35); ax2.grid(axis="x", alpha=0.18, lw=0.5)
ax2.text(-0.045, 2.35, "Outcome definition matters", fontsize=8, style="italic", color="#555")
plt.tight_layout(); save(fig, "Figure11_ZeroCostBenchmarkAudit", 8.4, 4.0)

# ---------------- Figure 12 容量约束排序比较（核心新图） ----------------
ccrc = pd.read_csv(os.path.join(TAB, 'Table3_capacity_constrained_ranking.csv'))
cv = ccrc[ccrc['sample'] == 'temporal_validation']
rules = ['Model risk score', 'Baseline condition count', 'Age alone',
         'BMI alone', 'Depression score', 'Random allocation']
styles = {'Model risk score': (CMAP["model"], "-", "o"),
          'Baseline condition count': (CMAP["bench"], "-", "s"),
          'Age alone': (CMAP["orange"], "--", "^"),
          'BMI alone': (CMAP["purple"], "--", "v"),
          'Depression score': (CMAP["teal"], "--", "d"),
          'Random allocation': (CMAP["random"], ":", "x")}
fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.9))
ax = axes[0]
for r in rules:
    sub = cv[cv.rule == r].sort_values('budget')
    c, ls, mk = styles[r]
    ax.plot(sub['budget'] * 100, sub['capture_rate'] * 100, ls, color=c, marker=mk,
            ms=3.6, lw=1.3, label=r)
ax.plot([0, 50], [0, 50], color="#999999", lw=0.9, ls="--")
ax.set_xlabel("Screening capacity (% of eligible population screened)")
ax.set_ylabel("Incident cases captured (%)")
ax.set_xlim(0, 52); ax.set_ylim(0, 55)
ax.legend(frameon=False, fontsize=7, loc="upper left")
ax.grid(alpha=0.18, lw=0.5)
ax.text(26, 7, "dashed grey = random allocation", fontsize=7, color="#777")

ax2 = axes[1]
sub20 = cv[cv.budget == 0.20].set_index('rule')
rr = ['Model risk score', 'Depression score', 'BMI alone', 'Random allocation',
      'Age alone', 'Baseline condition count']          # 从上到下
vals = [sub20.loc[r, 'capture_rate'] * 100 for r in rr]
ys = list(range(len(rr)))[::-1]                          # 第一条在最上
cols = [CMAP["model"] if r == 'Model risk score' else
        (CMAP["bench"] if r == 'Baseline condition count' else CMAP["random"]) for r in rr]
ax2.barh(ys, vals, color=cols, height=0.65)
for y_, v in zip(ys, vals):
    ax2.text(v + 0.6, y_, "%.1f%%" % v, va="center", fontsize=7.6)
ax2.set_yticks(ys)
ax2.set_yticklabels(rr, fontsize=7.4)
ax2.set_xlabel("Incident cases captured at 20% capacity (%)")
ax2.set_xlim(0, 38); ax2.grid(axis="x", alpha=0.18, lw=0.5)
plt.tight_layout(); save(fig, "Figure12_CapacityConstrainedRanking", 8.4, 3.9)

# ---------------- Figure 13 再校准前后（核心新图） ----------------
fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.6))
for ax, (y, p, tit) in zip(axes, [(y_te, p_te, "Internal test"), (y_va, p_va, "Temporal validation")]):
    pr, ob = calcurve(y, p); pr2, ob2 = calcurve(y, platt(p))
    ax.plot([0, 1], [0, 1], ls="--", lw=0.9, color="#999999")
    ax.plot(pr, ob, "o-", color=CMAP["bench"], ms=4.5, lw=1.3, label="As transported")
    ax.plot(pr2, ob2, "s-", color=CMAP["model"], ms=4.5, lw=1.3, label="After recalibration")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("Predicted probability"); ax.set_ylabel("Observed proportion")
    ax.set_title(tit, fontsize=9.5); ax.legend(frameon=False, fontsize=7.5, loc="upper left")
    ax.grid(alpha=0.18, lw=0.5)
plt.tight_layout(); save(fig, "Figure13_Recalibration", 7.6, 3.6)

# ---------------- Figure S1 ROC ----------------
from sklearn.metrics import roc_curve, roc_auc_score
fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.6))
for ax, (y, p, tit) in zip(axes, [(y_te, p_te, "Internal test"), (y_va, p_va, "Temporal validation")]):
    fpr, tpr, _ = roc_curve(y, p); auc = roc_auc_score(y, p)
    ax.plot(fpr, tpr, color=CMAP["model"], lw=1.5, label="AUC = %.3f" % auc)
    ax.plot([0, 1], [0, 1], ls="--", lw=0.9, color="#999999")
    ax.set_xlabel("1 - specificity"); ax.set_ylabel("Sensitivity")
    ax.set_title(tit, fontsize=9.5); ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.grid(alpha=0.18, lw=0.5)
plt.tight_layout(); save(fig, "FigureS1_ROC", 7.6, 3.6)

# ---------------- Figure S2 分层病例密度 ----------------
st = pd.read_csv(os.path.join(TAB, 'Table5_stratified_allocation.csv'))
svst = st[st['sample'] == 'temporal_validation']
fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.7))
ax = axes[0]
lbl = svst['stratum'].tolist()
ax.bar(range(5), svst['case_density_per_100'], color=[CMAP["model"], "#3d6fa8", "#5f8fc4",
                                                     "#8fb3dd", "#c2d8ef"])
ax.set_xticks(range(5)); ax.set_xticklabels(["Q5\nhighest", "Q4", "Q3", "Q2", "Q1\nlowest"], fontsize=7.8)
ax.set_ylabel("Incident cases per 100 residents screened")
ax.grid(axis="y", alpha=0.18, lw=0.5)
for i, v in enumerate(svst['case_density_per_100']):
    ax.text(i, v + 1, "%.1f" % v, ha="center", fontsize=7.6)
ax2 = axes[1]
ax2.plot(range(1, 6), svst['cumulative_capture'] * 100, "o-", color=CMAP["model"], lw=1.4, ms=5,
         label="Model-directed")
ax2.plot(range(1, 6), np.arange(1, 6) * 20, "s--", color=CMAP["random"], lw=1.2, ms=4,
         label="Random allocation")
ax2.set_xticks(range(1, 6)); ax2.set_xticklabels(["20", "40", "60", "80", "100"], fontsize=8)
ax2.set_xlabel("Cumulative population screened (%)")
ax2.set_ylabel("Cumulative incident cases captured (%)")
ax2.legend(frameon=False, fontsize=7.6, loc="upper left"); ax2.grid(alpha=0.18, lw=0.5)
plt.tight_layout(); save(fig, "FigureS2_StratifiedAllocation", 8.4, 3.7)

print("FIGURES_V4_DONE")
