# -*- coding: utf-8 -*-
"""Figure 1 (study flow + three-stage protocol) and Figure 2 (zero-cost audit,
capacity-constrained ranking, recalibration). No in-figure titles.
Run with MANAGED python (matplotlib)."""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                     "axes.edgecolor": "#333333", "axes.linewidth": 0.8,
                     "svg.fonttype": "none", "ps.fonttype": 42})

BASE = r"D:\02XIHO Files\Daily Work\03Scientific Research\00Article_File\00Student\01单任务论文\PublicHealth_Submission"
V4 = os.path.join(BASE, "_Archive_旧版", "04_中间产物", "Intermediate", "Intermediate", "v4")
V5 = os.path.join(BASE, "04_Data_and_Results", "v5")
FIG = os.path.join(BASE, "02_Figures", "02-2_ResultFigures")
os.makedirs(FIG, exist_ok=True)

D2 = json.load(open(os.path.join(V5, "fig2_data.json"), encoding="utf-8"))
M = json.load(open(os.path.join(BASE, "_Archive_旧版", "09_20260924终清理_重复与旧版",
                                 "04_Data_旧版", "metrics_v4.json"), encoding="utf-8"))

def save3(fig, name):
    fig.savefig(os.path.join(FIG, name + ".png"), dpi=300, facecolor="white", bbox_inches="tight")
    fig.savefig(os.path.join(FIG, name + ".svg"), facecolor="white", bbox_inches="tight")
    try:
        fig.savefig(os.path.join(FIG, name + ".eps"), facecolor="white", bbox_inches="tight")
    except Exception as e:
        print("eps skipped:", e)
    plt.close(fig)
    print("saved", name)

CM = {"model": "#1f4e79", "nat": "#888780", "rev": "#c00000", "rnd": "#7f7f7f",
      "box": "#EEF3F8", "box2": "#FAEEDA", "edge": "#333333"}

# =============================== Figure 1 ===============================
fig, ax = plt.subplots(figsize=(7.2, 6.2))
ax.set_xlim(0, 10); ax.set_ylim(0, 10.6); ax.axis("off")

def box(x0, y0, w, h, text, fc=CM["box"], fs=8.2, weight="normal", ec=CM["edge"]):
    ax.add_patch(FancyBboxPatch((x0, y0), w, h, boxstyle="round,pad=0.06,rounding_size=0.12",
                                fc=fc, ec=ec, lw=0.8, zorder=2))
    ax.text(x0 + w / 2, y0 + h / 2, text, ha="center", va="center", fontsize=fs,
            fontweight=weight, zorder=3)

def arrow(x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=11,
                                 lw=0.9, color=CM["edge"], zorder=1))

box(2.6, 9.5, 4.8, 0.95, "CHARLS waves 2011, 2013, 2015, 2018, 2020\n(self-reported chronic conditions)", fs=8.6)
arrow(5.0, 9.5, 2.45, 8.75)
arrow(5.0, 9.5, 7.55, 8.75)
box(0.30, 7.15, 4.30, 1.55,
    "Derivation cohort\n2011 baseline \u2192 2013 follow-up\nn = 2,638; 598 incident (22.7%)", fs=8.4)
box(5.40, 7.15, 4.30, 1.55,
    "Temporal validation cohort\n2015 baseline \u2192 2018 follow-up\nn = 3,740; 1,705 incident (45.6%)", fs=8.4)
arrow(2.45, 7.15, 4.3, 6.35)
arrow(7.55, 7.15, 5.7, 6.35)
box(0.30, 5.00, 9.40, 1.30,
    "29 routinely collected baseline predictors\n"
    "(15 socio-demographic / behavioural / clinical + 14 baseline conditions)  \u2192  LightGBM (80/20 split)", fs=8.4)
for x in (1.85, 5.0, 8.15):
    arrow(x, 5.00, x, 4.25)
box(0.30, 2.65, 2.95, 1.60,
    "Stage 1\nZero-cost benchmark audit\nmodel vs. rules requiring\nno data collection,\nboth orientations of the\ncount rule", fs=7.6)
box(3.55, 2.65, 2.95, 1.60,
    "Stage 2\nCapacity-constrained ranking\ncase capture at a fixed\nscreening capacity\n(top-20% of the cohort)", fs=7.6)
box(6.75, 2.65, 2.95, 1.60,
    "Stage 3\nTransportability & recalibration\ntemporal drift, calibration\nslope/intercept, offline\nPlatt recalibration", fs=7.6)
arrow(1.775, 2.65, 1.775, 1.95)
arrow(5.025, 2.65, 5.025, 1.95)
arrow(8.225, 2.65, 8.225, 1.95)
box(0.30, 0.45, 9.40, 1.50,
    "Outcome measurement-stability audit\n"
    "within-person change in the reported condition set between adjacent waves;\n"
    "prevalence by wave in balanced cohorts (participants present at every wave)", fc=CM["box2"], fs=8.2)
save3(fig, "Figure1_StudyFlow_Protocol")

# =============================== Figure 2 ===============================
fig, axes = plt.subplots(3, 1, figsize=(6.9, 9.2))
fig.subplots_adjust(hspace=0.45)

# ---- Panel A: AUC comparison ----
ax = axes[0]
groups = [("Internal test", D2["internal_test"]), ("Temporal validation", D2["temporal_validation"])]
labels = [("model", "LightGBM model", CM["model"], "o"),
          ("count_reversed", "Condition count,\nreversed orientation", CM["rev"], "s"),
          ("count_natural", "Condition count,\nnatural orientation", CM["nat"], "^")]
ypos = {"Internal test": 1.0, "Temporal validation": 0.0}
for gi, (gname, vals) in enumerate(groups):
    for yi, (key, lab, c, mk) in enumerate(labels):
        y = ypos[gname] + (1 - yi) * 0.26
        ax.plot(vals[key], y, mk, color=c, ms=7, zorder=3)
        ax.text(vals[key] + 0.012, y, "%.3f" % vals[key], va="center", fontsize=7.5, color=c)
ax.axvline(0.5, color=CM["rnd"], lw=0.9, ls="--", zorder=1)
ax.text(0.5075, 1.30, "random (0.500)", fontsize=7.5, color="#5F5E5A", va="center")
ax.set_yticks([0.0, 1.0]); ax.set_yticklabels(["Temporal\nvalidation\n(n=3,740)", "Internal test\n(n=528)"])
ax.set_xlim(0.28, 0.78); ax.set_ylim(-0.45, 1.80)
ax.set_xlabel("Area under the ROC curve")
short = ["LightGBM model", "Count, reversed", "Count, natural"]
ax.legend(handles=[plt.Line2D([], [], color=c, marker=mk, ls="", label=s, ms=6)
                   for ((_, lab, c, mk), s) in zip(labels, short)],
          loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=3, frameon=False,
          fontsize=7.2, columnspacing=1.0, handletextpad=0.35)
ax.spines[["top", "right", "left"]].set_visible(False)
ax.grid(axis="x", color="#D3D1C7", lw=0.5, zorder=0); ax.set_axisbelow(True)
ax.text(-0.145, 1.03, "A", transform=ax.transAxes, fontsize=11, fontweight="bold", va="top")

# ---- Panel B: capture vs capacity ----
ax = axes[1]
caps = D2["caps"]
ax.plot(caps, D2["curves"]["model"], "-o", color=CM["model"], lw=1.6, ms=4.5,
        label="LightGBM model", zorder=3)
ax.plot(caps, D2["curves"]["count_reversed"], "--s", color=CM["rev"], lw=1.6, ms=4,
        mfc="white", label="Condition count, reversed", zorder=3)
ax.plot(caps, D2["random"], ":", color=CM["rnd"], lw=1.2, label="Random selection", zorder=2)
ax.axvline(0.20, color="#888780", lw=0.8, ls=":", zorder=1)
ax.annotate("at 20%% capacity:\nmodel %.1f%% vs count %.1f%%"
            % (100 * D2["capture_top20"]["model_val"], 100 * D2["capture_top20"]["count_reversed_val"]),
            xy=(0.20, D2["capture_top20"]["model_val"]), xytext=(0.285, 0.52),
            fontsize=7.5, color="#333333",
            arrowprops=dict(arrowstyle="-|>", color="#888780", lw=0.8))
ax.set_xlabel("Screening capacity (proportion of the cohort prioritised)")
ax.set_ylabel("Proportion of incident cases captured")
ax.set_xlim(0.03, 0.52); ax.set_ylim(0, 0.62)
ax.legend(loc="upper left", frameon=False, fontsize=7.5)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(color="#D3D1C7", lw=0.5, zorder=0); ax.set_axisbelow(True)
ax.text(-0.145, 1.03, "B", transform=ax.transAxes, fontsize=11, fontweight="bold", va="top")

# ---- Panel C: calibration before/after ----
ax = axes[2]
cb = np.load(os.path.join(V4, "cal_curve_temporal_validation_before.npy"))
ca = np.load(os.path.join(V4, "cal_curve_temporal_validation_after.npy"))
ax.plot([0, 1], [0, 1], ls="--", color="#888780", lw=0.9, label="Ideal", zorder=1)
ax.plot(cb[0], cb[1], "-o", color="#D85A30", lw=1.5, ms=4, label="Before recalibration", zorder=3)
ax.plot(ca[0], ca[1], "-s", color="#378ADD", lw=1.5, ms=4, label="After Platt recalibration", zorder=3)
rb = M["recalibration"]["temporal_validation"]
ax.text(0.985, 0.16,
        "slope %.2f \u2192 %.2f\nBrier %.3f \u2192 %.3f\nAUC unchanged (%.3f)"
        % (rb["slope_before"], rb["slope_after"], rb["brier_before"], rb["brier_after"], rb["auc_after"]),
        ha="right", va="bottom", fontsize=7.5, color="#333333",
        bbox=dict(fc="#F1EFE8", ec="none", pad=2.5))
ax.set_xlabel("Mean predicted probability")
ax.set_ylabel("Observed event proportion")
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
ax.legend(loc="upper left", frameon=False, fontsize=7.5)
ax.spines[["top", "right"]].set_visible(False)
ax.text(-0.145, 1.03, "C", transform=ax.transAxes, fontsize=11, fontweight="bold", va="top")

save3(fig, "Figure2_ThreeStage_Evaluation")
print("done")
